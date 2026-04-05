"""ColBERT late interaction model for Korean legal case retrieval.

Implements the ColBERT architecture using KoBERT as the backbone:
  - Encodes query tokens and candidate tokens independently through KoBERT.
  - Projects each token representation to a lower-dimensional space (128-dim).
  - Scores via MaxSim: for each query token, find the maximum cosine similarity
    with any candidate token, then sum across all query tokens.
  - Training combines MSE loss with pairwise margin ranking loss, consistent
    with the neural_models.py pattern.

Evaluation uses 5-fold stratified pair-level cross-validation.
Early BERT layers (10/12) are frozen to prevent overfitting.
Mixed-precision (AMP) is enabled throughout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import StratifiedKFold

from src.data_loader import QueryGroup, load_dataset
from src.metrics import evaluate_retrieval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "monologg/kobert"
MAX_LEN = 256
PROJECTION_DIM = 128
EPOCHS = 5
BATCH_SIZE = 8
LR = 2e-5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ColBERTPairDataset(Dataset):
    """Flat dataset of pre-tokenized (query, candidate, label) triples for ColBERT.

    Stores separate tokenizations for query and candidate so that each
    can be independently encoded through the model.
    """

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            s["query_input_ids"],
            s["query_attention_mask"],
            s["candidate_input_ids"],
            s["candidate_attention_mask"],
            torch.tensor(s["label"], dtype=torch.float),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ColBERTEncoder(nn.Module):
    """ColBERT late interaction encoder using KoBERT.

    Encodes query and candidate sequences independently, projects each
    token to a 128-dimensional representation, and scores via MaxSim.
    """

    def __init__(self, model_name: str = MODEL_NAME,
                 projection_dim: int = PROJECTION_DIM):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.projection = nn.Linear(hidden_size, projection_dim)

    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a sequence and project each token to the low-dim space.

        Returns normalized token embeddings of shape (batch, seq_len, projection_dim).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = self.projection(outputs.last_hidden_state)
        return F.normalize(token_embs, p=2, dim=-1)

    def forward(self, query_input_ids: torch.Tensor,
                query_attention_mask: torch.Tensor,
                candidate_input_ids: torch.Tensor,
                candidate_attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute MaxSim scores for a batch of query-candidate pairs."""
        query_embs = self.encode(query_input_ids, query_attention_mask)
        candidate_embs = self.encode(candidate_input_ids, candidate_attention_mask)
        return maxsim(query_embs, candidate_embs,
                      query_attention_mask, candidate_attention_mask)


def maxsim(query_embs: torch.Tensor, candidate_embs: torch.Tensor,
           query_mask: torch.Tensor,
           candidate_mask: torch.Tensor) -> torch.Tensor:
    """MaxSim scoring: late interaction between query and candidate token embeddings.

    For each query token, computes the maximum cosine similarity with any
    non-padding candidate token, then sums over all non-padding query tokens.

    Args:
        query_embs: (batch, q_len, dim) -- normalized query token embeddings.
        candidate_embs: (batch, c_len, dim) -- normalized candidate token embeddings.
        query_mask: (batch, q_len) -- 1 for real tokens, 0 for padding.
        candidate_mask: (batch, c_len) -- 1 for real tokens, 0 for padding.

    Returns:
        scores: (batch,) -- MaxSim score for each pair.
    """
    # Similarity matrix: (batch, q_len, c_len)
    sim_matrix = torch.einsum("bqd,bcd->bqc", query_embs, candidate_embs)

    # Mask out padding positions in the candidate dimension
    # Expand candidate_mask: (batch, 1, c_len) for broadcasting
    candidate_mask_expanded = candidate_mask.unsqueeze(1).float()  # (batch, 1, c_len)
    sim_matrix = sim_matrix * candidate_mask_expanded + (1 - candidate_mask_expanded) * (-1e9)

    # For each query token, take max similarity over candidate tokens
    max_sim_per_query, _ = sim_matrix.max(dim=2)  # (batch, q_len)

    # Mask out padding positions in the query dimension and sum
    query_mask_float = query_mask.float()  # (batch, q_len)
    scores = (max_sim_per_query * query_mask_float).sum(dim=1)  # (batch,)

    return scores


# ---------------------------------------------------------------------------
# Layer freezing
# ---------------------------------------------------------------------------

def _freeze_bert_early_layers(model: ColBERTEncoder) -> None:
    """Freeze embedding layer and first 10 of 12 encoder layers.

    Only the last two encoder layers and the projection head remain
    trainable, preventing overfitting on the small Korean legal case dataset.
    """
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer_idx in range(10):
        for param in model.bert.encoder.layer[layer_idx].parameters():
            param.requires_grad = False


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(scores: torch.Tensor, labels: torch.Tensor,
                          margin: float = 0.3) -> torch.Tensor:
    """Pairwise margin ranking loss for nDCG optimisation.

    For every ordered pair (i, j) where label_i > label_j,
    enforce score_i > score_j + margin * (label_i - label_j).
    """
    n = scores.size(0)
    if n < 2:
        return torch.tensor(0.0, device=scores.device)

    s_i = scores.unsqueeze(1)  # (n, 1)
    s_j = scores.unsqueeze(0)  # (1, n)
    l_i = labels.unsqueeze(1)  # (n, 1)
    l_j = labels.unsqueeze(0)  # (1, n)

    diff_mask = (l_i > l_j).float()
    label_diff = (l_i - l_j).float()

    loss = (F.relu(margin * label_diff - (s_i - s_j)) * diff_mask).sum()
    return loss / diff_mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_colbert(model: ColBERTEncoder, train_loader: DataLoader,
                  epochs: int = EPOCHS) -> ColBERTEncoder:
    """Train the ColBERT model with combined MSE + pairwise ranking loss.

    MSE teaches absolute label prediction; ranking loss teaches ordering.
    Mixed-precision is used throughout for memory efficiency.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01,
    )
    scaler = GradScaler("cuda")
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for q_ids, q_mask, c_ids, c_mask, labels in train_loader:
            q_ids = q_ids.to(DEVICE)
            q_mask = q_mask.to(DEVICE)
            c_ids = c_ids.to(DEVICE)
            c_mask = c_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                scores = model(q_ids, q_mask, c_ids, c_mask)
                mse_loss = F.mse_loss(scores, labels)
                rank_loss = pairwise_ranking_loss(scores, labels)
                loss = mse_loss + rank_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)

    return model


# ---------------------------------------------------------------------------
# Pre-tokenization
# ---------------------------------------------------------------------------

def pretokenize_all_pairs(groups: list[QueryGroup], tokenizer: BertTokenizer,
                          max_len: int = MAX_LEN) -> list[list[dict]]:
    """Pre-tokenize all query-candidate pairs independently for ColBERT.

    Returns a list of lists (one per group), each containing dicts with
    separate query/candidate input_ids, attention_mask, and label.
    """
    print("  Pre-tokenizing all pairs...", flush=True)
    all_group_data: list[list[dict]] = []
    total = sum(len(g.pairs) for g in groups)
    done = 0

    for g in groups:
        group_data: list[dict] = []
        q_enc = tokenizer(
            g.query_note,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        for pair in g.pairs:
            c_enc = tokenizer(
                pair.candidate_note,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            group_data.append({
                "query_input_ids": q_enc["input_ids"].squeeze(0),
                "query_attention_mask": q_enc["attention_mask"].squeeze(0),
                "candidate_input_ids": c_enc["input_ids"].squeeze(0),
                "candidate_attention_mask": c_enc["attention_mask"].squeeze(0),
                "label": pair.label,
            })
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{total} pairs tokenized", flush=True)
        all_group_data.append(group_data)

    print(f"    {total}/{total} pairs tokenized (done)", flush=True)
    return all_group_data


# ---------------------------------------------------------------------------
# 5-fold cross-validated evaluation
# ---------------------------------------------------------------------------

def run_colbert(
    groups: list[QueryGroup],
) -> tuple[list[list[int]], list[list[float]]]:
    """Run ColBERT with 5-fold stratified pair-level CV.

    Steps
    -----
    1. Pre-tokenize all query-candidate pairs (done once).
    2. Flatten pairs with group-index tracking.
    3. Run 5-fold StratifiedKFold on the flattened pairs.
    4. Per fold: train a fresh ColBERT model, score held-out pairs via MaxSim.
    5. Reassemble per-query label / score lists.
    """
    print("Loading tokenizer...", flush=True)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    all_group_data = pretokenize_all_pairs(groups, tokenizer)

    # -- Flatten all pairs with group-index tracking --------------------------
    flat_items: list[dict] = []
    flat_labels: list[int] = []
    flat_group_idx: list[int] = []

    for gi, gd in enumerate(all_group_data):
        for item in gd:
            flat_items.append(item)
            flat_labels.append(item["label"])
            flat_group_idx.append(gi)

    # -- 5-fold stratified CV -------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred_scores: list[float | None] = [None] * len(flat_items)

    for fold, (train_idx, test_idx) in enumerate(skf.split(flat_items, flat_labels)):
        print(f"\n  Fold {fold + 1}/5 (train={len(train_idx)}, test={len(test_idx)})",
              flush=True)

        train_data = [flat_items[i] for i in train_idx]
        train_dataset = ColBERTPairDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = ColBERTEncoder().to(DEVICE)
        _freeze_bert_early_layers(model)
        model = train_colbert(model, train_loader)

        # -- Score held-out pairs via MaxSim ----------------------------------
        model.eval()
        with torch.no_grad():
            for idx in test_idx:
                item = flat_items[idx]
                with autocast("cuda"):
                    score = model(
                        item["query_input_ids"].unsqueeze(0).to(DEVICE),
                        item["query_attention_mask"].unsqueeze(0).to(DEVICE),
                        item["candidate_input_ids"].unsqueeze(0).to(DEVICE),
                        item["candidate_attention_mask"].unsqueeze(0).to(DEVICE),
                    ).item()
                pred_scores[idx] = score

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- Reassemble per-query results -----------------------------------------
    all_labels: list[list[int]] = [[] for _ in groups]
    all_scores: list[list[float]] = [[] for _ in groups]
    for fi in range(len(flat_items)):
        gi = flat_group_idx[fi]
        all_labels[gi].append(flat_labels[fi])
        all_scores[gi].append(pred_scores[fi])

    return all_labels, all_scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point: load data, run ColBERT CV, and print results."""
    print("Loading dataset...", flush=True)
    groups = load_dataset()
    print(f"Using device: {DEVICE}", flush=True)

    print(f"\n{'=' * 60}")
    print("Running ColBERT...")
    print("=" * 60, flush=True)

    all_labels, all_scores = run_colbert(groups)
    metrics = evaluate_retrieval(all_labels, all_scores)

    print("\n=== ColBERT Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    return metrics


if __name__ == "__main__":
    main()

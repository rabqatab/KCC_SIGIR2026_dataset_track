"""Ordinal Contrastive Ranking Network for Korean legal case retrieval.

Exploits the 4-level graded relevance labels (0-3) with ordinal-aware
contrastive learning.  Instead of treating labels as binary positive/negative,
the loss enforces margins proportional to label differences:
    sim(q, label3) > sim(q, label2) > sim(q, label1) > sim(q, label0)

Architecture:
  - KoBERT bi-encoder with mean pooling + L2 normalization (like SBERT).
  - Combined loss = InfoNCE (in-batch negatives) + ordinal margin loss.
  - Scoring via cosine similarity between query and candidate embeddings.

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
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
ORDINAL_ALPHA = 0.1
ORDINAL_LAMBDA = 1.0
TEMPERATURE = 0.05


# ---------------------------------------------------------------------------
# Pre-tokenization
# ---------------------------------------------------------------------------

def pretokenize_single_texts(groups, tokenizer, max_len=MAX_LEN):
    """Pre-tokenize query and candidate notes independently.

    Returns a list of lists, one per group, each containing dicts with
    separate tokenizations for the query and candidate, plus the label.
    """
    print("  Pre-tokenizing query and candidate texts...", flush=True)
    all_group_data = []
    total = sum(len(g.pairs) for g in groups)
    done = 0

    for g in groups:
        group_data = []
        q_enc = tokenizer(
            g.query_note,
            max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        q_ids = q_enc["input_ids"].squeeze(0)
        q_mask = q_enc["attention_mask"].squeeze(0)
        q_type = q_enc["token_type_ids"].squeeze(0)

        for pair in g.pairs:
            c_enc = tokenizer(
                pair.candidate_note,
                max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            group_data.append({
                "q_input_ids": q_ids,
                "q_attention_mask": q_mask,
                "q_token_type_ids": q_type,
                "c_input_ids": c_enc["input_ids"].squeeze(0),
                "c_attention_mask": c_enc["attention_mask"].squeeze(0),
                "c_token_type_ids": c_enc["token_type_ids"].squeeze(0),
                "label": pair.label,
            })
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{total} pairs tokenized", flush=True)

        all_group_data.append(group_data)

    print(f"    {total}/{total} pairs tokenized (done)", flush=True)
    return all_group_data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OCRNDataset(Dataset):
    """Dataset wrapping pre-tokenized query-candidate pairs for OCRN training.

    Returns tokenized query and candidate tensors together with the
    graded relevance label (0-3) needed for the ordinal contrastive loss.
    """

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return (
            item["q_input_ids"],
            item["q_attention_mask"],
            item["q_token_type_ids"],
            item["c_input_ids"],
            item["c_attention_mask"],
            item["c_token_type_ids"],
            torch.tensor(item["label"], dtype=torch.float),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class OCRNEncoder(nn.Module):
    """Ordinal Contrastive Ranking Network encoder.

    KoBERT bi-encoder with mean pooling over non-padded token
    representations, followed by L2 normalization.  Cosine similarity
    between query and candidate vectors serves as the relevance score.
    """

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def _mean_pool(self, last_hidden_state, attention_mask):
        """Mean-pool token embeddings, masking out padding tokens."""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        return summed / lengths

    def encode(self, input_ids, attention_mask, token_type_ids):
        """Encode a batch of texts into L2-normalized embeddings."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def forward(self, q_ids, q_mask, q_type, c_ids, c_mask, c_type):
        """Encode query and candidate batches, return both embedding tensors.

        Returns:
            q_emb: (batch, hidden) L2-normalized query embeddings.
            c_emb: (batch, hidden) L2-normalized candidate embeddings.
        """
        q_emb = self.encode(q_ids, q_mask, q_type)
        c_emb = self.encode(c_ids, c_mask, c_type)
        return q_emb, c_emb


# ---------------------------------------------------------------------------
# Layer freezing
# ---------------------------------------------------------------------------

def _freeze_bert_early_layers(model):
    """Freeze embedding layer and first 10 of 12 encoder layers.

    Only the last two encoder layers remain trainable, along with
    any task-specific parameters.  This prevents overfitting when
    fine-tuning on the small Korean legal case dataset.
    """
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer_idx in range(10):
        for param in model.bert.encoder.layer[layer_idx].parameters():
            param.requires_grad = False


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def infonce_loss(q_emb, c_emb, temperature=TEMPERATURE):
    """InfoNCE contrastive loss with in-batch negatives.

    The similarity matrix is queries @ candidates.T.  Diagonal entries
    are the positive pairs (each query matches its own candidate).
    Cross-entropy over this matrix with arange labels implements InfoNCE.
    """
    sim_matrix = q_emb @ c_emb.T / temperature
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    return F.cross_entropy(sim_matrix, labels)


def ordinal_margin_loss(q_emb, c_emb, labels, alpha=ORDINAL_ALPHA):
    """Ordinal contrastive margin loss exploiting graded relevance labels.

    For every ordered pair (i, j) in the batch where label_i > label_j,
    enforces that the query-candidate similarity for the higher-relevance
    item exceeds the lower-relevance item by a margin proportional to the
    label difference:

        loss += max(0, alpha * (label_i - label_j) - (sim_i - sim_j))

    This encourages the model to produce rankings that respect the full
    ordinal structure of the 4-level grading scheme.
    """
    # Per-pair cosine similarities: (batch,)
    sims = (q_emb * c_emb).sum(dim=1)
    n = sims.size(0)
    if n < 2:
        return torch.tensor(0.0, device=sims.device)

    # Pairwise differences
    s_i = sims.unsqueeze(1)       # (n, 1)
    s_j = sims.unsqueeze(0)       # (1, n)
    l_i = labels.unsqueeze(1)     # (n, 1)
    l_j = labels.unsqueeze(0)     # (1, n)

    # Only consider pairs where label_i > label_j
    diff_mask = (l_i > l_j).float()
    label_diff = (l_i - l_j).float()

    margins = alpha * label_diff
    violations = F.relu(margins - (s_i - s_j)) * diff_mask

    num_pairs = diff_mask.sum().clamp(min=1)
    return violations.sum() / num_pairs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_ocrn(model, train_loader, epochs=EPOCHS, alpha=ORDINAL_ALPHA,
                lam=ORDINAL_LAMBDA):
    """Train OCRN with combined InfoNCE + ordinal margin loss.

    The total loss is:
        L = L_InfoNCE + lambda * L_ordinal_margin

    InfoNCE provides strong base representation learning through in-batch
    negatives, while the ordinal margin term fine-tunes the embedding
    space to respect graded relevance ordering.  Mixed precision is used
    throughout for memory efficiency.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01,
    )
    scaler = GradScaler("cuda")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_info = 0.0
        total_ord = 0.0
        n_batches = 0

        for batch in train_loader:
            q_ids, q_mask, q_type, c_ids, c_mask, c_type, labels = [
                b.to(DEVICE) for b in batch
            ]
            optimizer.zero_grad()

            with autocast("cuda"):
                q_emb, c_emb = model(q_ids, q_mask, q_type,
                                     c_ids, c_mask, c_type)

                loss_info = infonce_loss(q_emb, c_emb)
                loss_ord = ordinal_margin_loss(q_emb, c_emb, labels, alpha)
                loss = loss_info + lam * loss_ord

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_info += loss_info.item()
            total_ord += loss_ord.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_info = total_info / max(n_batches, 1)
        avg_ord = total_ord / max(n_batches, 1)
        print(
            f"    Epoch {epoch + 1}/{epochs}, "
            f"Loss: {avg_loss:.4f} "
            f"(InfoNCE: {avg_info:.4f}, Ordinal: {avg_ord:.4f})",
            flush=True,
        )

    return model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_pairs(model, items):
    """Compute cosine similarity scores for held-out query-candidate pairs.

    Each pair is encoded independently; the score is the dot product
    of the two L2-normalized embeddings (i.e. cosine similarity).
    """
    scores = []
    model.eval()
    with torch.no_grad():
        for item in items:
            with autocast("cuda"):
                q_emb = model.encode(
                    item["q_input_ids"].unsqueeze(0).to(DEVICE),
                    item["q_attention_mask"].unsqueeze(0).to(DEVICE),
                    item["q_token_type_ids"].unsqueeze(0).to(DEVICE),
                )
                c_emb = model.encode(
                    item["c_input_ids"].unsqueeze(0).to(DEVICE),
                    item["c_attention_mask"].unsqueeze(0).to(DEVICE),
                    item["c_token_type_ids"].unsqueeze(0).to(DEVICE),
                )
            score = (q_emb * c_emb).sum().item()
            scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# 5-fold cross-validated evaluation
# ---------------------------------------------------------------------------

def run_ocrn(groups):
    """Run OCRN with 5-fold pair-level stratified cross-validation.

    Steps:
        1. Pre-tokenize all query and candidate texts independently.
        2. Flatten pairs with flat_group_idx tracking for reassembly.
        3. Stratified 5-fold split on graded relevance labels.
        4. Per fold: train a fresh OCRNEncoder with combined InfoNCE +
           ordinal margin loss, predict held-out pairs via cosine similarity.
        5. Reassemble per-query results for retrieval evaluation.

    Returns:
        all_labels: list of label lists, one per query group.
        all_scores: list of score lists, one per query group.
    """
    print("Loading tokenizer...", flush=True)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    all_group_data = pretokenize_single_texts(groups, tokenizer)

    # Flatten pairs with group tracking
    flat_items = []
    flat_labels = []
    flat_group_idx = []
    for gi, gd in enumerate(all_group_data):
        for item in gd:
            flat_items.append(item)
            flat_labels.append(item["label"])
            flat_group_idx.append(gi)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred_scores = [None] * len(flat_items)

    for fold, (train_idx, test_idx) in enumerate(skf.split(flat_items, flat_labels)):
        print(
            f"\n  Fold {fold + 1}/5 "
            f"(train={len(train_idx)}, test={len(test_idx)})",
            flush=True,
        )

        train_items = [flat_items[i] for i in train_idx]
        train_dataset = OCRNDataset(train_items)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        )

        model = OCRNEncoder().to(DEVICE)
        _freeze_bert_early_layers(model)
        model = _train_ocrn(model, train_loader, epochs=EPOCHS)

        test_items = [flat_items[i] for i in test_idx]
        fold_scores = _score_pairs(model, test_items)
        for idx, score in zip(test_idx, fold_scores):
            pred_scores[idx] = score

        del model
        torch.cuda.empty_cache()

    # Reassemble per-query results
    all_labels = []
    all_scores = []
    for gi in range(len(groups)):
        g_labels = []
        g_scores = []
        for fi in range(len(flat_items)):
            if flat_group_idx[fi] == gi:
                g_labels.append(flat_labels[fi])
                g_scores.append(pred_scores[fi])
        all_labels.append(g_labels)
        all_scores.append(g_scores)

    return all_labels, all_scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run OCRN and print retrieval metrics."""
    print("Loading dataset...", flush=True)
    groups = load_dataset()
    print(f"Using device: {DEVICE}", flush=True)

    print(f"\n{'=' * 60}")
    print("Running OCRN (Ordinal Contrastive Ranking Network)...")
    print("=" * 60, flush=True)

    labels, scores = run_ocrn(groups)
    metrics = evaluate_retrieval(labels, scores)

    print("\n=== OCRN Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    return {"OCRN": metrics}


if __name__ == "__main__":
    main()

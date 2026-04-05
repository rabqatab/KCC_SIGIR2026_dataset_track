"""LCube cross-encoder for Korean legal case retrieval.

Uses lbox/lcube-base (GPT-2 architecture, pretrained on a Korean legal corpus) as a cross-encoder.
Query-candidate pairs are concatenated and mean-pooled hidden states are mapped to a scalar relevance score via a linear head.
Training combines MSE and pairwise margin ranking loss.

Evaluation follows 5-fold stratified pair-level cross-validation (StratifiedKFold, n_splits=5, shuffle=True, random_state=42),
consistent with the neural and BERT baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold

from src.data_loader import QueryGroup, load_dataset
from src.metrics import evaluate_retrieval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "lbox/lcube-base"
MAX_LEN = 512
EPOCHS = 2
BATCH_SIZE = 8
LR = 2e-5


class LCubeDataset(Dataset):
    """Flat dataset of pre-tokenized (input_ids, attention_mask, label) triples."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            s["input_ids"],
            s["attention_mask"],
            torch.tensor(s["label"], dtype=torch.float),
        )


class LCubeCrossEncoder(nn.Module):
    """LCube (GPT-2) cross-encoder for case similarity scoring.

    The GPT-2 model produces contextualised representations for the concatenated query-candidate input.
    Mean pooling (masked) yields a fixed-size vector, which a linear head maps to a scalar score.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "encoder"):
            outputs = self.encoder.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden = outputs.last_hidden_state

        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (
            (hidden * mask_expanded).sum(dim=1)
            / mask_expanded.sum(dim=1).clamp(min=1e-9)
        )
        return self.regressor(pooled).squeeze(-1)


def pairwise_ranking_loss(scores: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Pairwise margin ranking loss for nDCG optimisation.

    For every ordered pair (i, j) where label_i > label_j, 
    enforce score_i > score_j + margin * (label_i - label_j).
    """
    n = scores.size(0)
    if n < 2:
        return torch.tensor(0.0, device=scores.device)

    s_i = scores.unsqueeze(1)
    s_j = scores.unsqueeze(0)
    l_i = labels.unsqueeze(1)
    l_j = labels.unsqueeze(0)

    diff_mask = (l_i > l_j).float()
    label_diff = (l_i - l_j).float()

    loss = (F.relu(margin * label_diff - (s_i - s_j)) * diff_mask).sum()
    return loss / diff_mask.sum().clamp(min=1)


def train_lcube(model: LCubeCrossEncoder, train_loader: DataLoader,
                epochs: int = EPOCHS) -> LCubeCrossEncoder:
    """Train the LCube cross-encoder with MSE + pairwise ranking loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            scores = model(input_ids, attention_mask)
            mse = F.mse_loss(scores, labels)
            rank = pairwise_ranking_loss(scores, labels)
            loss = mse + rank
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)

    return model


def pretokenize_all_pairs(
    groups: list[QueryGroup],
    tokenizer,
    max_len: int = MAX_LEN,
) -> list[list[dict]]:
    """Pre-tokenize all query-candidate pairs once.

    Returns a list of lists (one per group), each containing dicts with input_ids, attention_mask, and label.
    """
    print("  Pre-tokenizing all pairs...", flush=True)
    all_group_data: list[list[dict]] = []
    total = sum(len(g.pairs) for g in groups)
    done = 0

    for g in groups:
        group_data: list[dict] = []
        for pair in g.pairs:
            text = f"{g.query_note} </s> {pair.candidate_note}"
            encoding = tokenizer(
                text,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            group_data.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "label": pair.label,
            })
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{total} pairs tokenized", flush=True)
        all_group_data.append(group_data)

    print(f"    {total}/{total} pairs tokenized (done)", flush=True)
    return all_group_data


def run_lcube_ce(
    groups: list[QueryGroup],
) -> tuple[list[list[int]], list[list[float]]]:
    """Run LCube cross-encoder with 5-fold stratified pair-level CV.

    Steps
    -----
    1. Pre-tokenize all query-candidate pairs once.
    2. Flatten pairs with group-index tracking.
    3. Run 5-fold StratifiedKFold on the flattened pairs.
    4. Reassemble per-query label / score lists after all folds.
    """
    print("Loading LCube tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_group_data = pretokenize_all_pairs(groups, tokenizer)

    # -- flatten all pairs with group-index tracking --------------------------
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
        print(
            f"\n  Fold {fold + 1}/5 "
            f"(train={len(train_idx)}, test={len(test_idx)})",
            flush=True,
        )

        train_data = [flat_items[i] for i in train_idx]
        train_dataset = LCubeDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = LCubeCrossEncoder().to(DEVICE)
        model = train_lcube(model, train_loader)

        # -- score held-out pairs ---------------------------------------------
        model.eval()
        with torch.no_grad():
            for idx in test_idx:
                item = flat_items[idx]
                score = model(
                    item["input_ids"].unsqueeze(0).to(DEVICE),
                    item["attention_mask"].unsqueeze(0).to(DEVICE),
                ).item()
                pred_scores[idx] = score

        del model
        torch.cuda.empty_cache()

    # -- reassemble per-query results -----------------------------------------
    all_labels: list[list[int]] = [[] for _ in groups]
    all_scores: list[list[float]] = [[] for _ in groups]
    for fi in range(len(flat_items)):
        gi = flat_group_idx[fi]
        all_labels[gi].append(flat_labels[fi])
        all_scores[gi].append(pred_scores[fi])

    return all_labels, all_scores


def main():
    """Entry point: load data, run LCube CV, and print results."""
    print("Loading dataset...")
    groups = load_dataset()
    print(f"Using device: {DEVICE}")

    print(f"\n{'=' * 60}")
    print("Running LCube (CE)...")
    print("=" * 60)

    all_labels, all_scores = run_lcube_ce(groups)
    metrics = evaluate_retrieval(all_labels, all_scores)

    print("\n=== LCube (CE) Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    return metrics


if __name__ == "__main__":
    main()

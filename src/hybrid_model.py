"""BM25 + BERT Cross-Encoder hybrid re-ranking for Korean legal case retrieval.

Stage 1 uses unsupervised BM25 to score all candidates per query.
Stage 2 trains a BERT cross-encoder via 5-fold CV and re-scores the
top-K candidates (by BM25 rank) for each query.  Candidates outside
the top-K keep their BM25 score so they always rank below the
re-ranked set.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, BertModel
from sklearn.model_selection import StratifiedKFold
from rank_bm25 import BM25Okapi

from src.bm25_baseline import tokenize_korean
from src.data_loader import QueryGroup, load_dataset
from src.metrics import evaluate_retrieval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "klue/bert-base"
MAX_LEN = 512
NUM_LABELS = 4
EPOCHS = 5
BATCH_SIZE = 16
LR = 5e-5
RERANK_K = 20


# ── BM25 scoring ───────────────────────────────────────────────────────


def compute_bm25_scores(
    groups: list[QueryGroup],
) -> list[list[float]]:
    """Compute BM25 scores for every query-candidate pair.

    Each query group gets its own BM25 index built from its candidate
    notes, and the query note is scored against all candidates.
    Returns one score list per group, aligned with ``group.pairs``.
    """
    all_scores: list[list[float]] = []

    for group in groups:
        candidate_tokens = [tokenize_korean(n) for n in group.candidate_notes]
        bm25 = BM25Okapi(candidate_tokens)
        query_tokens = tokenize_korean(group.query_note)
        scores = bm25.get_scores(query_tokens).tolist()
        all_scores.append(scores)

    return all_scores


# ── BERT cross-encoder ──────────────────────────────────────────────────


class BertCrossEncoder(nn.Module):
    """4-class BERT cross-encoder for graded relevance scoring."""

    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, num_labels),
        )
        self.label_weights = torch.arange(num_labels, dtype=torch.float)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Return raw logits over the 4 relevance classes."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

    def predict_score(self, input_ids, attention_mask, token_type_ids):
        """Expected relevance value from softmax class probabilities."""
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        probs = F.softmax(logits, dim=-1)
        weights = self.label_weights.to(probs.device)
        return (probs * weights).sum(dim=-1)


# ── Dataset ─────────────────────────────────────────────────────────────


class PreTokenizedDataset(Dataset):
    """Wraps a flat list of pre-tokenized dicts for DataLoader."""

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return (
            item["input_ids"],
            item["attention_mask"],
            item["token_type_ids"],
            torch.tensor(item["label"], dtype=torch.float),
        )


# ── Helpers ─────────────────────────────────────────────────────────────


def pretokenize_all_pairs(groups, tokenizer):
    """Pre-tokenize every query-candidate pair for BERT.

    Returns a nested list: ``all_group_data[group_idx][pair_idx]`` is a
    dict with ``input_ids``, ``attention_mask``, ``token_type_ids``, and
    ``label``.
    """
    print("  Pre-tokenizing all pairs for hybrid model...", flush=True)
    all_group_data: list[list[dict]] = []
    total = sum(len(g.pairs) for g in groups)
    done = 0

    for g in groups:
        group_data: list[dict] = []
        for pair in g.pairs:
            enc = tokenizer(
                g.query_note,
                pair.candidate_note,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            group_data.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "token_type_ids": enc["token_type_ids"].squeeze(0),
                "label": pair.label,
            })
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{total} pairs tokenized", flush=True)

        all_group_data.append(group_data)

    print(f"    {total}/{total} pairs tokenized (done)", flush=True)
    return all_group_data


def _freeze_bert_early_layers(model: BertCrossEncoder) -> None:
    """Freeze embeddings and the first 10 of 12 encoder layers."""
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer_idx in range(10):
        for param in model.bert.encoder.layer[layer_idx].parameters():
            param.requires_grad = False


def _train_cross_encoder(model, train_loader):
    """Train the cross-encoder with mixed-precision AMP.

    Uses cross-entropy with label smoothing (0.1) and AdamW.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = GradScaler("cuda")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        n_batches = 0

        for input_ids, attn_mask, type_ids, labels in train_loader:
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            type_ids = type_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(input_ids, attn_mask, type_ids)
                loss = F.cross_entropy(logits, labels.long(), label_smoothing=0.1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"    Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}", flush=True)

    return model


# ── Main pipeline ───────────────────────────────────────────────────────


def run_hybrid(
    groups: list[QueryGroup],
) -> tuple[list[list[int]], list[list[float]]]:
    """BM25 + BERT cross-encoder hybrid with 5-fold stratified CV.

    1. Compute BM25 scores for all pairs (unsupervised, no CV needed).
    2. Pre-tokenize all pairs for the BERT cross-encoder.
    3. Flatten pairs with a group-index tracker.
    4. Run 5-fold stratified CV to train the cross-encoder on training
       pairs and predict on held-out pairs.
    5. For each held-out query group, if a pair is among the top-K
       candidates by BM25 score, replace its score with the BERT
       prediction; otherwise keep the BM25 score.
    6. Reassemble per-query results.

    Returns ``(all_labels, all_scores)`` aligned by query group.
    """
    # ── 1. BM25 scores (unsupervised, computed once) ────────────────
    print("Computing BM25 scores for all groups...", flush=True)
    bm25_scores = compute_bm25_scores(groups)

    # ── 2. Pre-tokenize for BERT ────────────────────────────────────
    print("Loading BERT tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    all_group_data = pretokenize_all_pairs(groups, tokenizer)

    # ── 3. Flatten with group-index tracking ────────────────────────
    flat_items: list[dict] = []
    flat_labels: list[int] = []
    flat_group_idx: list[int] = []
    flat_pair_idx: list[int] = []  # position within its group

    for gi, gd in enumerate(all_group_data):
        for pi, item in enumerate(gd):
            flat_items.append(item)
            flat_labels.append(item["label"])
            flat_group_idx.append(gi)
            flat_pair_idx.append(pi)

    # ── 4. 5-fold stratified CV over BERT cross-encoder ─────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bert_scores: list[float | None] = [None] * len(flat_items)

    for fold, (train_idx, test_idx) in enumerate(skf.split(flat_items, flat_labels)):
        print(
            f"\n  Fold {fold + 1}/5 (train={len(train_idx)}, test={len(test_idx)})",
            flush=True,
        )

        # Build training set
        train_data = [flat_items[i] for i in train_idx]
        train_dataset = PreTokenizedDataset(train_data)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        )

        # Train a fresh cross-encoder
        model = BertCrossEncoder().to(DEVICE)
        _freeze_bert_early_layers(model)
        model = _train_cross_encoder(model, train_loader)

        # Predict on held-out pairs
        model.eval()
        with torch.no_grad():
            for idx in test_idx:
                item = flat_items[idx]
                with autocast("cuda"):
                    score = model.predict_score(
                        item["input_ids"].unsqueeze(0).to(DEVICE),
                        item["attention_mask"].unsqueeze(0).to(DEVICE),
                        item["token_type_ids"].unsqueeze(0).to(DEVICE),
                    ).item()
                bert_scores[idx] = score

        del model
        torch.cuda.empty_cache()

    # ── 5 & 6. Merge BM25 + BERT scores per query group ────────────
    all_labels: list[list[int]] = []
    all_scores: list[list[float]] = []

    for gi in range(len(groups)):
        g_labels: list[int] = []
        g_bert: list[float | None] = []
        g_pair_indices: list[int] = []

        for fi in range(len(flat_items)):
            if flat_group_idx[fi] == gi:
                g_labels.append(flat_labels[fi])
                g_bert.append(bert_scores[fi])
                g_pair_indices.append(flat_pair_idx[fi])

        g_bm25 = bm25_scores[gi]

        # Identify top-K candidates by BM25 score within this group
        n_candidates = len(g_bm25)
        topk_set = set(
            int(i)
            for i in np.argsort(g_bm25)[::-1][:min(RERANK_K, n_candidates)]
        )

        # Assign final scores
        g_scores: list[float] = []
        for local_pos, pair_idx in enumerate(g_pair_indices):
            if pair_idx in topk_set:
                # Use BERT cross-encoder score for top-K candidates
                g_scores.append(g_bert[local_pos])
            else:
                # Keep a very low score so non-reranked items rank below
                g_scores.append(-1.0)

        all_labels.append(g_labels)
        all_scores.append(g_scores)

    return all_labels, all_scores


# ── Standalone entry point ──────────────────────────────────────────────


def main() -> None:
    """Run the hybrid BM25 + BERT re-ranking pipeline and print results."""
    print("Loading dataset...", flush=True)
    groups = load_dataset()
    print(f"Loaded {len(groups)} query groups, using device: {DEVICE}", flush=True)

    print(f"\n{'=' * 60}")
    print("Running BM25 + BERT Hybrid Re-Ranker...")
    print(f"{'=' * 60}", flush=True)

    all_labels, all_scores = run_hybrid(groups)
    results = evaluate_retrieval(all_labels, all_scores)

    print(f"\n=== BM25 + BERT Hybrid Results (RERANK_K={RERANK_K}) ===")
    for key, value in results.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()

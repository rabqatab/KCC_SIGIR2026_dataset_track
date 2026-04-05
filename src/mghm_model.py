"""Multi-Granularity Hierarchical Matching (MGHM) for Korean legal case retrieval.

Matches query and candidate at three granularity levels simultaneously,
then fuses the signals through a learned linear gate:

  1. **Keyword-level** (noun overlap): Jaccard similarity and weighted
     term-overlap computed from Kiwi-extracted nouns.  Two scalar features
     per pair, computed once with no gradients.
  2. **Sentence-level** (bi-encoder): KLUE-BERT encodes query and candidate
     independently with mean pooling; cosine similarity yields one scalar.
  3. **Document-level** (cross-encoder): KLUE-BERT cross-encoder produces
     4-class logits whose expected value is the relevance score; one scalar.

The four features are concatenated and passed through a Linear(4, 1) gate
that learns which granularity matters most.  The bi-encoder and
cross-encoder share the same KLUE-BERT backbone but use it differently
(mean pooling vs. [CLS] classification).

Training minimises MSE on labels plus a pairwise margin-ranking loss.
The first 10 of 12 BERT layers are frozen.  Mixed-precision (AMP) is
enabled throughout.

Evaluation uses 5-fold stratified pair-level cross-validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, BertModel
from sklearn.model_selection import StratifiedKFold
from kiwipiepy import Kiwi

from src.data_loader import QueryGroup, load_dataset
from src.metrics import evaluate_retrieval

_kiwi = Kiwi()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "klue/bert-base"
MAX_LEN = 256
NUM_LABELS = 4
EPOCHS = 5
BATCH_SIZE = 8
LR = 2e-5


# ── Keyword feature extraction ────────────────────────────────────────


def extract_nouns(text: str) -> list[str]:
    """Extract nouns from Korean text using the module-level Kiwi instance."""
    return [t.form for t in _kiwi.tokenize(text) if t.tag.startswith("N")]


def _keyword_features(query_nouns: list[str], cand_nouns: list[str]) -> tuple[float, float]:
    """Compute Jaccard similarity and weighted term overlap.

    Returns:
        jaccard: |intersection| / |union|, or 0 if both empty.
        term_overlap: fraction of query nouns found in the candidate,
            weighted by 1/|candidate nouns| to penalise very long
            documents that trivially contain many terms.
    """
    q_set = set(query_nouns)
    c_set = set(cand_nouns)

    if not q_set and not c_set:
        return 0.0, 0.0

    intersection = q_set & c_set
    union = q_set | c_set
    jaccard = len(intersection) / len(union) if union else 0.0

    if not q_set or not c_set:
        return jaccard, 0.0

    overlap_count = sum(1 for n in query_nouns if n in c_set)
    term_overlap = overlap_count / len(query_nouns)

    return jaccard, term_overlap


# ── Pre-tokenization ──────────────────────────────────────────────────


def pretokenize_mghm_pairs(groups, tokenizer, max_len=MAX_LEN):
    """Pre-tokenize all pairs for the three granularity branches.

    For each query-candidate pair produces:
      - ``kw_jaccard``, ``kw_overlap``: keyword features (floats).
      - ``q_*`` / ``c_*``: separate bi-encoder tokenizations.
      - ``ce_*``: joint cross-encoder tokenization.
      - ``label``: graded relevance (0--3).
    """
    print("  Pre-tokenizing MGHM pairs (keyword + bi + cross)...", flush=True)
    all_group_data: list[list[dict]] = []
    total = sum(len(g.pairs) for g in groups)
    done = 0

    for g in groups:
        group_data: list[dict] = []
        query_nouns = extract_nouns(g.query_note)

        # Bi-encoder: tokenize query once per group
        q_enc = tokenizer(
            g.query_note,
            max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        q_ids = q_enc["input_ids"].squeeze(0)
        q_mask = q_enc["attention_mask"].squeeze(0)
        q_type = q_enc["token_type_ids"].squeeze(0)

        for pair in g.pairs:
            # Keyword features
            cand_nouns = extract_nouns(pair.candidate_note)
            jaccard, overlap = _keyword_features(query_nouns, cand_nouns)

            # Bi-encoder: candidate tokenization
            c_enc = tokenizer(
                pair.candidate_note,
                max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )

            # Cross-encoder: joint tokenization
            ce_enc = tokenizer(
                g.query_note, pair.candidate_note,
                max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )

            group_data.append({
                # Keyword (no gradient)
                "kw_jaccard": jaccard,
                "kw_overlap": overlap,
                # Bi-encoder query
                "q_input_ids": q_ids,
                "q_attention_mask": q_mask,
                "q_token_type_ids": q_type,
                # Bi-encoder candidate
                "c_input_ids": c_enc["input_ids"].squeeze(0),
                "c_attention_mask": c_enc["attention_mask"].squeeze(0),
                "c_token_type_ids": c_enc["token_type_ids"].squeeze(0),
                # Cross-encoder
                "ce_input_ids": ce_enc["input_ids"].squeeze(0),
                "ce_attention_mask": ce_enc["attention_mask"].squeeze(0),
                "ce_token_type_ids": ce_enc["token_type_ids"].squeeze(0),
                # Label
                "label": pair.label,
            })
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{total} pairs tokenized", flush=True)

        all_group_data.append(group_data)

    print(f"    {total}/{total} pairs tokenized (done)", flush=True)
    return all_group_data


# ── Dataset ───────────────────────────────────────────────────────────


class MGHMDataset(Dataset):
    """Dataset wrapping pre-tokenized MGHM pairs for DataLoader."""

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return (
            # Keyword features
            torch.tensor([item["kw_jaccard"], item["kw_overlap"]], dtype=torch.float),
            # Bi-encoder query
            item["q_input_ids"],
            item["q_attention_mask"],
            item["q_token_type_ids"],
            # Bi-encoder candidate
            item["c_input_ids"],
            item["c_attention_mask"],
            item["c_token_type_ids"],
            # Cross-encoder
            item["ce_input_ids"],
            item["ce_attention_mask"],
            item["ce_token_type_ids"],
            # Label
            torch.tensor(item["label"], dtype=torch.float),
        )


# ── Model ─────────────────────────────────────────────────────────────


class MGHMNetwork(nn.Module):
    """Multi-Granularity Hierarchical Matching network.

    The bi-encoder and cross-encoder branches share a single KLUE-BERT
    backbone.  Keyword features pass through without gradients.  A
    learned Linear(4, 1) gate fuses the four scalar features into a
    final relevance score.
    """

    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        # Cross-encoder head: 4-class classifier on [CLS]
        self.ce_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )
        self.label_weights = torch.arange(num_labels, dtype=torch.float)

        # Gated fusion: [jaccard, term_overlap, bi_sim, ce_score] -> score
        self.gate = nn.Linear(4, 1)

    # ── Bi-encoder branch ─────────────────────────────────────────

    def _mean_pool(self, last_hidden_state, attention_mask):
        """Mean-pool token embeddings, masking out padding tokens."""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        return summed / lengths

    def encode(self, input_ids, attention_mask, token_type_ids):
        """Encode a batch of texts into L2-normalised embeddings."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def bi_similarity(self, q_ids, q_mask, q_type, c_ids, c_mask, c_type):
        """Cosine similarity between independently encoded query and candidate."""
        q_emb = self.encode(q_ids, q_mask, q_type)
        c_emb = self.encode(c_ids, c_mask, c_type)
        # Per-pair cosine similarity (element-wise dot of normalised vecs)
        return (q_emb * c_emb).sum(dim=-1)  # (batch,)

    # ── Cross-encoder branch ──────────────────────────────────────

    def ce_score(self, input_ids, attention_mask, token_type_ids):
        """Expected relevance score from 4-class cross-encoder logits."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.ce_classifier(cls_output)
        probs = F.softmax(logits, dim=-1)
        weights = self.label_weights.to(probs.device)
        return (probs * weights).sum(dim=-1)  # (batch,)

    # ── Forward: fuse all granularities ───────────────────────────

    def forward(
        self,
        kw_feats,
        q_ids, q_mask, q_type,
        c_ids, c_mask, c_type,
        ce_ids, ce_mask, ce_type,
    ):
        """Compute the fused multi-granularity relevance score.

        Args:
            kw_feats: (batch, 2) keyword features [jaccard, overlap].
            q_*/c_*: bi-encoder inputs for query and candidate.
            ce_*: cross-encoder joint inputs.

        Returns:
            scores: (batch,) fused relevance scores.
        """
        bi_sim = self.bi_similarity(q_ids, q_mask, q_type, c_ids, c_mask, c_type)
        ce = self.ce_score(ce_ids, ce_mask, ce_type)

        # Concatenate all features: [jaccard, overlap, bi_sim, ce_score]
        features = torch.cat([
            kw_feats,                   # (batch, 2)
            bi_sim.unsqueeze(-1),       # (batch, 1)
            ce.unsqueeze(-1),           # (batch, 1)
        ], dim=-1)                      # (batch, 4)

        return self.gate(features).squeeze(-1)  # (batch,)


# ── Training ──────────────────────────────────────────────────────────


def _freeze_bert_early_layers(model: MGHMNetwork) -> None:
    """Freeze embeddings and the first 10 of 12 BERT encoder layers."""
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer_idx in range(10):
        for param in model.bert.encoder.layer[layer_idx].parameters():
            param.requires_grad = False


def _pairwise_ranking_loss(scores, labels, margin=0.5):
    """Compute pairwise margin-ranking loss over all ordered pairs.

    For every pair (i, j) in the batch where label_i > label_j, we want
    score_i > score_j by at least ``margin``.
    """
    n = scores.size(0)
    if n < 2:
        return torch.tensor(0.0, device=scores.device)

    # All pairs
    s_i = scores.unsqueeze(1).expand(n, n)
    s_j = scores.unsqueeze(0).expand(n, n)
    l_i = labels.unsqueeze(1).expand(n, n)
    l_j = labels.unsqueeze(0).expand(n, n)

    # Mask: only pairs where i has strictly higher label
    mask = (l_i > l_j).float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device)

    # Hinge loss: max(0, margin - (score_i - score_j))
    loss = F.relu(margin - (s_i - s_j))
    loss = (loss * mask).sum() / mask.sum()
    return loss


def _train_mghm(model, train_loader, epochs=EPOCHS):
    """Train the MGHM network with MSE + pairwise ranking loss and AMP.

    Keyword features are detached (no gradient flows through them).
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

        for batch in train_loader:
            (kw_feats,
             q_ids, q_mask, q_type,
             c_ids, c_mask, c_type,
             ce_ids, ce_mask, ce_type,
             labels) = [b.to(DEVICE) for b in batch]

            # Keyword features are pre-computed; detach to avoid spurious grads
            kw_feats = kw_feats.detach()

            optimizer.zero_grad()
            with autocast("cuda"):
                scores = model(
                    kw_feats,
                    q_ids, q_mask, q_type,
                    c_ids, c_mask, c_type,
                    ce_ids, ce_mask, ce_type,
                )
                mse_loss = F.mse_loss(scores, labels)
                rank_loss = _pairwise_ranking_loss(scores, labels)
                loss = mse_loss + rank_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)

    return model


# ── Inference helper ──────────────────────────────────────────────────


def _score_items(model, items):
    """Score a list of pre-tokenized items with the trained MGHM model."""
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for item in items:
            kw = torch.tensor(
                [item["kw_jaccard"], item["kw_overlap"]],
                dtype=torch.float,
            ).unsqueeze(0).to(DEVICE)

            with autocast("cuda"):
                score = model(
                    kw,
                    item["q_input_ids"].unsqueeze(0).to(DEVICE),
                    item["q_attention_mask"].unsqueeze(0).to(DEVICE),
                    item["q_token_type_ids"].unsqueeze(0).to(DEVICE),
                    item["c_input_ids"].unsqueeze(0).to(DEVICE),
                    item["c_attention_mask"].unsqueeze(0).to(DEVICE),
                    item["c_token_type_ids"].unsqueeze(0).to(DEVICE),
                    item["ce_input_ids"].unsqueeze(0).to(DEVICE),
                    item["ce_attention_mask"].unsqueeze(0).to(DEVICE),
                    item["ce_token_type_ids"].unsqueeze(0).to(DEVICE),
                ).item()
            scores.append(score)
    return scores


# ── 5-fold CV pipeline ────────────────────────────────────────────────


def run_mghm(
    groups: list[QueryGroup],
) -> tuple[list[list[int]], list[list[float]]]:
    """Run MGHM with 5-fold pair-level stratified cross-validation.

    Steps:
        1. Pre-tokenize all pairs (keyword features + bi-encoder tokens
           + cross-encoder tokens).
        2. Flatten pairs with group-index tracking.
        3. StratifiedKFold(5) on graded relevance labels.
        4. Per fold: train a fresh MGHMNetwork, predict held-out pairs.
        5. Reassemble per-query results for retrieval evaluation.

    Returns:
        all_labels: list of label lists, one per query group.
        all_scores: list of score lists, one per query group.
    """
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    all_group_data = pretokenize_mghm_pairs(groups, tokenizer)

    # Flatten pairs with group tracking
    flat_items: list[dict] = []
    flat_labels: list[int] = []
    flat_group_idx: list[int] = []
    for gi, gd in enumerate(all_group_data):
        for item in gd:
            flat_items.append(item)
            flat_labels.append(item["label"])
            flat_group_idx.append(gi)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred_scores: list[float | None] = [None] * len(flat_items)

    for fold, (train_idx, test_idx) in enumerate(skf.split(flat_items, flat_labels)):
        print(
            f"\n  Fold {fold + 1}/5 (train={len(train_idx)}, test={len(test_idx)})",
            flush=True,
        )

        # Build training set
        train_items = [flat_items[i] for i in train_idx]
        train_dataset = MGHMDataset(train_items)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        )

        # Train a fresh MGHM model
        model = MGHMNetwork().to(DEVICE)
        _freeze_bert_early_layers(model)
        model = _train_mghm(model, train_loader, epochs=EPOCHS)

        # Predict on held-out pairs
        test_items = [flat_items[i] for i in test_idx]
        fold_scores = _score_items(model, test_items)
        for idx, score in zip(test_idx, fold_scores):
            pred_scores[idx] = score

        del model
        torch.cuda.empty_cache()

    # Reassemble per-query results
    all_labels: list[list[int]] = []
    all_scores: list[list[float]] = []
    for gi in range(len(groups)):
        g_labels: list[int] = []
        g_scores: list[float] = []
        for fi in range(len(flat_items)):
            if flat_group_idx[fi] == gi:
                g_labels.append(flat_labels[fi])
                g_scores.append(pred_scores[fi])
        all_labels.append(g_labels)
        all_scores.append(g_scores)

    return all_labels, all_scores


# ── Standalone entry point ────────────────────────────────────────────


def main() -> None:
    """Run the MGHM pipeline and print retrieval metrics."""
    print("Loading dataset...", flush=True)
    groups = load_dataset()
    print(f"Loaded {len(groups)} query groups, using device: {DEVICE}", flush=True)

    print(f"\n{'=' * 60}")
    print("Running Multi-Granularity Hierarchical Matching (MGHM)...")
    print(f"{'=' * 60}", flush=True)

    all_labels, all_scores = run_mghm(groups)
    results = evaluate_retrieval(all_labels, all_scores)

    print("\n=== MGHM Results ===")
    for key, value in results.items():
        print(f"  {key}: {value:.2f}")

    return {"MGHM": results}


if __name__ == "__main__":
    main()

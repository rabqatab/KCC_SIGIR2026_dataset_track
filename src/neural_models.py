"""
Neural baseline models for Korean case law retrieval: 1D-CNN and LSTM.

Encodes query and candidate case notes using noun-level tokens extracted via Kiwi, with pretrained FastText embeddings.
Training uses a combined MSE + pairwise margin ranking loss so that the model learns both absolute relevance calibration 
and correct pairwise ordering -- the latter being the key driver of nDCG performance.

Evaluation follows 5-fold stratified pair-level cross-validation (StratifiedKFold, n_splits=5, shuffle=True, random_state=42),
consistent with the BERT baselines in bert_models.py.
"""

# ---------------------------------------------------------------------------
# The pairwise ranking loss (see `pairwise_ranking_loss`) is the primary mechanism through which these models optimise nDCG.
# By enforcing a margin between scores of differently-graded pairs, the loss directly encourages the ranked list to place 
# highly-relevant cases above less-relevant ones, which is exactly what nDCG measures.
# ---------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from kiwipiepy import Kiwi
from sklearn.model_selection import StratifiedKFold

from src.data_loader import QueryGroup, load_dataset
from src.metrics import evaluate_retrieval

_kiwi = Kiwi()

# ---------------------------------------------------------------------------
# Hyperparameters & paths
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_PATH = "results/embeddings/fasttext_legal.model"
MAX_SEQ_LEN = 128
EMBED_DIM = 128
HIDDEN_DIM = 128
EPOCHS = 3
BATCH_SIZE = 32
LR = 1e-3


# ---------------------------------------------------------------------------
# Tokenisation & vocabulary
# ---------------------------------------------------------------------------

def extract_nouns(text: str) -> list[str]:
    """Extract nouns from Korean text using Kiwi."""
    return [t.form for t in _kiwi.tokenize(text) if t.tag.startswith("NN")]


def precompute_all_tokens(groups: list[QueryGroup]) -> dict[str, list[str]]:
    """Pre-compute noun tokens for every unique text across all groups."""
    token_cache: dict[str, list[str]] = {}
    for g in groups:
        if g.query_note not in token_cache:
            token_cache[g.query_note] = extract_nouns(g.query_note)
        for pair in g.pairs:
            if pair.candidate_note not in token_cache:
                token_cache[pair.candidate_note] = extract_nouns(pair.candidate_note)
    print(f"  Tokenised {len(token_cache)} unique texts", flush=True)
    return token_cache


def build_vocab_from_cache(token_cache: dict[str, list[str]],
                           min_freq: int = 2) -> dict[str, int]:
    """Build word-to-index vocabulary from pre-computed tokens."""
    counter: Counter = Counter()
    for tokens in token_cache.values():
        counter.update(tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def load_pretrained_embeddings(
    vocab: dict[str, int],
    embed_dim: int = EMBED_DIM,
    model_path: str = EMBEDDING_PATH,
) -> tuple[torch.Tensor | None, float]:
    """Load pretrained FastText/Word2Vec embeddings into a weight matrix.

    Returns
    -------
    embedding_matrix : torch.Tensor or None
    coverage_pct : float
    """
    from gensim.models import FastText as FTModel, Word2Vec as W2VModel

    if not os.path.exists(model_path):
        print(f"  WARNING: no pretrained embeddings at {model_path}, using random init",
              flush=True)
        return None, 0.0

    if "fasttext" in model_path:
        gensim_model = FTModel.load(model_path)
    else:
        gensim_model = W2VModel.load(model_path)

    wv = gensim_model.wv
    matrix = np.zeros((len(vocab), embed_dim), dtype=np.float32)
    found = 0

    for word, idx in vocab.items():
        if idx == 0:  # <pad> stays zero
            continue
        if word in wv:
            matrix[idx] = wv[word][:embed_dim]
            found += 1
        elif hasattr(wv, "get_vector"):
            # FastText can generate vectors for OOV words via subwords
            try:
                matrix[idx] = wv.get_vector(word)[:embed_dim]
                found += 1
            except KeyError:
                matrix[idx] = np.random.randn(embed_dim) * 0.1
        else:
            matrix[idx] = np.random.randn(embed_dim) * 0.1

    coverage = found / max(len(vocab) - 1, 1) * 100  # exclude <pad>
    print(f"  Embedding coverage: {found}/{len(vocab) - 1} words ({coverage:.1f}%)",
          flush=True)
    return torch.tensor(matrix), coverage


def tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    """Convert a token list to a padded / truncated integer-ID sequence."""
    ids = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    if len(ids) < MAX_SEQ_LEN:
        ids = ids + [0] * (MAX_SEQ_LEN - len(ids))
    else:
        ids = ids[:MAX_SEQ_LEN]
    return ids


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Flat dataset of (query_ids, candidate_ids, label) triples."""

    def __init__(self, samples: list[tuple[list[int], list[int], int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        q, c, label = self.samples[idx]
        return (
            torch.tensor(q, dtype=torch.long),
            torch.tensor(c, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CNN1DRanker(nn.Module):
    """1D-CNN model for case similarity scoring."""

    def __init__(self, vocab_size: int,
                 pretrained_weights: torch.Tensor | None = None,
                 freeze_embeddings: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        self.conv1 = nn.Conv1d(EMBED_DIM, HIDDEN_DIM, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(HIDDEN_DIM, HIDDEN_DIM, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).permute(0, 2, 1)
        h = F.relu(self.conv1(emb))
        h = F.relu(self.conv2(h))
        return h.max(dim=2).values

    def forward(self, query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        q_enc = self.encode(query)
        c_enc = self.encode(candidate)
        combined = torch.cat([q_enc, c_enc], dim=1)
        return self.fc(combined).squeeze(-1)


class LSTMRanker(nn.Module):
    """Bidirectional LSTM model for case similarity scoring."""

    def __init__(self, vocab_size: int,
                 pretrained_weights: torch.Tensor | None = None,
                 freeze_embeddings: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        return torch.cat([h_n[0], h_n[1]], dim=1)

    def forward(self, query: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        q_enc = self.encode(query)
        c_enc = self.encode(candidate)
        combined = torch.cat([q_enc, c_enc], dim=1)
        return self.fc(combined).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss & training
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(scores: torch.Tensor, labels: torch.Tensor,
                          margin: float = 0.5) -> torch.Tensor:
    """Pairwise margin ranking loss.

    For every ordered pair (i, j) where label_i > label_j, 
    enforce score_i > score_j + margin * (label_i - label_j).
    This directly optimises for ranking quality (nDCG-aligned).
    """
    n = scores.size(0)
    if n < 2:
        return torch.tensor(0.0, device=scores.device)

    s_i = scores.unsqueeze(1)   # (n, 1)
    s_j = scores.unsqueeze(0)   # (1, n)
    l_i = labels.unsqueeze(1)   # (n, 1)
    l_j = labels.unsqueeze(0)   # (1, n)

    diff_mask = (l_i > l_j).float()
    label_diff = (l_i - l_j).float()

    loss_matrix = F.relu(margin * label_diff - (s_i - s_j)) * diff_mask
    n_pairs = diff_mask.sum().clamp(min=1)
    return loss_matrix.sum() / n_pairs


def train_model(model: nn.Module, train_loader: DataLoader,
                epochs: int = EPOCHS) -> nn.Module:
    """Train with combined MSE + pairwise ranking loss.

    MSE teaches absolute label prediction; ranking loss teaches ordering.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for q, c, labels in train_loader:
            q, c, labels = q.to(DEVICE), c.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            scores = model(q, c)

            mse_loss = F.mse_loss(scores, labels)
            rank_loss = pairwise_ranking_loss(scores, labels, margin=0.3)
            loss = mse_loss + rank_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg:.4f}", flush=True)

    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_scores(model: nn.Module, group: QueryGroup,
                   vocab: dict[str, int],
                   token_cache: dict[str, list[str]]) -> list[float]:
    """Return prediction scores for every candidate in a query group."""
    model.eval()
    scores: list[float] = []
    q_ids = torch.tensor(
        [tokens_to_ids(token_cache[group.query_note], vocab)],
        dtype=torch.long,
    ).to(DEVICE)

    with torch.no_grad():
        for pair in group.pairs:
            c_ids = torch.tensor(
                [tokens_to_ids(token_cache[pair.candidate_note], vocab)],
                dtype=torch.long,
            ).to(DEVICE)
            scores.append(model(q_ids, c_ids).item())

    return scores


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------

def run_neural_model(
    model_class: type[nn.Module],
    groups: list[QueryGroup],
    model_name: str,
    embedding_path: str = EMBEDDING_PATH,
) -> tuple[list[list[int]], list[list[float]]]:
    """Run 5-fold stratified pair-level CV for a neural ranking model.

    Steps
    -----
    1. Pre-compute noun tokens and build shared vocabulary / embeddings.
    2. Flatten all (query, candidate) pairs with group-index tracking.
    3. Run 5-fold StratifiedKFold on the flattened pairs.
    4. Reassemble per-query label / score lists after all folds.
    """
    # -- shared preprocessing (done once) ----------------------------------
    print("\nPre-computing noun tokens for all texts...", flush=True)
    token_cache = precompute_all_tokens(groups)

    vocab = build_vocab_from_cache(token_cache)
    print(f"  Vocabulary size: {len(vocab)}", flush=True)

    pretrained_weights, _ = load_pretrained_embeddings(
        vocab, EMBED_DIM, embedding_path,
    )

    # -- flatten all pairs with group / pair index tracking ----------------
    flat_samples: list[tuple[list[int], list[int], int]] = []
    flat_labels: list[int] = []
    flat_group_idx: list[int] = []
    flat_pair_idx: list[int] = []

    for gi, g in enumerate(groups):
        q_ids = tokens_to_ids(token_cache[g.query_note], vocab)
        for pi, pair in enumerate(g.pairs):
            c_ids = tokens_to_ids(token_cache[pair.candidate_note], vocab)
            flat_samples.append((q_ids, c_ids, pair.label))
            flat_labels.append(pair.label)
            flat_group_idx.append(gi)
            flat_pair_idx.append(pi)

    # -- 5-fold stratified CV ----------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred_scores: list[float | None] = [None] * len(flat_samples)

    for fold, (train_idx, test_idx) in enumerate(skf.split(flat_samples, flat_labels)):
        print(f"\n  Fold {fold + 1}/5 (train={len(train_idx)}, test={len(test_idx)})",
              flush=True)

        train_data = [flat_samples[i] for i in train_idx]
        train_dataset = PairDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = model_class(
            len(vocab),
            pretrained_weights=pretrained_weights,
            freeze_embeddings=False,
        ).to(DEVICE)
        model = train_model(model, train_loader)

        # -- score held-out pairs ------------------------------------------
        model.eval()
        with torch.no_grad():
            for idx in test_idx:
                q_ids_t, c_ids_t, _ = flat_samples[idx]
                q_tensor = torch.tensor([q_ids_t], dtype=torch.long).to(DEVICE)
                c_tensor = torch.tensor([c_ids_t], dtype=torch.long).to(DEVICE)
                pred_scores[idx] = model(q_tensor, c_tensor).item()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- reassemble per-query results --------------------------------------
    all_labels: list[list[int]] = [[] for _ in groups]
    all_scores: list[list[float]] = [[] for _ in groups]
    for fi in range(len(flat_samples)):
        gi = flat_group_idx[fi]
        all_labels[gi].append(flat_labels[fi])
        all_scores[gi].append(pred_scores[fi])

    return all_labels, all_scores


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset...", flush=True)
    groups = load_dataset()
    print(f"Using device: {DEVICE}", flush=True)

    results = {}

    for model_name, model_class in [("1D-CNN", CNN1DRanker), ("LSTM", LSTMRanker)]:
        print(f"\n{'=' * 50}")
        print(f"Running {model_name}...")
        print(f"{'=' * 50}")

        all_labels, all_scores = run_neural_model(model_class, groups, model_name)
        metrics = evaluate_retrieval(all_labels, all_scores)
        results[model_name] = metrics

        print(f"\n=== {model_name} Results ===")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")

    return results


if __name__ == "__main__":
    main()

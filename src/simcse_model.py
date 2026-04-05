"""SimCSE contrastive learning for Korean legal case retrieval.

Implements a two-phase training approach using KLUE-BERT as the backbone:
  - Phase 1 (unsupervised pre-training): Passes each case note through KLUE-BERT twice
    with different dropout masks to form positive pairs, using in-batch negatives
    with InfoNCE loss. This builds a high-quality sentence embedding space without labels.
  - Phase 2 (supervised fine-tuning): Uses labeled pairs with supervised contrastive
    loss -- pairs with label >= 2 are positives, label <= 1 are negatives.
  - Scoring: Cosine similarity between query and candidate embeddings.

Evaluation uses 5-fold stratified pair-level cross-validation.
Early BERT layers (10/12) are frozen to prevent overfitting.
Mixed-precision (AMP) is enabled throughout.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, BertModel
from sklearn.model_selection import StratifiedKFold

from src.data_loader import QueryGroup, load_dataset
from src.metrics import evaluate_retrieval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "klue/bert-base"
MAX_LEN = 256
PRETRAIN_EPOCHS = 3
FINETUNE_EPOCHS = 5
PRETRAIN_BATCH_SIZE = 64
FINETUNE_BATCH_SIZE = 16
PRETRAIN_LR = 3e-5
FINETUNE_LR = 2e-5
TEMPERATURE = 0.05


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class UnsupervisedNoteDataset(Dataset):
    """Dataset of individual case notes for unsupervised SimCSE pre-training.

    Each sample is a single tokenized case note. During training, the same
    note is passed through the encoder twice with different dropout masks
    to produce a positive pair.
    """

    def __init__(self, encodings: list[dict]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int):
        enc = self.encodings[idx]
        return enc["input_ids"], enc["attention_mask"]


class SupervisedPairDataset(Dataset):
    """Dataset of (query, candidate, binary_label) triples for supervised fine-tuning."""

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
            torch.tensor(s["binary_label"], dtype=torch.float),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SimCSEEncoder(nn.Module):
    """KLUE-BERT encoder with mean pooling for SimCSE.

    Produces fixed-size sentence embeddings via mean pooling over
    non-padded token representations from the last hidden state.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input and return mean-pooled sentence embeddings."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        summed = (hidden * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts  # (batch, hidden)


# ---------------------------------------------------------------------------
# Layer freezing
# ---------------------------------------------------------------------------

def _freeze_bert_early_layers(model: SimCSEEncoder) -> None:
    """Freeze embedding layer and first 10 of 12 encoder layers.

    Only the last two encoder layers remain trainable, preventing
    overfitting on the small Korean legal case dataset.
    """
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer_idx in range(10):
        for param in model.bert.encoder.layer[layer_idx].parameters():
            param.requires_grad = False


# ---------------------------------------------------------------------------
# Unsupervised pre-training (Phase 1)
# ---------------------------------------------------------------------------

def collect_unique_notes(groups: list[QueryGroup]) -> list[str]:
    """Collect all unique case note texts from query notes and candidate notes."""
    seen: set[str] = set()
    notes: list[str] = []
    for g in groups:
        if g.query_note and g.query_note not in seen:
            seen.add(g.query_note)
            notes.append(g.query_note)
        for pair in g.pairs:
            if pair.candidate_note and pair.candidate_note not in seen:
                seen.add(pair.candidate_note)
                notes.append(pair.candidate_note)
    return notes


def tokenize_notes(notes: list[str], tokenizer: AutoTokenizer,
                   max_len: int = MAX_LEN) -> list[dict]:
    """Tokenize a list of case note texts into input_ids and attention_mask."""
    encodings: list[dict] = []
    for note in notes:
        enc = tokenizer(
            note,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encodings.append({
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        })
    return encodings


def unsupervised_pretrain(model: SimCSEEncoder, dataloader: DataLoader,
                          epochs: int = PRETRAIN_EPOCHS) -> SimCSEEncoder:
    """Unsupervised SimCSE pre-training with InfoNCE loss.

    Each note is passed through the encoder twice with different dropout
    masks. The two representations form a positive pair while all other
    notes in the batch serve as negatives.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PRETRAIN_LR, weight_decay=0.01,
    )
    scaler = GradScaler("cuda")
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            batch_size = input_ids.size(0)

            optimizer.zero_grad()
            with autocast("cuda"):
                # Pass the same batch twice -- different dropout masks
                emb1 = model(input_ids, attention_mask)  # (B, hidden)
                emb2 = model(input_ids, attention_mask)  # (B, hidden)

                # Normalize for cosine similarity
                emb1 = F.normalize(emb1, p=2, dim=1)
                emb2 = F.normalize(emb2, p=2, dim=1)

                # Cosine similarity matrix: (B, B)
                sim_matrix = torch.mm(emb1, emb2.t()) / TEMPERATURE

                # Positive pairs lie on the diagonal
                labels = torch.arange(batch_size, device=DEVICE)
                loss = F.cross_entropy(sim_matrix, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"    Pretrain Epoch {epoch + 1}/{epochs}, InfoNCE Loss: {avg_loss:.4f}",
              flush=True)

    return model


# ---------------------------------------------------------------------------
# Supervised fine-tuning (Phase 2)
# ---------------------------------------------------------------------------

def supervised_finetune(model: SimCSEEncoder, dataloader: DataLoader,
                        epochs: int = FINETUNE_EPOCHS) -> SimCSEEncoder:
    """Supervised contrastive fine-tuning.

    Pairs with binary_label=1 (original label >= 2) are treated as positives,
    pairs with binary_label=0 (original label <= 1) as negatives.
    The loss pushes positive-pair embeddings closer and negative-pair
    embeddings apart, using cosine similarity scaled by temperature.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR, weight_decay=0.01,
    )
    scaler = GradScaler("cuda")
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for q_ids, q_mask, c_ids, c_mask, binary_labels in dataloader:
            q_ids = q_ids.to(DEVICE)
            q_mask = q_mask.to(DEVICE)
            c_ids = c_ids.to(DEVICE)
            c_mask = c_mask.to(DEVICE)
            binary_labels = binary_labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                q_emb = F.normalize(model(q_ids, q_mask), p=2, dim=1)
                c_emb = F.normalize(model(c_ids, c_mask), p=2, dim=1)

                # Cosine similarity for each pair
                cos_sim = (q_emb * c_emb).sum(dim=1) / TEMPERATURE  # (B,)

                # Supervised contrastive: positive pairs should have high sim,
                # negative pairs should have low sim
                # Use binary cross-entropy on sigmoid of scaled similarity
                loss = F.binary_cross_entropy_with_logits(cos_sim, binary_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"    Finetune Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}",
              flush=True)

    return model


# ---------------------------------------------------------------------------
# Pre-tokenization for supervised pairs
# ---------------------------------------------------------------------------

def pretokenize_pairs(groups: list[QueryGroup], tokenizer: AutoTokenizer,
                      max_len: int = MAX_LEN) -> list[list[dict]]:
    """Pre-tokenize all query-candidate pairs for supervised fine-tuning.

    Returns a list of lists (one per group), each containing dicts with
    separate query/candidate encodings and labels.
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
                "binary_label": 1 if pair.label >= 2 else 0,
            })
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{total} pairs tokenized", flush=True)
        all_group_data.append(group_data)

    print(f"    {total}/{total} pairs tokenized (done)", flush=True)
    return all_group_data


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_pairs(model: SimCSEEncoder,
                flat_items: list[dict],
                indices: list[int]) -> list[tuple[int, float]]:
    """Score held-out pairs by cosine similarity between query and candidate embeddings."""
    model.eval()
    results: list[tuple[int, float]] = []

    with torch.no_grad():
        for idx in indices:
            item = flat_items[idx]
            with autocast("cuda"):
                q_emb = model(
                    item["query_input_ids"].unsqueeze(0).to(DEVICE),
                    item["query_attention_mask"].unsqueeze(0).to(DEVICE),
                )
                c_emb = model(
                    item["candidate_input_ids"].unsqueeze(0).to(DEVICE),
                    item["candidate_attention_mask"].unsqueeze(0).to(DEVICE),
                )
                q_emb = F.normalize(q_emb, p=2, dim=1)
                c_emb = F.normalize(c_emb, p=2, dim=1)
                score = (q_emb * c_emb).sum(dim=1).item()
            results.append((idx, score))

    return results


# ---------------------------------------------------------------------------
# 5-fold cross-validated evaluation
# ---------------------------------------------------------------------------

def run_simcse(
    groups: list[QueryGroup],
) -> tuple[list[list[int]], list[list[float]]]:
    """Run SimCSE with unsupervised pre-training + 5-fold supervised fine-tuning.

    Steps
    -----
    1. Unsupervised pre-train on all unique case notes (done once, outside CV).
    2. Pre-tokenize all query-candidate pairs.
    3. Flatten pairs with group-index tracking.
    4. Run 5-fold StratifiedKFold on the flattened pairs.
    5. Per fold: load pre-trained weights, fine-tune on train pairs, score
       test pairs by cosine similarity.
    6. Reassemble per-query label / score lists.
    """
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -- Phase 1: unsupervised pre-training on all unique notes ---------------
    print("\nCollecting unique case notes for unsupervised pre-training...", flush=True)
    unique_notes = collect_unique_notes(groups)
    print(f"  Found {len(unique_notes)} unique case notes", flush=True)

    note_encodings = tokenize_notes(unique_notes, tokenizer)
    note_dataset = UnsupervisedNoteDataset(note_encodings)
    note_loader = DataLoader(note_dataset, batch_size=PRETRAIN_BATCH_SIZE,
                             shuffle=True, drop_last=True)

    print("\nPhase 1: Unsupervised SimCSE pre-training...", flush=True)
    pretrain_model = SimCSEEncoder().to(DEVICE)
    _freeze_bert_early_layers(pretrain_model)
    pretrain_model = unsupervised_pretrain(pretrain_model, note_loader)

    # Save pre-trained state for reuse across folds
    pretrained_state = copy.deepcopy(pretrain_model.state_dict())
    del pretrain_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Pre-tokenize all pairs -----------------------------------------------
    all_group_data = pretokenize_pairs(groups, tokenizer)

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
    print("\nPhase 2: Supervised fine-tuning with 5-fold CV...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred_scores: list[float | None] = [None] * len(flat_items)

    for fold, (train_idx, test_idx) in enumerate(skf.split(flat_items, flat_labels)):
        print(f"\n  Fold {fold + 1}/5 (train={len(train_idx)}, test={len(test_idx)})",
              flush=True)

        # Build training data for this fold
        train_data = [flat_items[i] for i in train_idx]
        train_dataset = SupervisedPairDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=FINETUNE_BATCH_SIZE,
                                  shuffle=True)

        # Load pre-trained weights and fine-tune
        model = SimCSEEncoder().to(DEVICE)
        model.load_state_dict(pretrained_state)
        _freeze_bert_early_layers(model)
        model = supervised_finetune(model, train_loader)

        # Score held-out pairs by cosine similarity
        for idx, score in score_pairs(model, flat_items, test_idx):
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
    """Entry point: load data, run SimCSE CV, and print results."""
    print("Loading dataset...", flush=True)
    groups = load_dataset()
    print(f"Using device: {DEVICE}", flush=True)

    print(f"\n{'=' * 60}")
    print("Running SimCSE...")
    print("=" * 60, flush=True)

    all_labels, all_scores = run_simcse(groups)
    metrics = evaluate_retrieval(all_labels, all_scores)

    print("\n=== SimCSE Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    return metrics


if __name__ == "__main__":
    main()

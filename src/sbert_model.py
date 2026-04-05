"""Sentence-BERT bi-encoder for Korean legal case retrieval.

Encodes query and candidate independently through KLUE-BERT with mean pooling,
normalizes embeddings, and computes cosine similarity as the relevance score.
Training uses InfoNCE contrastive loss with in-batch negatives.

Evaluation uses 5-fold stratified pair-level cross-validation.
Early BERT layers (10/12) are frozen to prevent overfitting.
Mixed-precision (AMP) is enabled.
"""

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
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5


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


class SBERTDataset(Dataset):
    """Dataset wrapping pre-tokenized query-candidate pairs for bi-encoder training."""

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
        )


class SentenceBERT(nn.Module):
    """Sentence-BERT bi-encoder with mean pooling over KLUE-BERT.

    Encodes a single text into a normalized dense vector.
    Cosine similarity between query and candidate vectors serves as
    the relevance score.
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
        """Encode query and candidate batches, return similarity matrix.

        Returns the cosine similarity matrix of shape (batch, batch) where
        entry (i, j) is the similarity between query i and candidate j.
        """
        q_emb = self.encode(q_ids, q_mask, q_type)
        c_emb = self.encode(c_ids, c_mask, c_type)
        return q_emb @ c_emb.T


def _freeze_bert_early_layers(model):
    """Freeze embedding and first 10 of 12 encoder layers.

    Only the last two encoder layers remain trainable, along with
    any task-specific parameters.  This prevents overfitting when
    fine-tuning on the small Korean legal case dataset.
    """
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer_idx in range(10):
        for param in model.bert.encoder.layer[layer_idx].parameters():
            param.requires_grad = False


def _train_sbert(model, train_loader, epochs=EPOCHS):
    """Train Sentence-BERT with InfoNCE contrastive loss and mixed precision.

    For each batch of (query, candidate) pairs the similarity matrix is
    queries @ candidates.T.  The diagonal entries are the positive pairs
    (each query matches its own candidate).  Cross-entropy over this
    matrix with arange labels implements InfoNCE with in-batch negatives.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01,
    )
    scaler = GradScaler("cuda")
    temperature = 0.05

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            q_ids, q_mask, q_type, c_ids, c_mask, c_type = [
                b.to(DEVICE) for b in batch
            ]
            optimizer.zero_grad()

            with autocast("cuda"):
                sim_matrix = model(q_ids, q_mask, q_type, c_ids, c_mask, c_type)
                sim_matrix = sim_matrix / temperature
                labels = torch.arange(sim_matrix.size(0), device=DEVICE)
                loss = F.cross_entropy(sim_matrix, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)

    return model


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


def run_sbert(groups):
    """Run Sentence-BERT bi-encoder with 5-fold pair-level stratified CV.

    Steps:
        1. Pre-tokenize all query and candidate texts independently.
        2. Flatten pairs with flat_group_idx tracking for reassembly.
        3. Stratified 5-fold split on graded relevance labels.
        4. Per fold: train a fresh SentenceBERT, predict held-out pairs
           via cosine similarity.
        5. Reassemble per-query results for retrieval evaluation.

    Returns:
        all_labels: list of label lists, one per query group.
        all_scores: list of score lists, one per query group.
    """
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
        print(f"\n  Fold {fold + 1}/5 (train={len(train_idx)}, test={len(test_idx)})", flush=True)

        train_items = [flat_items[i] for i in train_idx]
        train_dataset = SBERTDataset(train_items)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        )

        model = SentenceBERT().to(DEVICE)
        _freeze_bert_early_layers(model)
        model = _train_sbert(model, train_loader, epochs=EPOCHS)

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


def main():
    """Run Sentence-BERT and print retrieval metrics."""
    print("Loading dataset...", flush=True)
    groups = load_dataset()
    print(f"Using device: {DEVICE}", flush=True)

    print(f"\n{'=' * 60}")
    print("Running Sentence-BERT (Bi-Encoder)...")
    print("=" * 60, flush=True)

    labels, scores = run_sbert(groups)
    metrics = evaluate_retrieval(labels, scores)

    print("\n=== Sentence-BERT Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    return {"Sentence-BERT": metrics}


if __name__ == "__main__":
    main()

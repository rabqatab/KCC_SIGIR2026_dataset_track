"""BERT-based models for Korean legal case similarity.

Implements three approaches using KoBERT (monologg/kobert) as the backbone PLM:
  - BERT Cross-Encoder (CE): 4-class graded relevance via [CLS] classification.
  - BERT-PLI: Paragraph-Level Interaction with BiLSTM aggregation over paragraph-pair [CLS] embeddings, 4-class classification.
  - Finetuned BERT Binary: Binary (similar/dissimilar) classification with class-weighted cross-entropy.

Evaluation uses 5-fold stratified pair-level cross-validation.
Training uses cross-entropy with label smoothing (0.1) for graded relevance tasks 
and class-weighted cross-entropy for the binary task.
Early BERT layers (10/12) are frozen to prevent overfitting.
Mixed-precision (AMP) is enabled.
"""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import StratifiedKFold

from src.data_loader import QueryGroup, load_dataset
from src.metrics import evaluate_retrieval, evaluate_binary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "monologg/kobert" # if klue-bert for use, use "klue/bert-base", AutoModel and AutoTokenizer
MAX_LEN = 512
PARA_MAX_LEN = 128
NUM_LABELS = 4  # 4-class classification for graded relevance
EPOCHS = 5
BATCH_SIZE = 16
LR = 5e-5


def pretokenize_all_pairs(groups, tokenizer, max_len=MAX_LEN):
    """Pre-tokenize all query-candidate pairs once.

    Returns a list of lists, one per group, each containing dicts with
    input_ids, attention_mask, token_type_ids, label, and binary_label.
    """
    print("  Pre-tokenizing all pairs...", flush=True)
    all_group_data = []
    total = sum(len(g.pairs) for g in groups)
    done = 0

    for g in groups:
        group_data = []
        for pair in g.pairs:
            enc = tokenizer(
                g.query_note, pair.candidate_note,
                max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            group_data.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "token_type_ids": enc["token_type_ids"].squeeze(0),
                "label": pair.label,
                "binary_label": 1 if pair.label >= 2 else 0,
            })
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{total} pairs tokenized", flush=True)

        all_group_data.append(group_data)

    print(f"    {total}/{total} pairs tokenized (done)", flush=True)
    return all_group_data


class PreTokenizedDataset(Dataset):
    """Dataset wrapping pre-tokenized query-candidate pairs."""

    def __init__(self, group_data_list, binary=False):
        self.samples = []
        for gd in group_data_list:
            for item in gd:
                label = item["binary_label"] if binary else item["label"]
                self.samples.append((
                    item["input_ids"],
                    item["attention_mask"],
                    item["token_type_ids"],
                    label,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, mask, types, label = self.samples[idx]
        return ids, mask, types, torch.tensor(label, dtype=torch.float)


class BertCrossEncoder(nn.Module):
    """BERT Cross-Encoder for case similarity scoring (4-class classification)."""

    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_LABELS):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, num_labels),
        )
        self.label_weights = torch.arange(num_labels, dtype=torch.float)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

    def predict_score(self, input_ids, attention_mask, token_type_ids):
        """Get expected relevance score from class probabilities."""
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        probs = F.softmax(logits, dim=-1)
        weights = self.label_weights.to(probs.device)
        return (probs * weights).sum(dim=-1)


class BertBinaryClassifier(nn.Module):
    """Finetuned BERT for binary similar/dissimilar classification."""

    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 2),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


class BertPLI(nn.Module):
    """BERT-PLI: Paragraph-Level Interactions for legal case retrieval.

    Encodes each query-paragraph x candidate-paragraph pair through BERT,
    then aggregates [CLS] representations with a BiLSTM before 4-class classification.
    """

    def __init__(self, model_name=MODEL_NAME, max_paras=4, num_labels=NUM_LABELS):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.aggregator = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )
        self.label_weights = torch.arange(num_labels, dtype=torch.float)

    def forward(self, para_input_ids, para_attention_mask, para_token_type_ids, n_pairs):
        batch_size = para_input_ids.size(0)
        n_total = para_input_ids.size(1)

        flat_ids = para_input_ids.view(-1, para_input_ids.size(-1))
        flat_mask = para_attention_mask.view(-1, para_attention_mask.size(-1))
        flat_type = para_token_type_ids.view(-1, para_token_type_ids.size(-1))

        chunk_size = 16
        all_cls = []
        for i in range(0, flat_ids.size(0), chunk_size):
            outputs = self.bert(
                input_ids=flat_ids[i:i + chunk_size],
                attention_mask=flat_mask[i:i + chunk_size],
                token_type_ids=flat_type[i:i + chunk_size],
            )
            all_cls.append(outputs.last_hidden_state[:, 0, :])

        cls_outputs = torch.cat(all_cls, dim=0)
        cls_outputs = cls_outputs.view(batch_size, n_total, -1)
        lstm_out, _ = self.aggregator(cls_outputs)
        pooled = lstm_out.mean(dim=1)
        return self.classifier(pooled)

    def predict_score(self, para_input_ids, para_attention_mask, para_token_type_ids, n_pairs):
        """Get expected relevance score from class probabilities."""
        logits = self.forward(para_input_ids, para_attention_mask, para_token_type_ids, n_pairs)
        probs = F.softmax(logits, dim=-1)
        weights = self.label_weights.to(probs.device)
        return (probs * weights).sum(dim=-1)


def split_paragraphs(text, max_paras):
    """Split Korean legal text into meaningful paragraphs.

    Korean legal sentences end with patterns like:
      - declarative endings (다. / 한다. / 것이다.)
      - existence/state endings (있다. / 없다. / 된다.)
    Bracket markers like 【주문】【이유】 also separate sections.

    Sentences are split on these boundaries and then grouped into at most
    `max_paras` roughly equal paragraphs.
    """
    sentences = re.split(r'(?<=다)\.\s*|(?<=음)\.\s*|【[^】]+】', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    if not sentences:
        return [text]
    if len(sentences) <= max_paras:
        return sentences

    per_para = max(1, len(sentences) // max_paras)
    paras = []
    for i in range(0, len(sentences), per_para):
        para = " ".join(sentences[i:i + per_para])
        paras.append(para)
        if len(paras) >= max_paras:
            break
    return paras


def pretokenize_pli_pairs(groups, tokenizer, max_paras=4, max_para_len=PARA_MAX_LEN):
    """Pre-tokenize paragraph-level pairs for BERT-PLI.

    For each query-candidate pair, splits both texts into paragraphs 
    and tokenizes all query-paragraph x candidate-paragraph combinations,
    padding to ``max_paras^2`` entries.
    """
    print("  Pre-tokenizing PLI paragraph pairs...", flush=True)
    all_group_data = []
    total = sum(len(g.pairs) for g in groups)
    done = 0
    target_n = max_paras * max_paras

    for g in groups:
        q_paras = split_paragraphs(g.query_note, max_paras)
        group_data = []

        for pair in g.pairs:
            c_paras = split_paragraphs(pair.candidate_note, max_paras)
            para_encodings = []
            for qp in q_paras:
                for cp in c_paras:
                    enc = tokenizer(
                        qp, cp, max_length=max_para_len,
                        padding="max_length", truncation=True,
                        return_tensors="pt",
                    )
                    para_encodings.append({
                        "input_ids": enc["input_ids"].squeeze(0),
                        "attention_mask": enc["attention_mask"].squeeze(0),
                        "token_type_ids": enc["token_type_ids"].squeeze(0),
                    })

            while len(para_encodings) < target_n:
                para_encodings.append({
                    "input_ids": torch.zeros(max_para_len, dtype=torch.long),
                    "attention_mask": torch.zeros(max_para_len, dtype=torch.long),
                    "token_type_ids": torch.zeros(max_para_len, dtype=torch.long),
                })
            para_encodings = para_encodings[:target_n]

            group_data.append({
                "para_input_ids": torch.stack([p["input_ids"] for p in para_encodings]),
                "para_attention_mask": torch.stack([p["attention_mask"] for p in para_encodings]),
                "para_token_type_ids": torch.stack([p["token_type_ids"] for p in para_encodings]),
                "n_pairs": min(len(q_paras) * len(c_paras), target_n),
                "label": pair.label,
            })
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{total} PLI pairs tokenized", flush=True)

        all_group_data.append(group_data)

    print(f"    {total}/{total} PLI pairs done", flush=True)
    return all_group_data


class PLIDataset(Dataset):
    """Dataset wrapping pre-tokenized PLI paragraph-pair data."""

    def __init__(self, group_data_list):
        self.samples = []
        for gd in group_data_list:
            for item in gd:
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["para_input_ids"],
            s["para_attention_mask"],
            s["para_token_type_ids"],
            s["n_pairs"],
            torch.tensor(s["label"], dtype=torch.float),
        )


def train_bert_model(model, train_loader, epochs=EPOCHS, loss_type="ce"):
    """Train a BERT model with mixed-precision (AMP).

    Args:
        model: The BERT model to train.
        train_loader: DataLoader providing training batches.
        epochs: Number of training epochs.
        loss_type: 'ce' for cross-entropy with label smoothing (graded relevance), 
                   'binary' for class-weighted binary cross-entropy.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = GradScaler("cuda")
    binary_class_weights = torch.tensor([0.5, 3.5], device=DEVICE)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            if len(batch) == 4:
                input_ids, attn_mask, type_ids, labels = [b.to(DEVICE) for b in batch]
                optimizer.zero_grad()
                with autocast("cuda"):
                    logits = model(input_ids, attn_mask, type_ids)
                    if loss_type == "ce":
                        loss = F.cross_entropy(logits, labels.long(), label_smoothing=0.1)
                    elif loss_type == "binary":
                        loss = F.cross_entropy(logits, labels.long(), weight=binary_class_weights)
                    else:
                        loss = F.mse_loss(logits.squeeze(-1), labels)
            else:
                p_ids, p_mask, p_type, n_pairs, labels = batch
                p_ids = p_ids.to(DEVICE)
                p_mask = p_mask.to(DEVICE)
                p_type = p_type.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                with autocast("cuda"):
                    logits = model(p_ids, p_mask, p_type, n_pairs)
                    if loss_type == "ce":
                        loss = F.cross_entropy(logits, labels.long(), label_smoothing=0.1)
                    else:
                        loss = F.mse_loss(logits.squeeze(-1), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}", flush=True)

    return model


def _freeze_bert_early_layers(model):
    """Freeze embedding and first 10 of 12 encoder layers.

    Only the last two encoder layers and the task-specific head remain trainable.
    This prevents overfitting when fine-tuning on the small Korean legal case dataset.
    """
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer_idx in range(10):
        for param in model.bert.encoder.layer[layer_idx].parameters():
            param.requires_grad = False


def run_bert_ce(groups):
    """Run BERT Cross-Encoder with 5-fold pair-level stratified CV.

    Pairs are stratified by their 4-class relevance label. 
    Each fold trains a fresh BertCrossEncoder and collects predicted relevance scores on the held-out pairs. 
    Results are reassembled per query group for retrieval metric computation.
    """
    print("Loading tokenizer...", flush=True)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    all_group_data = pretokenize_all_pairs(groups, tokenizer)

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

        train_data = [[flat_items[i] for i in train_idx]]
        train_dataset = PreTokenizedDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = BertCrossEncoder().to(DEVICE)
        _freeze_bert_early_layers(model)
        model = train_bert_model(model, train_loader, epochs=EPOCHS, loss_type="ce")

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
                pred_scores[idx] = score

        del model
        torch.cuda.empty_cache()

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


def run_bert_pli(groups):
    """Run BERT-PLI with 5-fold pair-level stratified CV.

    Each query-candidate pair is decomposed into paragraph-level interactions before encoding.
    Stratification is on the 4-class relevance label.  A fresh BertPLI model is trained per fold 
    and predictions are reassembled per query group.
    """
    print("Loading tokenizer...", flush=True)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    all_group_data = pretokenize_pli_pairs(groups, tokenizer)

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

        train_samples = [flat_items[i] for i in train_idx]
        train_dataset = PLIDataset([train_samples])
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        model = BertPLI().to(DEVICE)
        _freeze_bert_early_layers(model)
        model = train_bert_model(model, train_loader, epochs=EPOCHS, loss_type="ce")

        model.eval()
        with torch.no_grad():
            for idx in test_idx:
                item = flat_items[idx]
                p_ids = item["para_input_ids"].unsqueeze(0).to(DEVICE)
                p_mask = item["para_attention_mask"].unsqueeze(0).to(DEVICE)
                p_type = item["para_token_type_ids"].unsqueeze(0).to(DEVICE)
                with autocast("cuda"):
                    score = model.predict_score(
                        p_ids, p_mask, p_type, item["n_pairs"]
                    ).item()
                pred_scores[idx] = score

        del model
        torch.cuda.empty_cache()

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


def run_finetuned_bert_binary(groups):
    """Run finetuned BERT binary classification with 5-fold pair-level CV.

    Labels are binarized (similar if graded label >= 2, dissimilar otherwise) and stratification is on this binary label. 
    A fresh BertBinaryClassifier is trained per fold with class-weighted cross-entropy to handle label imbalance.
    """
    print("Loading tokenizer...", flush=True)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    all_group_data = pretokenize_all_pairs(groups, tokenizer)

    flat_items = []
    flat_binary = []
    for gd in all_group_data:
        for item in gd:
            flat_items.append(item)
            flat_binary.append(1 if item["label"] >= 2 else 0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_true = [None] * len(flat_items)
    all_pred = [None] * len(flat_items)

    for fold, (train_idx, test_idx) in enumerate(skf.split(flat_items, flat_binary)):
        print(f"\n  Fold {fold + 1}/5 (train={len(train_idx)}, test={len(test_idx)})", flush=True)

        train_samples = [flat_items[i] for i in train_idx]
        train_dataset = PreTokenizedDataset([train_samples], binary=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = BertBinaryClassifier().to(DEVICE)
        _freeze_bert_early_layers(model)
        model = train_bert_model(model, train_loader, epochs=EPOCHS, loss_type="binary")

        model.eval()
        with torch.no_grad():
            for idx in test_idx:
                item = flat_items[idx]
                with autocast("cuda"):
                    logits = model(
                        item["input_ids"].unsqueeze(0).to(DEVICE),
                        item["attention_mask"].unsqueeze(0).to(DEVICE),
                        item["token_type_ids"].unsqueeze(0).to(DEVICE),
                    )
                all_pred[idx] = logits.argmax(dim=-1).item()
                all_true[idx] = flat_binary[idx]

        del model
        torch.cuda.empty_cache()

    return all_true, all_pred


def main():
    print("Loading dataset...", flush=True)
    groups = load_dataset()
    print(f"Using device: {DEVICE}", flush=True)

    # BERT Cross-Encoder
    print(f"\n{'=' * 60}")
    print("Running BERT (CE) Cross-Encoder...")
    print("=" * 60, flush=True)
    ce_labels, ce_scores = run_bert_ce(groups)
    ce_metrics = evaluate_retrieval(ce_labels, ce_scores)
    print("\n=== BERT (CE) Results ===")
    for k, v in ce_metrics.items():
        print(f"  {k}: {v:.2f}")

    # BERT-PLI
    print(f"\n{'=' * 60}")
    print("Running BERT-PLI...")
    print("=" * 60, flush=True)
    pli_labels, pli_scores = run_bert_pli(groups)
    pli_metrics = evaluate_retrieval(pli_labels, pli_scores)
    print("\n=== BERT-PLI Results ===")
    for k, v in pli_metrics.items():
        print(f"  {k}: {v:.2f}")

    # Finetuned BERT binary
    print(f"\n{'=' * 60}")
    print("Running Finetuned BERT (Binary Classification)...")
    print("=" * 60, flush=True)
    true_labels, pred_labels = run_finetuned_bert_binary(groups)
    binary_metrics = evaluate_binary(true_labels, pred_labels)
    print("\n=== Finetuned BERT Binary Results ===")
    for k, v in binary_metrics.items():
        print(f"  {k}: {v:.2f}")

    return {
        "BERT (CE)": ce_metrics,
        "BERT-PLI": pli_metrics,
        "finetuned BERT": binary_metrics,
    }


if __name__ == "__main__":
    main()

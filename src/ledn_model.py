"""Legal Element Decomposition Network (LEDN) for Korean legal case retrieval.

Korean legal case notes contain implicit structure: factual descriptions,
legal provisions/principles, and judicial conclusions.  Instead of encoding
the whole note as flat text, LEDN decomposes it into these three elements
and learns per-element similarity plus a fusion layer that weights their
contributions.

Architecture
------------
1. Element Decomposition  -- regex-based sentence classification into
   *facts*, *legal provisions*, and *reasoning/conclusion* segments.
2. Per-element encoding   -- shared KLUE-BERT backbone with [CLS] pooling
   produces three vectors per case (fact_vec, law_vec, reason_vec).
3. Cross-element cosine similarity -- one similarity score per element type.
4. Learned fusion         -- a linear layer combines the three cosine
   similarities into a final relevance score.
5. Training               -- MSE + pairwise margin ranking loss on graded
   labels (0--3), matching the neural_models.py pattern.

Evaluation uses 5-fold stratified pair-level cross-validation
(StratifiedKFold, n_splits=5, shuffle=True, random_state=42).
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, BertModel
from sklearn.model_selection import StratifiedKFold

from src.data_loader import QueryGroup, load_dataset
from src.metrics import evaluate_retrieval

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "klue/bert-base"
MAX_LEN = 128  # per element (shorter since each text is a segment)
EPOCHS = 5
BATCH_SIZE = 8
LR = 2e-5

# ---------------------------------------------------------------------------
# Regex patterns for element decomposition
# ---------------------------------------------------------------------------
_FACT_KEYWORDS = re.compile(
    r"피고|원고|계약|사건|행위|사실|거래|매매|손해|채무|당사자|피해"
)
_LAW_KEYWORDS = re.compile(
    r"법\s|조\s|항\s|규정|민법|상법|형법|헌법|소송법|시행령|판례|법률|법원"
)
_REASON_KEYWORDS = re.compile(
    r"판단|인정|이유|결론|타당|정당|상당|기각|각하|파기|취소|위법|적법"
)


# ---------------------------------------------------------------------------
# Element decomposition
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split Korean legal text into sentences.

    Korean legal sentences typically end with declarative endings
    (다. / 한다. / 것이다.) or section markers (【...】).
    """
    parts = re.split(r"(?<=다)\.\s*|(?<=음)\.\s*|【[^】]+】", text)
    return [s.strip() for s in parts if s and s.strip()]


def decompose_elements(text: str) -> dict[str, str]:
    """Decompose a case note into three legal elements.

    Each sentence is classified by keyword matching:
      - facts:  sentences mentioning parties, contracts, actions, etc.
      - law:    sentences referencing statutes, articles, provisions.
      - reason: sentences containing judicial reasoning / conclusions.

    A sentence may match multiple categories; it is placed in the first
    matching bucket (facts > law > reason).  Unmatched sentences go to
    the reason bucket.  If any segment is empty after classification,
    the full original text is used as fallback for that segment.

    Returns
    -------
    dict with keys ``"fact"``, ``"law"``, ``"reason"``.
    """
    sentences = _split_sentences(text)

    facts: list[str] = []
    laws: list[str] = []
    reasons: list[str] = []

    for sent in sentences:
        if _FACT_KEYWORDS.search(sent):
            facts.append(sent)
        elif _LAW_KEYWORDS.search(sent):
            laws.append(sent)
        elif _REASON_KEYWORDS.search(sent):
            reasons.append(sent)
        else:
            reasons.append(sent)  # default bucket

    fact_text = " ".join(facts) if facts else text
    law_text = " ".join(laws) if laws else text
    reason_text = " ".join(reasons) if reasons else text

    return {"fact": fact_text, "law": law_text, "reason": reason_text}


# ---------------------------------------------------------------------------
# Pre-tokenization
# ---------------------------------------------------------------------------

def pretokenize_all_pairs(
    groups: list[QueryGroup],
    tokenizer: AutoTokenizer,
    max_len: int = MAX_LEN,
) -> list[list[dict]]:
    """Pre-tokenize all query-candidate pairs with element decomposition.

    For each pair, 6 texts are tokenized (q_fact, q_law, q_reason,
    c_fact, c_law, c_reason) and stored alongside the graded label.

    Returns a list of lists (one per group), each containing dicts with
    per-element input_ids/attention_mask/token_type_ids plus label.
    """
    print("  Pre-tokenizing LEDN element pairs...", flush=True)
    all_group_data: list[list[dict]] = []
    total = sum(len(g.pairs) for g in groups)
    done = 0

    element_keys = ("fact", "law", "reason")

    for g in groups:
        q_elems = decompose_elements(g.query_note)
        q_enc = {}
        for key in element_keys:
            enc = tokenizer(
                q_elems[key],
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            q_enc[key] = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "token_type_ids": enc["token_type_ids"].squeeze(0),
            }

        group_data: list[dict] = []
        for pair in g.pairs:
            c_elems = decompose_elements(pair.candidate_note)
            item: dict = {"label": pair.label}

            for key in element_keys:
                # query element encodings (shared across candidates in the group)
                item[f"q_{key}_input_ids"] = q_enc[key]["input_ids"]
                item[f"q_{key}_attention_mask"] = q_enc[key]["attention_mask"]
                item[f"q_{key}_token_type_ids"] = q_enc[key]["token_type_ids"]

                # candidate element encodings
                c_enc = tokenizer(
                    c_elems[key],
                    max_length=max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                item[f"c_{key}_input_ids"] = c_enc["input_ids"].squeeze(0)
                item[f"c_{key}_attention_mask"] = c_enc["attention_mask"].squeeze(0)
                item[f"c_{key}_token_type_ids"] = c_enc["token_type_ids"].squeeze(0)

            group_data.append(item)
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{total} pairs tokenized", flush=True)

        all_group_data.append(group_data)

    print(f"    {total}/{total} pairs tokenized (done)", flush=True)
    return all_group_data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LEDNDataset(Dataset):
    """Flat dataset of pre-tokenized element-decomposed pairs."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            # query elements
            s["q_fact_input_ids"],
            s["q_fact_attention_mask"],
            s["q_fact_token_type_ids"],
            s["q_law_input_ids"],
            s["q_law_attention_mask"],
            s["q_law_token_type_ids"],
            s["q_reason_input_ids"],
            s["q_reason_attention_mask"],
            s["q_reason_token_type_ids"],
            # candidate elements
            s["c_fact_input_ids"],
            s["c_fact_attention_mask"],
            s["c_fact_token_type_ids"],
            s["c_law_input_ids"],
            s["c_law_attention_mask"],
            s["c_law_token_type_ids"],
            s["c_reason_input_ids"],
            s["c_reason_attention_mask"],
            s["c_reason_token_type_ids"],
            # label
            torch.tensor(s["label"], dtype=torch.float),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LEDNModel(nn.Module):
    """Legal Element Decomposition Network.

    A shared KLUE-BERT encoder produces [CLS] representations for each of the
    three legal elements (facts, legal provisions, reasoning) of both the
    query and candidate case notes.  Per-element cosine similarities are
    combined through a learned linear fusion layer.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.fusion = nn.Linear(3, 1)  # w1*fact_sim + w2*law_sim + w3*reason_sim + bias

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of texts and return projected [CLS] vectors."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_vec = outputs.last_hidden_state[:, 0, :]  # (B, H)
        return self.projection(cls_vec)

    def forward(
        self,
        q_fact_ids: torch.Tensor, q_fact_mask: torch.Tensor, q_fact_types: torch.Tensor,
        q_law_ids: torch.Tensor, q_law_mask: torch.Tensor, q_law_types: torch.Tensor,
        q_reason_ids: torch.Tensor, q_reason_mask: torch.Tensor, q_reason_types: torch.Tensor,
        c_fact_ids: torch.Tensor, c_fact_mask: torch.Tensor, c_fact_types: torch.Tensor,
        c_law_ids: torch.Tensor, c_law_mask: torch.Tensor, c_law_types: torch.Tensor,
        c_reason_ids: torch.Tensor, c_reason_mask: torch.Tensor, c_reason_types: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fused relevance score from per-element cosine similarities.

        Returns a (B,) tensor of scalar scores.
        """
        q_fact_vec = self._encode(q_fact_ids, q_fact_mask, q_fact_types)
        q_law_vec = self._encode(q_law_ids, q_law_mask, q_law_types)
        q_reason_vec = self._encode(q_reason_ids, q_reason_mask, q_reason_types)

        c_fact_vec = self._encode(c_fact_ids, c_fact_mask, c_fact_types)
        c_law_vec = self._encode(c_law_ids, c_law_mask, c_law_types)
        c_reason_vec = self._encode(c_reason_ids, c_reason_mask, c_reason_types)

        fact_sim = F.cosine_similarity(q_fact_vec, c_fact_vec, dim=-1)      # (B,)
        law_sim = F.cosine_similarity(q_law_vec, c_law_vec, dim=-1)         # (B,)
        reason_sim = F.cosine_similarity(q_reason_vec, c_reason_vec, dim=-1)  # (B,)

        sims = torch.stack([fact_sim, law_sim, reason_sim], dim=-1)  # (B, 3)
        score = self.fusion(sims).squeeze(-1)  # (B,)
        return score


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
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
# Freeze early BERT layers
# ---------------------------------------------------------------------------

def _freeze_bert_early_layers(model: LEDNModel) -> None:
    """Freeze embedding and first 10 of 12 encoder layers.

    Only the last two encoder layers, the projection head, and the
    fusion layer remain trainable.
    """
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer_idx in range(10):
        for param in model.bert.encoder.layer[layer_idx].parameters():
            param.requires_grad = False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ledn(
    model: LEDNModel,
    train_loader: DataLoader,
    epochs: int = EPOCHS,
) -> LEDNModel:
    """Train LEDN with MSE + pairwise ranking loss under mixed precision.

    MSE teaches absolute label prediction; ranking loss teaches ordering.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=0.01,
    )
    scaler = GradScaler("cuda")
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            *tensors, labels = batch
            tensors = [t.to(DEVICE) for t in tensors]
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                scores = model(*tensors)
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
# 5-fold cross-validated evaluation
# ---------------------------------------------------------------------------

def run_ledn(
    groups: list[QueryGroup],
) -> tuple[list[list[int]], list[list[float]]]:
    """Run LEDN with 5-fold stratified pair-level cross-validation.

    Steps
    -----
    1. Pre-tokenize all element-decomposed pairs once.
    2. Flatten pairs with group-index tracking.
    3. Run 5-fold StratifiedKFold on the flattened pairs.
    4. Per fold: train a fresh LEDNModel and score held-out pairs.
    5. Reassemble per-query label / score lists after all folds.
    """
    print("Loading KLUE-BERT tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
        train_dataset = LEDNDataset(train_data)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        )

        model = LEDNModel().to(DEVICE)
        _freeze_bert_early_layers(model)
        model = train_ledn(model, train_loader)

        # -- score held-out pairs ---------------------------------------------
        model.eval()
        with torch.no_grad():
            for idx in test_idx:
                item = flat_items[idx]
                with autocast("cuda"):
                    score = model(
                        item["q_fact_input_ids"].unsqueeze(0).to(DEVICE),
                        item["q_fact_attention_mask"].unsqueeze(0).to(DEVICE),
                        item["q_fact_token_type_ids"].unsqueeze(0).to(DEVICE),
                        item["q_law_input_ids"].unsqueeze(0).to(DEVICE),
                        item["q_law_attention_mask"].unsqueeze(0).to(DEVICE),
                        item["q_law_token_type_ids"].unsqueeze(0).to(DEVICE),
                        item["q_reason_input_ids"].unsqueeze(0).to(DEVICE),
                        item["q_reason_attention_mask"].unsqueeze(0).to(DEVICE),
                        item["q_reason_token_type_ids"].unsqueeze(0).to(DEVICE),
                        item["c_fact_input_ids"].unsqueeze(0).to(DEVICE),
                        item["c_fact_attention_mask"].unsqueeze(0).to(DEVICE),
                        item["c_fact_token_type_ids"].unsqueeze(0).to(DEVICE),
                        item["c_law_input_ids"].unsqueeze(0).to(DEVICE),
                        item["c_law_attention_mask"].unsqueeze(0).to(DEVICE),
                        item["c_law_token_type_ids"].unsqueeze(0).to(DEVICE),
                        item["c_reason_input_ids"].unsqueeze(0).to(DEVICE),
                        item["c_reason_attention_mask"].unsqueeze(0).to(DEVICE),
                        item["c_reason_token_type_ids"].unsqueeze(0).to(DEVICE),
                    ).item()
                pred_scores[idx] = score

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- reassemble per-query results -----------------------------------------
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
    """Entry point: load data, run LEDN CV, and print results."""
    print("Loading dataset...", flush=True)
    groups = load_dataset()
    print(f"Using device: {DEVICE}", flush=True)

    print(f"\n{'=' * 60}")
    print("Running LEDN (Legal Element Decomposition Network)...")
    print("=" * 60, flush=True)

    all_labels, all_scores = run_ledn(groups)
    metrics = evaluate_retrieval(all_labels, all_scores)

    print("\n=== LEDN Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")

    return metrics


if __name__ == "__main__":
    main()

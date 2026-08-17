# KCC: Korean Civil Case Dataset for Legal Information Retrieval

Benchmark experiments for the KCC dataset (Cho et al., SIGIR 2026).
Implements traditional IR, neural, Transformer-based, and prompt engineering
baselines evaluated on graded relevance retrieval and binary classification.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Dataset

The released benchmark lives in [`dataset_anonymized/`](dataset_anonymized/):
20 JSON files, one per query case, named `KoCivSCMdataset_<query precedent number>.json`.
Together they contain 2,939 query–candidate pairs with four-level graded
relevance labels (label 3: 201, label 2: 172, label 1: 349, label 0: 2,217).

Each file is a JSON object mapping a pair id to a record with 15 fields:

| Field | Description |
|-------|-------------|
| `query_caseNumber`, `candidate_caseNumber` | Court case number (e.g. `74다2256`) |
| `query_caseName`, `candidate_caseName` | Case name (lawsuit objective) |
| `query_precedentNumber`, `candidate_precedentNumber` | Internal precedent id |
| `query_precedentAbstract`, `candidate_precedentAbstract` | Headnote / abstract |
| `query_precedentNote`, `candidate_precedentNote` | Case note (used by the baselines) |
| `query_precedentText`, `candidate_precedentText` | Full decision text |
| `query_sentenceDate`, `candidate_sentenceDate` | Decision date |
| `label` | Graded relevance, 0–3 (see [ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md)) |

For binary evaluation, labels {2, 3} map to *similar* and {0, 1} to *dissimilar*.

**Anonymization.** Party, attorney, and law-firm names are replaced with
role-preserving placeholders (원고1, 피고1, 변호사1, 로펌1, …) — 22,272
redactions in total, produced by [`scripts/anonymize.py`](scripts/anonymize.py).
Aggregate audit statistics are in `dataset_anonymized/_audit/summary.json`.

**Annotation guidelines.** The full annotation protocol (relevance criteria,
annotator qualifications, procedure, reliability statistics, and a worked
example) is in [ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md).

To run the benchmark code below, point it at the released data by placing the
JSON files in `dataset/` (e.g. `cp dataset_anonymized/*.json dataset/`).

## Running experiments

### Run all models

```bash
uv run python -m src.run_all
```

### Run specific models

```bash
uv run python -m src.run_all -m bm25
uv run python -m src.run_all -m bm25 -m cnn -m lstm
uv run python -m src.run_all -m bert_ce -m bert_pli -m bert_binary
uv run python -m src.run_all -m lcube
uv run python -m src.run_all -m prompt
```

Available model keys: `bm25`, `cnn`, `lstm`, `bert_ce`, `bert_pli`,
`bert_binary`, `lcube`, `prompt`.

### Train word embeddings (required before running CNN/LSTM)

```bash
uv run python -m src.train_embeddings
```

This trains Word2Vec and FastText on all legal text in the dataset.
Models are saved to `results/embeddings/`.

### Prompt-based methods (Legal-CoT, Legal-Syllogism)

Requires an OpenAI API key:

```bash
export OPENAI_API_KEY=your_key
uv run python -m src.run_all -m prompt
```

Results are cached to `results/prompt_cache/` for resumability.

### Custom output path

```bash
uv run python -m src.run_all -m bm25 -o results/bm25_only.json
```

## Models

| Key | Model | Type |
|-----|-------|------|
| `bm25` | BM25 | Traditional IR |
| `cnn` | 1D-CNN | Neural Network |
| `lstm` | LSTM | Neural Network |
| `bert_ce` | BERT (Cross-Encoder) | Transformer |
| `bert_pli` | BERT-PLI | Transformer |
| `lcube` | LCube (Cross-Encoder) | Transformer |
| `bert_binary` | Finetuned BERT | Transformer |
| `sbert` | Sentence-BERT (Bi-Encoder) | Bi-Encoder |
| `simcse` | SimCSE | Contrastive Learning |
| `bm25_rerank` | BM25 + Cross-Encoder | Hybrid |
| `colbert` | ColBERT | Late Interaction |
| `ocrn` | OCRN (Ordinal Contrastive Ranking) | Novel |
| `ledn` | LEDN (Legal Element Decomposition) | Novel |
| `mghm` | MGHM (Multi-Granularity Hierarchical) | Novel |
| `prompt` | Legal-CoT, Legal-Syllogism | Prompt Engineering |

## Output

Results are printed as formatted tables and saved to `results/all_results.json`.

## Project structure

```
dataset_anonymized/         Released KCC benchmark (one JSON per query case)
ANNOTATION_GUIDELINES.md    Annotation protocol and relevance criteria
scripts/anonymize.py        Anonymization pipeline used to produce the release
src/
  data_loader.py            Dataset loading and query group construction
  metrics.py                P@K, R@K, nDCG@K, Accuracy, F1
  bm25_baseline.py          BM25 with Kiwi Korean tokenization
  neural_models.py          1D-CNN and LSTM with pretrained FastText embeddings
  bert_models.py            BERT Cross-Encoder, BERT-PLI, finetuned BERT binary
  lcube_model.py            LCube cross-encoder (lbox/lcube-base)
  sbert_model.py            Sentence-BERT bi-encoder with contrastive loss
  simcse_model.py           SimCSE unsupervised + supervised contrastive learning
  hybrid_model.py           BM25 first-stage + BERT cross-encoder re-ranking
  colbert_model.py          ColBERT late interaction (token-level MaxSim)
  ocrn_model.py             Ordinal contrastive ranking with graded margins
  ledn_model.py             Legal element decomposition (fact/law/reasoning)
  mghm_model.py             Multi-granularity hierarchical matching
  prompt_models.py          Legal-CoT and Legal-Syllogism via LangChain (GPT-4o)
  train_embeddings.py       Word2Vec and FastText training on legal corpus
  run_all.py                Main experiment runner with CLI
results/                    Output directory for results and trained models
```

## License

- **Dataset** (`dataset_anonymized/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code**: [MIT](LICENSE)

## Citation

```bibtex
@inproceedings{cho2026kcc,
  title     = {{KCC}: Korean Civil Case Dataset for Legal Information Retrieval},
  author    = {Cho, Minhan and Park, Soyoung and Sundar, S. Shyam and Choi, Daejin and Han, Jinyoung},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on
               Research and Development in Information Retrieval (SIGIR '26)},
  year      = {2026},
  doi       = {10.1145/3805712.3808594}
}
```

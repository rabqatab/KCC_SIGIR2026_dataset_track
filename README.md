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

Place the KCC JSON files in `dataset/`. Each file corresponds to one query
case and contains all query-candidate pairs with graded relevance labels (0--3).

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
| `prompt` | Legal-CoT, Legal-Syllogism | Prompt Engineering |

## Output

Results are printed as formatted tables and saved to `results/all_results.json`.

## Project structure

```
dataset/                    KCC JSON files (one per query case)
src/
  data_loader.py            Dataset loading and query group construction
  metrics.py                P@K, R@K, nDCG@K, Accuracy, F1
  bm25_baseline.py          BM25 with Kiwi Korean tokenization
  neural_models.py          1D-CNN and LSTM with pretrained FastText embeddings
  bert_models.py            BERT Cross-Encoder, BERT-PLI, finetuned BERT binary
  lcube_model.py            LCube cross-encoder (lbox/lcube-base)
  prompt_models.py          Legal-CoT and Legal-Syllogism via LangChain (GPT-4o)
  train_embeddings.py       Word2Vec and FastText training on legal corpus
  run_all.py                Main experiment runner with CLI
results/                    Output directory for results and trained models
```

"""Main experiment runner for the KCC benchmark.

Usage:
    uv run python -m src.run_all                        # run all models
    uv run python -m src.run_all --models bm25           # single model
    uv run python -m src.run_all --models bm25 cnn lstm  # subset

All supervised models use 5-fold stratified pair-level cross-validation.
Results are printed as formatted tables and saved to results/all_results.json.
"""

import json
from pathlib import Path

import click

from src.data_loader import load_dataset
from src.metrics import evaluate_retrieval, evaluate_binary

# ── Model runner wrappers ────────────────────────────────────────────────


def run_bm25(groups):
    from src.bm25_baseline import run_bm25
    labels, scores = run_bm25(groups)
    return {"retrieval": evaluate_retrieval(labels, scores)}


def run_cnn(groups):
    from src.neural_models import run_neural_model, CNN1DRanker
    labels, scores = run_neural_model(CNN1DRanker, groups, "1D-CNN")
    return {"retrieval": evaluate_retrieval(labels, scores)}


def run_lstm(groups):
    from src.neural_models import run_neural_model, LSTMRanker
    labels, scores = run_neural_model(LSTMRanker, groups, "LSTM")
    return {"retrieval": evaluate_retrieval(labels, scores)}


def run_bert_ce(groups):
    from src.bert_models import run_bert_ce
    labels, scores = run_bert_ce(groups)
    return {"retrieval": evaluate_retrieval(labels, scores)}


def run_bert_pli(groups):
    from src.bert_models import run_bert_pli
    labels, scores = run_bert_pli(groups)
    return {"retrieval": evaluate_retrieval(labels, scores)}


def run_bert_binary(groups):
    from src.bert_models import run_finetuned_bert_binary
    true_labels, pred_labels = run_finetuned_bert_binary(groups)
    return {"binary": evaluate_binary(true_labels, pred_labels)}


def run_lcube(groups):
    from src.lcube_model import run_lcube_ce
    labels, scores = run_lcube_ce(groups)
    return {"retrieval": evaluate_retrieval(labels, scores)}


def run_prompt(groups):
    from src.prompt_models import (
        run_prompt_method, LEGAL_COT_SYSTEM, LEGAL_SYLLOGISM_SYSTEM,
    )
    results = {}
    for name, prompt in [
        ("Legal-CoT", LEGAL_COT_SYSTEM),
        ("Legal-Syllogism", LEGAL_SYLLOGISM_SYSTEM),
    ]:
        labels, scores, true_bin, pred_bin = run_prompt_method(
            groups, prompt, name,
        )
        results[name] = {
            "retrieval": evaluate_retrieval(labels, scores),
            "binary": evaluate_binary(true_bin, pred_bin),
        }
    return results


# ── Registry ─────────────────────────────────────────────────────────────

MODEL_RUNNERS = {
    "bm25":        ("BM25",           run_bm25),
    "cnn":         ("1D-CNN",         run_cnn),
    "lstm":        ("LSTM",           run_lstm),
    "bert_ce":     ("BERT (CE)",      run_bert_ce),
    "bert_pli":    ("BERT-PLI",       run_bert_pli),
    "bert_binary": ("finetuned BERT", run_bert_binary),
    "lcube":       ("LCube (CE)",     run_lcube),
    "prompt":      ("Prompt Methods", run_prompt),
}

ALL_MODEL_KEYS = list(MODEL_RUNNERS.keys())

# ── Table formatting ─────────────────────────────────────────────────────

TYPE_MAP = {
    "BM25": "Traditional",
    "1D-CNN": "Neural Networks",
    "LSTM": "Neural Networks",
    "BERT-PLI": "Neural Networks",
    "BERT (CE)": "Neural Networks",
    "LCube (CE)": "Neural Networks",
    "Legal-CoT": "Prompt Engineering",
    "Legal-Syllogism": "Prompt Engineering",
}

RETRIEVAL_MODEL_ORDER = [
    "BM25", "1D-CNN", "LSTM", "BERT-PLI",
    "BERT (CE)", "LCube (CE)", "Legal-CoT", "Legal-Syllogism",
]

BINARY_MODEL_ORDER = ["finetuned BERT", "Legal-CoT ZS", "Legal-Syllogism ZS"]


def _print_retrieval_row(model_type, model_name, m):
    print(
        f"{model_type:<20} {model_name:<17} "
        f"{m.get('Precision@5', 0):8.2f} {m.get('Precision@10', 0):8.2f} "
        f"{m.get('Recall@5', 0):8.2f} {m.get('Recall@10', 0):8.2f} "
        f"{m.get('nDCG@5', 0):8.2f} {m.get('nDCG@10', 0):8.2f}"
    )


def print_retrieval_results(results):
    """Print retrieval performance for highly relevant cases."""
    print("\n" + "=" * 110)
    print("Retrieval performance for highly relevant cases (label 3) on KCC")
    print("=" * 110)
    header = (
        f"{'Types':<20} {'Model':<17} "
        f"{'P@5':>8} {'P@10':>8} {'R@5':>8} {'R@10':>8} {'nDCG@5':>8} {'nDCG@10':>8}"
    )
    print(header)
    print("-" * 110)

    for name in RETRIEVAL_MODEL_ORDER:
        if name in results:
            _print_retrieval_row(TYPE_MAP.get(name, ""), name, results[name])


def print_binary_results(results):
    """Print binary relevance evaluation."""
    print("\n" + "=" * 60)
    print("Binary relevance evaluation on KCC")
    print("=" * 60)
    header = f"{'Model':<22} {'Acc.':>8} {'F1(similar)':>14} {'F1(dissimilar)':>16}"
    print(header)
    print("-" * 60)

    for name in BINARY_MODEL_ORDER:
        if name in results:
            m = results[name]
            print(
                f"{name:<22} {m.get('Acc.', 0):8.2f} "
                f"{m.get('F1(similar)', 0):14.2f} "
                f"{m.get('F1(dissimilar)', 0):16.2f}"
            )


# ── CLI ──────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--models", "-m",
    multiple=True,
    type=click.Choice(ALL_MODEL_KEYS, case_sensitive=False),
    help="Models to run (default: all). Can be specified multiple times.",
)
@click.option(
    "--output", "-o",
    default="results/all_results.json",
    type=click.Path(),
    help="Path to save JSON results.",
)
def main(models, output):
    """KCC Benchmark Runner.

    Run baseline retrieval and classification experiments on the
    Korean Civil Case dataset.
    """
    selected = list(models) if models else ALL_MODEL_KEYS

    click.echo("Loading dataset...")
    groups = load_dataset()
    click.echo(f"Loaded {len(groups)} query groups with "
               f"{sum(len(g.pairs) for g in groups)} total pairs")

    retrieval_results = {}
    binary_results = {}

    for model_key in selected:
        display_name, runner = MODEL_RUNNERS[model_key]
        click.echo(f"\n{'#' * 60}")
        click.echo(f"# Running: {display_name}")
        click.echo(f"{'#' * 60}")

        result = runner(groups)

        if model_key == "prompt":
            for method_name, method_results in result.items():
                if "retrieval" in method_results:
                    retrieval_results[method_name] = method_results["retrieval"]
                if "binary" in method_results:
                    binary_results[f"{method_name} ZS"] = method_results["binary"]
        else:
            if "retrieval" in result:
                retrieval_results[display_name] = result["retrieval"]
            if "binary" in result:
                binary_results[display_name] = result["binary"]

    if retrieval_results:
        print_retrieval_results(retrieval_results)
    if binary_results:
        print_binary_results(binary_results)

    # Persist results
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"retrieval": retrieval_results, "binary": binary_results}, f, indent=2)
    click.echo(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

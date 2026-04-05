"""Evaluation metrics for the KCC benchmark.

Retrieval: P@K, R@K, nDCG@K with graded relevance (label 3 = highly relevant).
Binary:    Accuracy, F1(similar), F1(dissimilar).
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def precision_at_k(
    labels: list[int],
    scores: list[float],
    k: int,
    relevant_threshold: int = 3,
) -> float:
    """Compute Precision@K: fraction of top-K items that are relevant."""
    ranked_indices = np.argsort(scores)[::-1][:k]
    relevant = sum(1 for i in ranked_indices if labels[i] >= relevant_threshold)
    return relevant / k


def recall_at_k(
    labels: list[int],
    scores: list[float],
    k: int,
    relevant_threshold: int = 3,
) -> float:
    """Compute Recall@K: fraction of all relevant items found in top-K."""
    total_relevant = sum(1 for label in labels if label >= relevant_threshold)
    if total_relevant == 0:
        return 0.0
    ranked_indices = np.argsort(scores)[::-1][:k]
    found = sum(1 for i in ranked_indices if labels[i] >= relevant_threshold)
    return found / total_relevant


def ndcg_at_k(labels: list[int], scores: list[float], k: int) -> float:
    """Compute nDCG@K using graded relevance labels (0--3)."""
    ranked_indices = np.argsort(scores)[::-1][:k]
    dcg = sum(
        labels[i] / np.log2(rank + 2)
        for rank, i in enumerate(ranked_indices)
    )
    ideal_labels = sorted(labels, reverse=True)[:k]
    idcg = sum(
        label / np.log2(rank + 2)
        for rank, label in enumerate(ideal_labels)
    )
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    all_labels: list[list[int]],
    all_scores: list[list[float]],
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Evaluate retrieval across all queries and return averaged metrics.

    Computes per-query P@K, R@K, and nDCG@K, then macro-averages.
    Default K values: {5, 10}.
    """
    if ks is None:
        ks = [5, 10]

    results: dict[str, float] = {}
    for k in ks:
        p_scores = [
            precision_at_k(labels, scores, k)
            for labels, scores in zip(all_labels, all_scores)
        ]
        r_scores = [
            recall_at_k(labels, scores, k)
            for labels, scores in zip(all_labels, all_scores)
        ]
        n_scores = [
            ndcg_at_k(labels, scores, k)
            for labels, scores in zip(all_labels, all_scores)
        ]
        results[f"Precision@{k}"] = float(np.mean(p_scores))
        results[f"Recall@{k}"] = float(np.mean(r_scores))
        results[f"nDCG@{k}"] = float(np.mean(n_scores))

    return results


def evaluate_binary(
    true_labels: list[int],
    pred_labels: list[int],
) -> dict[str, float]:
    """Evaluate binary classification.

    Labels 2, 3 map to similar (1); labels 0, 1 map to dissimilar (0).
    Returns accuracy, F1(similar), and F1(dissimilar).
    """
    acc = accuracy_score(true_labels, pred_labels)
    f1_sim = f1_score(true_labels, pred_labels, pos_label=1)
    f1_dis = f1_score(true_labels, pred_labels, pos_label=0)

    return {
        "Acc.": float(acc),
        "F1(similar)": float(f1_sim),
        "F1(dissimilar)": float(f1_dis),
    }


def format_retrieval_table(results: dict[str, dict[str, float]]) -> str:
    """Format a dict of model results as a retrieval results table."""
    header = (
        f"{'Types':<20} {'Model':<15} "
        f"{'P@5':>8} {'P@10':>8} {'R@5':>8} {'R@10':>8} "
        f"{'nDCG@5':>8} {'nDCG@10':>8}"
    )
    lines = [header, "-" * len(header)]

    type_map = {
        "BM25": "Traditional",
        "1D-CNN": "Neural Networks",
        "LSTM": "Neural Networks",
        "BERT-PLI": "Neural Networks",
        "BERT (CE)": "Neural Networks",
        "LCube (CE)": "Neural Networks",
        "Legal-CoT": "Prompt Engineering",
        "Legal-Syllogism": "Prompt Engineering",
    }

    for model_name, metrics in results.items():
        model_type = type_map.get(model_name, "Unknown")
        line = (
            f"{model_type:<20} {model_name:<15} "
            f"{metrics.get('Precision@5', 0):.2f}     "
            f"{metrics.get('Precision@10', 0):.2f}     "
            f"{metrics.get('Recall@5', 0):.2f}     "
            f"{metrics.get('Recall@10', 0):.2f}     "
            f"{metrics.get('nDCG@5', 0):.2f}     "
            f"{metrics.get('nDCG@10', 0):.2f}"
        )
        lines.append(line)

    return "\n".join(lines)


def format_binary_table(results: dict[str, dict[str, float]]) -> str:
    """Format a dict of model results as a binary classification table."""
    header = (
        f"{'Model':<20} {'Acc.':>8} "
        f"{'F1(similar)':>12} {'F1(dissimilar)':>15}"
    )
    lines = [header, "-" * len(header)]

    for model_name, metrics in results.items():
        line = (
            f"{model_name:<20} "
            f"{metrics.get('Acc.', 0):.2f}     "
            f"{metrics.get('F1(similar)', 0):.2f}         "
            f"{metrics.get('F1(dissimilar)', 0):.2f}"
        )
        lines.append(line)

    return "\n".join(lines)

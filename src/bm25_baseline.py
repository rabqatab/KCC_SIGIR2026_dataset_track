"""BM25 baseline for KCC retrieval.

Unsupervised keyword-matching baseline using Okapi BM25 with Korean
morphological tokenization via Kiwi. Content words (nouns and verbs)
are extracted from each case note to form the token stream.
"""

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from src.data_loader import QueryGroup, load_dataset
from src.metrics import evaluate_retrieval

_kiwi = Kiwi()


def tokenize_korean(text: str) -> list[str]:
    """Extract content words (nouns + verbs) from Korean text."""
    return [
        t.form for t in _kiwi.tokenize(text)
        if t.tag.startswith("N") or t.tag.startswith("V")
    ]


def run_bm25(
    query_groups: list[QueryGroup],
) -> tuple[list[list[int]], list[list[float]]]:
    """Score every candidate against its query using BM25.

    Returns per-query label lists and per-query score lists.
    """
    all_labels: list[list[int]] = []
    all_scores: list[list[float]] = []

    for group in query_groups:
        candidate_tokens = [tokenize_korean(n) for n in group.candidate_notes]
        bm25 = BM25Okapi(candidate_tokens)
        query_tokens = tokenize_korean(group.query_note)
        scores = bm25.get_scores(query_tokens).tolist()

        all_labels.append(group.labels)
        all_scores.append(scores)

    return all_labels, all_scores


def main() -> None:
    """Standalone entry point for quick BM25 evaluation."""
    print("Loading dataset...")
    groups = load_dataset()

    print("Running BM25...")
    all_labels, all_scores = run_bm25(groups)

    results = evaluate_retrieval(all_labels, all_scores)

    print("\n=== BM25 Results ===")
    for key, value in results.items():
        print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()

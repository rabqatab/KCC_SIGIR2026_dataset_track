"""KCC dataset loader.

Loads the Korean Civil Case (KCC) dataset from a directory of JSON files.
Each JSON file represents one query case and contains all query-candidate pairs with graded relevance labels (0--3).
"""

import json
import os
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class CasePair:
    """A single query-candidate pair with metadata and a relevance label."""

    pair_id: str
    query_id: int
    query_case_name: str
    query_note: str
    query_abstract: str
    query_text: str
    candidate_id: float
    candidate_case_name: str
    candidate_note: str
    candidate_abstract: str
    candidate_text: str
    label: int  # 0-3 graded relevance


@dataclass
class QueryGroup:
    """All candidate pairs for a single query case."""

    query_id: int
    query_case_name: str
    query_note: str
    query_abstract: str
    query_text: str
    pairs: list[CasePair] = field(default_factory=list)

    @property
    def candidate_notes(self) -> list[str]:
        """Return the candidate notes for every pair in this group."""
        return [p.candidate_note for p in self.pairs]

    @property
    def labels(self) -> list[int]:
        """Return the graded relevance labels for every pair."""
        return [p.label for p in self.pairs]

    @property
    def binary_labels(self) -> list[int]:
        """Binarise labels: 2, 3 -> 1 (similar); 0, 1 -> 0 (dissimilar)."""
        return [1 if label >= 2 else 0 for label in self.labels]


def load_dataset(data_dir: str = "dataset") -> list[QueryGroup]:
    """Load all query groups from the dataset directory.

    Each ``.json`` file is parsed into a ``QueryGroup`` containing its
    constituent ``CasePair`` instances.
    """
    query_groups: list[QueryGroup] = []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        pairs: list[CasePair] = []
        query_info: dict | None = None

        for pair_id, entry in data.items():
            pair = CasePair(
                pair_id=pair_id,
                query_id=entry["query_precedentNumber"],
                query_case_name=entry["query_caseName"],
                query_note=entry.get("query_precedentNote", ""),
                query_abstract=entry.get("query_precedentAbstract", ""),
                query_text=entry.get("query_precedentText", ""),
                candidate_id=entry["candidate_precedentNumber"],
                candidate_case_name=entry.get("candidate_caseName", ""),
                candidate_note=entry.get("candidate_precedentNote", "") or "",
                candidate_abstract=entry.get("candidate_precedentAbstract", "") or "",
                candidate_text=entry.get("candidate_precedentText", "") or "",
                label=entry["label"],
            )
            pairs.append(pair)

            if query_info is None:
                query_info = {
                    "query_id": pair.query_id,
                    "query_case_name": pair.query_case_name,
                    "query_note": pair.query_note,
                    "query_abstract": pair.query_abstract,
                    "query_text": pair.query_text,
                }

        query_groups.append(QueryGroup(**query_info, pairs=pairs))

    return query_groups


def print_dataset_stats(groups: list[QueryGroup]) -> None:
    """Print dataset statistics matching Table 4 in the paper."""
    print(
        f"{'Case':>8} | {'type of case':<45} | "
        f"{'label 3':>7} | {'label 2':>7} | {'label 1':>7} | {'label 0':>7} | "
        f"{'Total':>5}"
    )
    print("-" * 100)

    total_counts: Counter = Counter()
    total_pairs = 0

    for g in groups:
        counts = Counter(g.labels)
        total_counts += counts
        n = len(g.pairs)
        total_pairs += n
        print(
            f"{g.query_id:>8} | {g.query_case_name:<45} | "
            f"{counts[3]:>7} | {counts[2]:>7} | {counts[1]:>7} | {counts[0]:>7} | "
            f"{n:>5}"
        )

    print("-" * 100)
    print(
        f"{'Sum':>8} | {'':<45} | "
        f"{total_counts[3]:>7} | {total_counts[2]:>7} | "
        f"{total_counts[1]:>7} | {total_counts[0]:>7} | "
        f"{total_pairs:>5}"
    )


if __name__ == "__main__":
    groups = load_dataset()
    print_dataset_stats(groups)

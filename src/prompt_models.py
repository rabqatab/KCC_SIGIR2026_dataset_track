"""Prompt-based methods for KCC using LangChain + GPT-4o.

Implements Legal-CoT and Legal-Syllogism zero-shot prompting strategies.
Each query-candidate pair is evaluated five times with majority voting
for binary classification. Prompts use case notes (precedentNote) as input.
Results are cached to disk for resumability.
"""

import json
import os
import time
from collections import Counter
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.data_loader import QueryGroup, load_dataset
from src.metrics import evaluate_binary, evaluate_retrieval

# Model and inference settings
MODEL = "gpt-4o"
TEMPERATURE = 0
TOP_P = 1.0
MAX_TOKENS = 2048

# Exact prompts from the paper
LEGAL_COT_SYSTEM = (
    "Evaluate the similarity or dissimilarity of the legal judgments "
    "in claim 1 and claim 2 by analyzing factual circumstances, "
    "legal provisions, legal judgments, and decisions in a step-by-step "
    "legal reasoning process."
)

LEGAL_SYLLOGISM_SYSTEM = (
    "Evaluate the similarity or dissimilarity of the legal judgments "
    "in claim 1 and claim 2 by Major Hypothesis(Legal Circumstance), "
    "Minor Hypothesis(Factual Description) and Conclusion."
)

ANSWER_INSTRUCTION = "Answer by 'similar' or 'dissimilar'."


def build_prompt(query_note: str, candidate_note: str,
                 system_prompt: str) -> str:
    """Build the full prompt following the paper's exact format.

    The paper specifies: {prompt} + {query precedent note} +
    {candidate precedent note} + {"Answer by 'similar' or 'dissimilar'."}
    """
    return (
        f"{system_prompt}\n"
        f"{query_note}\n"
        f"{candidate_note}\n"
        f"{ANSWER_INSTRUCTION}"
    )


def _create_llm() -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance with paper settings."""
    return ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )


def call_llm(llm: ChatOpenAI, user_content: str) -> int:
    """Invoke GPT-4o via LangChain and return 1 (similar) or 0 (dissimilar)."""
    try:
        response = llm.invoke([HumanMessage(content=user_content)])
        text = response.content.strip().lower()
        if "similar" in text and "dissimilar" not in text:
            return 1
        return 0
    except Exception as e:
        print(f"  API error: {e}")
        time.sleep(5)
        return 0


def run_prompt_method(
    groups: list[QueryGroup],
    system_prompt: str,
    method_name: str,
    n_trials: int = 5,
    cache_dir: str = "results/prompt_cache",
) -> tuple[list[list[int]], list[list[int]], list[int], list[int]]:
    """Run a prompt-based method on all query groups.

    Each pair is evaluated ``n_trials`` times and the majority vote
    determines the binary prediction.  Results are cached to disk
    so that interrupted runs can resume.
    """
    llm = _create_llm()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"{method_name}_cache.json"

    cache: dict[str, list[int]] = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)

    all_labels: list[list[int]] = []
    all_scores: list[list[int]] = []
    all_true_binary: list[int] = []
    all_pred_binary: list[int] = []

    total_pairs = sum(len(g.pairs) for g in groups)
    done = 0

    for g in groups:
        group_labels: list[int] = []
        group_scores: list[int] = []

        for pair in g.pairs:
            cache_key = f"{pair.query_id}_{pair.candidate_id}"
            prompt = build_prompt(g.query_note, pair.candidate_note, system_prompt)

            if cache_key in cache and len(cache[cache_key]) >= n_trials:
                preds = cache[cache_key][:n_trials]
            else:
                preds = cache.get(cache_key, [])
                while len(preds) < n_trials:
                    pred = call_llm(llm, prompt)
                    preds.append(pred)
                    time.sleep(0.1)
                cache[cache_key] = preds

                if done % 50 == 0:
                    with open(cache_file, "w") as f:
                        json.dump(cache, f)

            group_scores.append(preds[0])
            group_labels.append(pair.label)

            vote = Counter(preds).most_common(1)[0][0]
            binary_true = 1 if pair.label >= 2 else 0
            all_true_binary.append(binary_true)
            all_pred_binary.append(vote)

            done += 1
            if done % 100 == 0:
                print(f"  Progress: {done}/{total_pairs}")

        all_labels.append(group_labels)
        all_scores.append(group_scores)

    with open(cache_file, "w") as f:
        json.dump(cache, f)

    return all_labels, all_scores, all_true_binary, all_pred_binary


def main() -> dict | None:
    """Entry point: run both prompt methods and print results."""
    print("Loading dataset...")
    groups = load_dataset()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        print("Set it with: export OPENAI_API_KEY=your_key")
        return None

    results: dict = {}

    for method_name, system_prompt in [
        ("Legal-CoT", LEGAL_COT_SYSTEM),
        ("Legal-Syllogism", LEGAL_SYLLOGISM_SYSTEM),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Running {method_name}...")
        print("=" * 60)

        labels, scores, true_bin, pred_bin = run_prompt_method(
            groups, system_prompt, method_name,
        )

        retrieval_metrics = evaluate_retrieval(labels, scores)
        binary_metrics = evaluate_binary(true_bin, pred_bin)

        print(f"\n=== {method_name} Retrieval Results ===")
        for k, v in retrieval_metrics.items():
            print(f"  {k}: {v:.2f}")

        print(f"\n=== {method_name} ZS Binary Results ===")
        for k, v in binary_metrics.items():
            print(f"  {k}: {v:.2f}")

        results[method_name] = retrieval_metrics
        results[f"{method_name} ZS"] = binary_metrics

    return results


if __name__ == "__main__":
    main()

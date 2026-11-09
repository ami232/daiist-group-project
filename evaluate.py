"""
Evaluation. `uv run python main.py evaluate` (this file, or evaluate.ipynb)

The template only runs every case in eval/cases.csv through the pipeline and
reports what does not depend on your product: review rate per kind, latency and
cost. The TODOs are yours:

  1. each component on its own, with the metric that fits it;
  2. `is_correct()` for the end-to-end result, and the comparison with the baseline;
  3. if outputs are free text, an LLM judge, checked against eval/human_labels.csv.

Write what you report into eval/ (results.csv is written for you) and commit it,
so the numbers in REPORT.md can be traced back.
"""

import csv
import time
from pathlib import Path

import pandas as pd

import env
import pipeline

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "eval" / "cases.csv"
RESULTS = ROOT / "eval" / "results.csv"


def is_correct(case: dict, result: dict) -> bool | None:
    """TODO: is this end-to-end result correct for this case? None if code cannot tell (use the judge).

    PLACEHOLDER: compares component A's label with `expected`.
    """
    return result["outputs"]["component_a"]["label"] == case["expected"].strip()


def main() -> None:
    env.load_env()
    with CASES.open(newline="", encoding="utf-8") as f:
        cases = [r for r in csv.DictReader(f) if (r.get("input") or "").strip()]
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"  {i}/{len(cases)}", end="\r", flush=True)
        started = time.perf_counter()
        result = pipeline.run(case["input"])
        rows.append({
            "id": case["id"],
            "kind": case["kind"].strip().lower(),
            "review": result["review"],
            "correct": is_correct(case, result),
            "latency_s": round(time.perf_counter() - started, 3),
            "tokens": result["tokens"],
            "cost_eur": result["cost_eur"],
            "outputs": result["outputs"],
        })
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS, index=False)
    print(f"Wrote {RESULTS.relative_to(ROOT)} ({len(results)} cases).\n")

    by_kind = results.groupby("kind").agg(
        cases=("id", "count"),
        sent_to_review=("review", "mean"),
        correct=("correct", "mean"),
        median_latency_s=("latency_s", "median"),
        mean_cost_eur=("cost_eur", "mean"),
    )
    print(by_kind.round(3).to_string())
    print(f"\nTotal cost of this run: €{results['cost_eur'].sum():.4f}, {results['tokens'].sum()} tokens.")

    # TODO 1: each component on its own (accuracy/F1, mAP, hit rate at k...), on its own test data.
    # TODO 2: the baseline: how would the `baseline` column of eval/cases.csv have done on the same cases?
    # TODO 3: LLM judge for free-text outputs, and its agreement with eval/human_labels.csv.


if __name__ == "__main__":
    main()

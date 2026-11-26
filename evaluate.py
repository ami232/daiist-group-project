"""Evaluate team-labeled excerpts against expected facts and a manual baseline."""

import csv
import json
import math
import time
from pathlib import Path

import pandas as pd

import env
import pipeline

EVAL = Path(__file__).resolve().parent / "eval"


def matching(expected, actual):
    if isinstance(expected, (float, int)) and actual is not None:
        return math.isclose(expected, actual, abs_tol=0.01)
    return expected == actual


def main() -> None:
    env.load_env()
    with (EVAL / "cases.csv").open(newline="", encoding="utf-8") as f:
        cases = [row for row in csv.DictReader(f) if row["input"].strip()]
    if not cases:
        print("No labeled cases yet. See eval/README.md for the Tender Desk format.")
        return
    rows = []
    for case in cases:
        expected = json.loads(case["expected"])
        started = time.perf_counter()
        result = pipeline.run(case["input"], expected.get("fit", pipeline.DEFAULT_FIT))
        seconds = time.perf_counter() - started
        fields, guidance = (result["outputs"][name] for name in ("component_a", "component_b"))
        checks = [key in fields and matching(value, fields[key])
                  for key, value in expected["facts"].items()]
        actual = expected.get("actual_offers")
        predicted = guidance["estimated_offers"]
        historical_median = pipeline.metrics()["historical_median"]
        rows.append({
            "id": case["id"], "kind": case["kind"],
            "extraction_accuracy": sum(checks) / len(checks) if checks else None,
            "decision_correct": guidance["decision"] == expected["decision"],
            "end_to_end_correct": all(checks) and guidance["decision"] == expected["decision"],
            "baseline_decision_correct": case["baseline"].strip() == expected["decision"],
            "offer_absolute_error": abs(predicted - actual) if actual is not None and predicted is not None else None,
            "median_absolute_error": abs(historical_median - actual) if actual is not None and predicted is not None else None,
            "review": result["review"], "estimated_offers": predicted,
            "latency_s": round(seconds, 3), "tokens": result["tokens"], "cost_eur": result["cost_eur"],
            "outputs": json.dumps(result["outputs"]),
        })
    results = pd.DataFrame(rows)
    results.to_csv(EVAL / "results.csv", index=False)
    print(results.groupby("kind").agg(
        cases=("id", "count"), extraction_accuracy=("extraction_accuracy", "mean"),
        decision_accuracy=("decision_correct", "mean"), manual_accuracy=("baseline_decision_correct", "mean"),
        end_to_end_accuracy=("end_to_end_correct", "mean"), review_rate=("review", "mean"),
        offer_mae=("offer_absolute_error", "mean"), median_mae=("median_absolute_error", "mean"),
        mean_latency_s=("latency_s", "mean"), mean_tokens=("tokens", "mean"),
        mean_cost_eur=("cost_eur", "mean"),
    ).round(4).to_string())


if __name__ == "__main__":
    main()

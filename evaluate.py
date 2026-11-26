"""Score team-authored cases. Run with uv run python main.py evaluate."""

import csv
import json
import time
from pathlib import Path

import pandas as pd

import env
import pipeline

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "eval" / "cases.csv"
RESULTS = ROOT / "eval" / "results.csv"


def compare(case, result):
    expected = json.loads(case["expected"])
    baseline = json.loads(case["baseline"])
    a, b = (result["outputs"][name] for name in ("component_a", "component_b"))
    vision_expected = expected.get("vision_ids")
    predicted = [match.get("product_id") for match in b["matches"]]
    expected_ids = expected.get("product_ids")
    route = b["route"]
    baseline_meters = baseline.get("meters")
    success = (
        result["review"] == expected["review"]
        and (expected_ids is None or predicted == expected_ids)
        and (expected["review"] or route is not None)
        and result["review_reason"] not in {"missing_key", "ai_unavailable", "gemini_credentials", "gemini_quota"}
    )
    return {
        "vision_exact_catalog_items": None if vision_expected is None else a.get("recognized_catalog_items", []) == vision_expected,
        "agent_exact_product_ids": None if expected_ids is None else predicted == expected_ids,
        "correct": success,
        "baseline_correct": baseline.get("correct"),
        "route_meters": route["meters"] if route else None,
        "baseline_meters": baseline_meters,
        "meters_saved": baseline_meters - route["meters"] if success and route and baseline_meters is not None else None,
    }


def main():
    env.load_env()
    if env.missing():
        raise SystemExit("Set GEMINI_API_KEY in .env before running the evaluation.")
    with CASES.open(newline="", encoding="utf-8-sig") as source:
        cases = [row for row in csv.DictReader(source) if row.get("input", "").strip()]
    if not cases:
        raise SystemExit("Add your team's cases to eval/cases.csv. See eval/README.md.")
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = pipeline.run(case["input"])
        if result["review_reason"] in {"missing_key", "gemini_credentials", "gemini_quota"}:
            raise SystemExit(f"Stopped at {case['id']} ({result['review_reason']}). Resolve the Gemini issue and rerun.")
        a, b = (result["outputs"][name]["usage"] for name in ("component_a", "component_b"))
        expected = json.loads(case["expected"])
        agent_items = expected.get("agent_items")
        independent_agent = None
        if agent_items is not None:
            if "product_ids" not in expected:
                raise ValueError(f"Case {case['id']} needs product_ids to score agent_items.")
            standalone_matches = pipeline.evaluate_agent(agent_items)
            independent_agent = [match.get("product_id") for match in standalone_matches] == expected["product_ids"]
        rows.append({
            "agent_standalone_exact": independent_agent,
            "id": case["id"],
            "kind": case["kind"],
            "review": result["review"],
            "review_reason": result["review_reason"],
            **compare(case, result),
            "latency_s": round(time.perf_counter() - started, 3),
            "tokens": result["tokens"],
            "estimated_eur": result["cost_eur"],
            "vision_tokens": a["input_tokens"] + a["output_tokens"],
            "vision_eur": a["estimated_eur"],
            "vision_seconds": a["seconds"],
            "agent_tokens": b["input_tokens"] + b["output_tokens"],
            "agent_eur": b["estimated_eur"],
            "agent_seconds": b["seconds"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS, index=False)
    print(f"Scored {len(frame)} end-to-end requests in {RESULTS.relative_to(ROOT)}.")
    metrics = ["vision_exact_catalog_items", "agent_standalone_exact", "agent_exact_product_ids", "correct", "baseline_correct",
               "review", "meters_saved", "latency_s", "tokens", "estimated_eur"]
    grouped = frame[metrics].apply(pd.to_numeric).groupby(frame["kind"])
    print(grouped.mean().round(3).to_string())
    print("\nCases contributing to each metric:")
    print(grouped.count().to_string())


if __name__ == "__main__":
    main()

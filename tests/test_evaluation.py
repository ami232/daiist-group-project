"""Tests for the WP4 evaluation code (no API calls)."""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import evaluate  # noqa: E402
import pipeline  # noqa: E402
from baselines import B2, cost_per_1000, load  # noqa: E402
from maintai import inputs  # noqa: E402
from maintai.schemas import OutOfScope  # noqa: E402


def cases():
    with (ROOT / "eval" / "cases.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_cases_meet_the_minimums():
    kinds = [c["kind"] for c in cases()]
    assert len(kinds) >= 30
    assert kinds.count("hard") >= 5 and kinds.count("review") >= 5 and kinds.count("out_of_scope") >= 3


def test_out_of_scope_cases_are_exactly_the_refused_ones():
    for c in cases():
        refused = isinstance(inputs.parse(c["input"]), OutOfScope)
        assert refused == (c["kind"] == "out_of_scope"), c["id"]


def test_parse_expected():
    assert evaluate.parse_expected("refuse") == {"decision": "refuse"}
    assert evaluate.parse_expected("decision=intervene; mode=HDF; review=any") == \
        {"decision": "intervene", "mode": "HDF", "review": "any"}


def test_is_correct_out_of_scope_needs_refusal_without_tokens():
    case = next(c for c in cases() if c["kind"] == "out_of_scope")
    result = pipeline.run(case["input"])
    assert evaluate.is_correct(case, result)
    assert not evaluate.is_correct(case, {**result, "tokens": 10})


def test_is_correct_review_yes_fails_without_review():
    case = {"expected": "review=yes"}
    result = pipeline.run(cases()[0]["input"])
    assert evaluate.is_correct(case, {**result, "review": True})
    assert not evaluate.is_correct(case, {**result, "review": False})


def test_cost_per_1000_matches_the_outcome_table():
    # one prevented failure (300 + 0.2 * 4000), one false alarm (300), one missed failure (4000), one TN (0)
    intervene = np.array([True, True, False, False])
    failed = np.array([True, False, True, False])
    assert cost_per_1000(intervene, failed) == 1000 * (1100 + 300 + 4000 + 0) / 4


def test_b2_switched_off_limits_never_fire():
    off = B2(wear_min=10_000, torque_nm=10_000, rpm_max=-1, temp_diff_k=-100)
    assert not off.decide(load("validation")).any()
    assert off.describe() == "never intervene"


def test_cohen_kappa():
    assert evaluate.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert evaluate.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0

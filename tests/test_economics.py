"""
Tests for the WP2 cost model. `uv run python tests/test_economics.py` (pytest-compatible too).
Synthetic data only, no API calls.
"""

import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maintai import economics as eco  # noqa: E402

A = eco.DEFAULT


def test_cost_vector_matches_outcome_cost_for_every_case():
    for intervened, failed, reviewed in itertools.product([False, True], repeat=3):
        want = eco.outcome_cost(intervened, failed, reviewed, llm_eur=0.002)
        got = eco.cost_vector([failed], [intervened], [reviewed], 0.002)[0]
        assert abs(want - got) < 1e-9, (intervened, failed, reviewed)


def test_outcome_table_values():
    assert abs(eco.outcome_cost(True, True, False) - 1100) < 1e-9  # TP
    assert eco.outcome_cost(True, False, False) == 300  # FP
    assert eco.outcome_cost(False, True, False) == 4000  # FN
    assert eco.outcome_cost(False, False, False) == 0  # TN


def test_break_even_is_where_intervening_stops_paying():
    p = eco.break_even_probability()
    assert abs(p - 0.09375) < 1e-12
    act = A.intervention_eur + p * (1 - A.prevention_effectiveness) * A.failure_eur
    wait = p * A.failure_eur
    assert abs(act - wait) < 1e-9


def test_policy_summary_counts_and_cost_per_1000():
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    d = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    r = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    s = eco.policy_summary(y, d, r)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 1, 1, 7)
    assert s["recall"] == 0.5 and s["precision"] == 0.5
    assert abs(s["cost_per_1000"] - (1100 + 300 + 4000 + 15) / 10 * 1000) < 1e-6
    assert s["reviews_per_1000"] == 100 and s["missed_failures_per_1000"] == 100


def test_b0_cost_is_failures_times_cf():
    y = np.zeros(1000, bool)
    y[:34] = True
    s = eco.policy_summary(y, np.zeros(1000, bool))
    assert abs(s["cost_per_1000"] - 34 * 4000) < 1e-6 and s["recall"] == 0.0


def test_paired_bootstrap_identical_policies_and_clear_winner():
    rng = np.random.default_rng(1)
    base = rng.choice([0.0, 300.0, 4000.0], size=2000, p=[0.8, 0.15, 0.05])
    same = eco.paired_saving(base, base.copy(), n_boot=300)
    assert same["saving_per_1000"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = eco.paired_saving(base, base * 0.5, n_boot=300)
    assert better["ci_low"] > 0 and abs(better["relative_saving"] - 0.5) < 1e-9
    worse = eco.paired_saving(base, base * 1.5, n_boot=300)
    assert worse["ci_high"] < 0


def test_paired_bootstrap_is_reproducible_and_validates_input():
    a, b = np.arange(100.0), np.arange(100.0) * 0.9
    assert eco.paired_saving(a, b, 200, seed=3) == eco.paired_saving(a, b, 200, seed=3)
    for bad in ((a, b[:50]), (a[:0], b[:0])):
        try:
            eco.paired_saving(*bad)
        except ValueError:
            continue
        raise AssertionError("should raise ValueError")


def test_tuned_threshold_is_near_break_even_with_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.beta(0.6, 8, size=60000)
    y = rng.random(60000) < p  # perfectly calibrated
    t = eco.tune_threshold(p, y, review_rule=lambda p, t: np.zeros(len(p), bool))["threshold"]
    assert abs(t - eco.break_even_probability()) < 0.03, t


def test_tune_cutoff_finds_the_rule_boundary():
    wear = np.arange(0, 250, dtype=float).repeat(4)
    y = wear >= 200  # failures above 200 min
    assert eco.tune_cutoff(wear, y)["cutoff"] == 200.0


def test_daily_load_and_monthly_value():
    load = eco.daily_load(0.05, 0.10, 2000)
    assert load["reviews_per_day"] == 100 and load["review_hours_per_day"] == 25 and load["interventions_per_day"] == 200
    assert eco.monthly_value(1000.0, 1000, 30) == 30000.0
    assert eco.daily_load(0.05, 0.1)["processes_per_day"] == A.processes_per_day


def test_assumptions_are_the_approved_ones():
    assert (A.intervention_eur, A.failure_eur, A.prevention_effectiveness, A.review_eur, A.processes_per_day) == \
        (300.0, 4000.0, 0.8, 15.0, 1000)
    assert replace(A, review_eur=0.0).review_eur == 0.0 and A.review_eur == 15.0  # frozen defaults untouched


def test_fx_and_prices_have_a_dated_source():
    assert abs(eco.USD_TO_EUR - 1 / eco.USD_PER_EUR) < 1e-4
    assert "17 Nov 2026" in eco.FX_SOURCE and "27 Nov 2026" in eco.PRICE_SOURCE
    cost = eco.llm_cost_eur("gemini-3.5-flash-lite", 1_000_000, 1_000_000)
    assert abs(cost - round((0.30 + 2.50) * eco.USD_TO_EUR, 6)) < 1e-9
    assert eco.llm_cost_eur("unknown-model", 1000, 1000) == 0.0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
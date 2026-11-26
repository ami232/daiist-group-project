"""
Bandit simulator (WP6). `uv run python eval/simulate_bandit.py`

SIMULATED DATA ONLY. No API calls, no real users. Nothing here is evidence of how
real maintenance teams rate the work orders, and none of it is fed to the live
bandit (production uses the neutral prior Beta(1, 1) and real ratings only).

What it answers:
1. Does the Thompson-sampling code learn? It runs the same functions as the live
   app (bandit.posterior_from_counts, thompson_choice, selection_probabilities)
   against simulated users with declared like-rates, and compares with a
   uniform-random policy (cumulative regret).
2. Which prior should production use? It compares three priors:
   - neutral       Beta(1, 1) for every arm (no pseudo-observations)
   - capped_plan   the master plan's simulated prior: 10 pseudo-observations per arm
                   split by ASSUMED rates (checklist 0.60, cost 0.50, root_cause 0.55)
   - wrong_belief  a strongly biased prior (checklist 0.90 at the 10 cap), to show that
                   Thompson sampling corrects a wrong initial belief

Outputs (all labelled source = "simulated"):
    eval/bandit_simulation.csv           mean learning curves per scenario and prior
    eval/bandit_simulation_summary.json  comparison table and the production choice
Reproducible: fixed seeds.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maintai import bandit  # noqa: E402
from maintai.schemas import ARMS  # noqa: E402

EVAL = ROOT / "eval"
PSEUDO = bandit.MAX_PRIOR_PSEUDO  # 10

# ASSUMPTIONS for the simulated priors (never used live).
PLAN_RATES = {"checklist": 0.60, "cost": 0.50, "root_cause": 0.55}
WRONG_RATES = {"checklist": 0.90, "cost": 0.50, "root_cause": 0.50}

# SIMULATED USERS (declared like-rates). In "one_better" the truly best arm is not the
# one either biased prior favours.
SCENARIOS = {
    "one_better": {"checklist": 0.45, "cost": 0.45, "root_cause": 0.75},
    "all_equal": {"checklist": 0.50, "cost": 0.50, "root_cause": 0.50},
}
RUNS = 100
ROUNDS = 300
RECORD_EVERY = 5
MC = 1000
SEED = 20261126
CHECKPOINTS = (30, 50, 100, 300)


def prior_from_rates(rates: dict[str, float], pseudo: int = PSEUDO) -> dict[str, tuple[float, float]]:
    """Beta(1 + pseudo*rate, 1 + pseudo*(1-rate)): the expected outcome of `pseudo` simulated ratings."""
    return {arm: (1 + pseudo * rates[arm], 1 + pseudo * (1 - rates[arm])) for arm in ARMS}


PRIORS = {
    "neutral": dict(bandit.NEUTRAL_PRIOR),
    "capped_plan": prior_from_rates(PLAN_RATES),
    "wrong_belief": prior_from_rates(WRONG_RATES),
}


def simulate(true_rates: dict[str, float], prior: dict, runs: int = RUNS, rounds: int = ROUNDS,
             seed: int = SEED, record_every: int = RECORD_EVERY, mc: int = MC) -> pd.DataFrame:
    """Mean learning curve over `runs` independent runs (same seeds for every prior: common random numbers)."""
    best_rate = max(true_rates.values())
    random_regret_per_round = best_rate - float(np.mean(list(true_rates.values())))
    rows = []
    for run in range(runs):
        rng = np.random.default_rng([seed, run])
        succ, fail = dict.fromkeys(ARMS, 0), dict.fromkeys(ARMS, 0)
        regret = 0.0
        for t in range(1, rounds + 1):
            arm = bandit.thompson_choice(bandit.posterior_from_counts(succ, fail, prior), rng)
            reward = int(rng.random() < true_rates[arm])
            (succ if reward else fail)[arm] += 1
            regret += best_rate - true_rates[arm]
            if t % record_every == 0 or t == 1:
                post = bandit.posterior_from_counts(succ, fail, prior)
                probs = bandit.selection_probabilities(post, rng, mc)
                rows.append({"run": run, "round": t, "regret_ts": regret, "regret_random": random_regret_per_round * t,
                             **{f"p_{a}": probs[a] for a in ARMS}})
    return pd.DataFrame(rows).groupby("round").mean(numeric_only=True).drop(columns=["run"]).reset_index()


def main() -> None:
    curves, comparison = [], []
    for scenario, rates in SCENARIOS.items():
        tie = len(set(rates.values())) == 1
        best = None if tie else max(rates, key=rates.get)
        for prior_name, prior in PRIORS.items():
            curve = simulate(rates, prior)
            curve.insert(0, "prior", prior_name)
            curve.insert(0, "scenario", scenario)
            curve.insert(0, "source", "simulated")
            curve["p_best"] = curve[f"p_{best}"] if best else np.nan
            curves.append(curve)
            row = {"scenario": scenario, "prior": prior_name, "best_arm": best}
            for c in CHECKPOINTS:
                at = curve[curve["round"] == c].iloc[0]
                row[f"regret_ts_{c}"] = round(float(at["regret_ts"]), 2)
                row[f"regret_random_{c}"] = round(float(at["regret_random"]), 2)
                row[f"p_best_{c}"] = None if tie else round(float(at["p_best"]), 3)
            reached = curve[curve["p_best"] >= 0.8]["round"] if best else []
            row["first_round_p_best_ge_0.8"] = int(reached.iloc[0]) if len(reached) else None
            row["final_p_select"] = {a: round(float(curve.iloc[-1][f"p_{a}"]), 3) for a in ARMS}
            comparison.append(row)

    pd.concat(curves, ignore_index=True).round(4).to_csv(EVAL / "bandit_simulation.csv", index=False)
    summary = {
        "source": "simulated",
        "note": "Simulated users with declared like-rates. Not real feedback. Not used by the live bandit.",
        "runs": RUNS, "rounds": ROUNDS, "seed": SEED,
        "scenarios": SCENARIOS,
        "priors": {k: {a: [round(x, 3) for x in v] for a, v in p.items()} for k, p in PRIORS.items()},
        "comparison": comparison,
        "production_prior": "neutral",
        "production_prior_reason": (
            "No evidence yet that one style is better. In simulation the capped plan prior only helps when its "
            "assumed ranking happens to be right, and a wrong prior costs extra regret before real ratings "
            "correct it. With a neutral prior every learned preference comes from real ratings, so the "
            "simulated/real separation is exact."),
    }
    (EVAL / "bandit_simulation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = pd.DataFrame(comparison)[["scenario", "prior", "p_best_50", "p_best_100", "p_best_300",
                                      "first_round_p_best_ge_0.8", "regret_ts_100", "regret_ts_300", "regret_random_300"]]
    print(table.to_string(index=False))
    print("\nWrote eval/bandit_simulation.csv and eval/bandit_simulation_summary.json (SIMULATED).")


if __name__ == "__main__":
    main()

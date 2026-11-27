"""
Calibration of the bandit's 95% credible intervals (WP6). `uv run python eval/bandit_calibration.py`

SIMULATED DATA ONLY. No API calls, no real users, nothing fed to the live policy.

Question: when the dashboard shows "estimated reward 62% [41%-80%]", how often does
the 95% interval contain the true like-rate? Thompson sampling collects data
adaptively (good styles get more ratings), so frequentist coverage of a Bayesian
interval is not guaranteed; this measures it with the production code
(bandit.posterior_from_counts, thompson_choice, the same Beta quantiles).

Scenarios (true like-rates, declared, never measured):
    prior_draw   each run draws the three rates from Uniform(0, 1), i.e. from the
                 neutral prior itself. Here a correct Bayesian interval covers 95%
                 on average by construction: a check of the implementation.
    one_better   checklist 0.45, cost 0.45, root_cause 0.75 (as in simulate_bandit.py)
    all_equal    0.50 for every style (no winner exists)
    low_rates    0.10, 0.15, 0.25 (users who rarely say "useful")

For every run, checkpoint and style: covered = ci_low <= true rate <= ci_high, plus
the interval width and the number of ratings that style had. Monte Carlo standard
error of a coverage estimate p over N intervals: sqrt(p (1 - p) / N).

Outputs (labelled source = "simulated"):
    eval/bandit_calibration.csv            coverage by scenario, checkpoint and style group
    eval/bandit_calibration_summary.json   the same, with settings and a plain conclusion
Reproducible: fixed seed, independent of simulate_bandit.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maintai import bandit  # noqa: E402
from maintai.schemas import ARMS  # noqa: E402

EVAL = ROOT / "eval"
SCENARIOS = {
    "prior_draw": None,  # drawn per run from the neutral prior
    "one_better": {"checklist": 0.45, "cost": 0.45, "root_cause": 0.75},
    "all_equal": {"checklist": 0.50, "cost": 0.50, "root_cause": 0.50},
    "low_rates": {"checklist": 0.10, "cost": 0.15, "root_cause": 0.25},
}
RUNS = 200
ROUNDS = 300
CHECKPOINTS = (10, 25, 50, 100, 200, 300)
SEED = 20261127
FEW_RATINGS = 10  # a style with fewer ratings than this at a checkpoint is "under-sampled"
MIN_CELL = 100  # cells with fewer intervals are reported but flagged "small sample" (coverage too noisy)


def run_scenario(name: str, rates: dict | None, runs: int = RUNS, rounds: int = ROUNDS, seed: int = SEED,
                 checkpoints: tuple = CHECKPOINTS) -> pd.DataFrame:
    """One row per run x checkpoint x style: true rate, interval, covered, ratings of that style."""
    tail = (1 - bandit.CI_LEVEL) / 2
    rows = []
    marks = set(checkpoints)
    for run in range(runs):
        rng = np.random.default_rng([seed, run, list(SCENARIOS).index(name)])
        true = rates or {a: float(rng.random()) for a in ARMS}
        best = max(true, key=true.get) if len(set(true.values())) > 1 else None
        succ, fail = dict.fromkeys(ARMS, 0), dict.fromkeys(ARMS, 0)
        for t in range(1, rounds + 1):
            post = bandit.posterior_from_counts(succ, fail, bandit.NEUTRAL_PRIOR)
            arm = bandit.thompson_choice(post, rng)
            (succ if rng.random() < true[arm] else fail)[arm] += 1
            if t in marks:
                post = bandit.posterior_from_counts(succ, fail, bandit.NEUTRAL_PRIOR)
                a = np.array([post[x]["alpha"] for x in ARMS])
                b = np.array([post[x]["beta"] for x in ARMS])
                low, high = beta_dist.ppf(tail, a, b), beta_dist.ppf(1 - tail, a, b)
                for i, x in enumerate(ARMS):
                    n = succ[x] + fail[x]
                    rows.append({"scenario": name, "run": run, "round": t, "arm": x, "true_rate": true[x],
                                 "is_best": x == best, "n_ratings": n, "ci_low": low[i], "ci_high": high[i],
                                 "width": high[i] - low[i], "covered": bool(low[i] <= true[x] <= high[i])})
    return pd.DataFrame(rows)


def _coverage(g: pd.DataFrame) -> dict:
    p, n = float(g["covered"].mean()), len(g)
    return {"intervals": n, "small_sample": n < MIN_CELL, "coverage": round(p, 4), "mc_se": round(float(np.sqrt(p * (1 - p) / n)), 4),
            "mean_width": round(float(g["width"].mean()), 4), "mean_ratings": round(float(g["n_ratings"].mean()), 1)}


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, rnd), g in raw.groupby(["scenario", "round"], sort=False):
        groups = {"all styles": g}
        if g["is_best"].any():
            groups["best style"] = g[g["is_best"]]
            groups["other styles"] = g[~g["is_best"]]
        groups[f"under-sampled (< {FEW_RATINGS} ratings)"] = g[g["n_ratings"] < FEW_RATINGS]
        for label, sub in groups.items():
            if len(sub):
                rows.append({"source": "simulated", "scenario": scenario, "round": int(rnd), "group": label, **_coverage(sub)})
    return pd.DataFrame(rows)


def main() -> None:
    raw = pd.concat([run_scenario(name, rates) for name, rates in SCENARIOS.items()], ignore_index=True)
    table = summarise(raw)
    table.to_csv(EVAL / "bandit_calibration.csv", index=False)
    overall = table[table["group"] == "all styles"]
    final = overall[overall["round"] == ROUNDS].set_index("scenario")
    solid = table[~table["small_sample"]]
    worst = solid.loc[solid["coverage"].idxmin()]
    starved = table[table["small_sample"] & table["group"].str.startswith("under-sampled") & (table["coverage"] < 0.8)]
    summary = {
        "source": "simulated",
        "note": "Simulated users with declared like-rates. Not real feedback. Not used by the live bandit.",
        "runs": RUNS, "rounds": ROUNDS, "checkpoints": list(CHECKPOINTS), "seed": SEED, "nominal": bandit.CI_LEVEL,
        "scenarios": {k: v or "Uniform(0, 1) per style and run" for k, v in SCENARIOS.items()},
        "final_coverage_all_styles": {s: float(final.loc[s, "coverage"]) for s in final.index},
        "worst_cell": {k: (v.item() if hasattr(v, "item") else v) for k, v in worst.to_dict().items()},
        "worst_cell_rule": f"lowest coverage among cells with at least {MIN_CELL} intervals",
        "starved_style_cells": starved.to_dict("records"),
        "table": table.to_dict("records"),
    }
    (EVAL / "bandit_calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    view = overall.pivot_table(index="scenario", columns="round", values="coverage", sort=False)
    print("Coverage of the 95% credible interval, all styles (SIMULATED):")
    print(view.to_string())
    for c in summary["starved_style_cells"]:
        print(f"Small sample: {c['scenario']}, round {c['round']}, {c['group']}: {c['coverage']:.0%} of {c['intervals']} intervals")
    print(f"\nWorst cell (>= {MIN_CELL} intervals): {summary['worst_cell']['scenario']}, round {summary['worst_cell']['round']}, "
          f"{summary['worst_cell']['group']}: {summary['worst_cell']['coverage']:.1%}")
    print("Wrote eval/bandit_calibration.csv and eval/bandit_calibration_summary.json (SIMULATED).")


if __name__ == "__main__":
    main()

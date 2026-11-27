"""Economic evaluation and sensitivity analysis.  OWNER: WP2.

    uv run python eval/economics_sensitivity.py [--llm-eur 0.0014] [--boot 2000]

What it does (REPORT.md, "Economic model"):
1. Scores MaintAI (component_a.assess) and the baselines B0, B1, B2 with ONE cost model (economics.py).
2. Tunes MaintAI's threshold and the baselines' limits ONLY on data/validation.csv, with the full cost model.
3. Reports costs per 1,000 processes ON THE TEST SPLIT (data/test.csv), which keeps the natural failure rate.
   It never reads eval/cases.csv, which over-represents hard cases.
4. Saving against the best baseline (B1 or B2, chosen on validation) with a paired bootstrap 95% interval.
5. Sensitivity: C_F/C_I 5-50, e 0.5-1.0, C_R 0-50 EUR, volume 200-5,000 per day; thresholds re-tuned on
   validation for every setting.

Outputs: eval/economics_results.json and eval/economics_sensitivity.csv.
Results are MODEL OUTPUTS UNDER ASSUMPTIONS, not observed savings. If component A is still a stub
(model_version starts with "stub"), the results are marked DEMO and mean nothing.

B1 and B2 below are WP2's provisional versions so the analysis can run. WP4 owns eval/baselines.py: when it
exists, agree which implementation is final and replace `provisional_baselines`.
"""

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maintai import component_a, economics  # noqa: E402
from maintai.schemas import MachineInput  # noqa: E402

COLS = {"type": "Type", "air_temperature_k": "Air temperature [K]", "process_temperature_k": "Process temperature [K]",
        "rotational_speed_rpm": "Rotational speed [rpm]", "torque_nm": "Torque [Nm]", "tool_wear_min": "Tool wear [min]"}
RATIOS = [5, 10, 20, 30, 40, 50]  # C_F / C_I, with C_I fixed at the working value
EFFECTIVENESS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REVIEW_COSTS = [0, 15, 25, 50]
VOLUMES = [200, 1000, 2000, 5000]


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / f"{name}.csv")
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    return df


def risk_scores(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """p_fail from component A for every row, and the model version that produced it."""
    ps, version = [], ""
    for _, row in df.iterrows():
        machine = MachineInput(**{k: row[c] for k, c in COLS.items()})
        risk = component_a.assess(machine)
        ps.append(risk.p_fail)
        version = risk.model_version
    return np.array(ps), version


def provisional_baselines(train_like: pd.DataFrame, y, a: economics.CostAssumptions) -> dict:
    """Tune B1 and B2 on `train_like` (validation) only. Returns functions df -> intervene (bool array)."""
    wear = train_like["Tool wear [min]"].to_numpy()
    b1_cut = economics.tune_cutoff(wear, y, a)["cutoff"]

    # B2: intervene if ANY sensor limit is crossed: wear high, torque high, rpm low or temp difference small.
    rules = {"Tool wear [min]": True, "Torque [Nm]": True, "Rotational speed [rpm]": False, "temp_diff": False}
    limits = {c: (np.inf if up else -np.inf) for c, up in rules.items()}  # start with every sensor switched off

    def flagged(df, lim):
        d = np.zeros(len(df), dtype=bool)
        for c, up in rules.items():
            d |= (df[c].to_numpy() >= lim[c]) if up else (df[c].to_numpy() <= lim[c])
        return d

    for _ in range(3):  # coordinate descent over the four limits
        for c, up in rules.items():
            grid = np.unique(np.quantile(train_like[c], np.linspace(0.005, 0.995, 120)))
            best = (limits[c], economics.cost_vector(y, flagged(train_like, limits), None, 0.0, a).mean())
            for g in grid:
                trial = {**limits, c: g}
                cost = economics.cost_vector(y, flagged(train_like, trial), None, 0.0, a).mean()
                if cost < best[1] - 1e-12:
                    best = (g, cost)
            limits[c] = best[0]

    return {"B1": (lambda df: df["Tool wear [min]"].to_numpy() >= b1_cut, {"wear_cutoff_min": b1_cut}),
            "B2": (lambda df: flagged(df, limits), {k: (None if np.isinf(v) else float(v)) for k, v in limits.items()})}


def evaluate_setting(a: economics.CostAssumptions, val, test, p_val, p_test, llm_eur, n_boot) -> dict:
    """Tune on validation, report on test, for one set of assumptions."""
    y_val, y_test = val["Machine failure"].to_numpy(bool), test["Machine failure"].to_numpy(bool)

    tuned = economics.tune_threshold(p_val, y_val, a, llm_eur=llm_eur)
    thr = tuned["threshold"]
    maint_val_cost = tuned["cost_per_1000"]

    baselines = provisional_baselines(val, y_val, a)
    val_cost = {k: economics.cost_vector(y_val, f(val), None, 0.0, a).mean() * 1000 for k, (f, _) in baselines.items()}
    main_name = min(val_cost, key=val_cost.get)  # main baseline chosen on validation, not on test
    main_fn, main_params = baselines[main_name]

    d_maint = p_test >= thr
    r_maint = economics.default_review_rule(p_test, thr)
    c_maint = economics.cost_vector(y_test, d_maint, r_maint, llm_eur, a)
    c_main = economics.cost_vector(y_test, main_fn(test), None, 0.0, a)
    c_b0 = economics.cost_vector(y_test, np.zeros(len(y_test), bool), None, 0.0, a)

    summaries = {
        "B0": economics.policy_summary(y_test, np.zeros(len(y_test), bool), None, 0.0, a),
        main_name: economics.policy_summary(y_test, main_fn(test), None, 0.0, a),
        "MaintAI": economics.policy_summary(y_test, d_maint, r_maint, llm_eur, a),
    }
    other = "B2" if main_name == "B1" else "B1"
    summaries[other] = economics.policy_summary(y_test, baselines[other][0](test), None, 0.0, a)
    saving = economics.paired_saving(c_main, c_maint, n_boot=n_boot)
    load = economics.daily_load(summaries["MaintAI"]["reviews_per_1000"] / 1000,
                                summaries["MaintAI"]["interventions_per_1000"] / 1000, a=a)
    return {"threshold": thr, "validation_cost_per_1000": {"MaintAI": maint_val_cost, **val_cost},
            "main_baseline": main_name, "baseline_params": {k: v[1] for k, v in baselines.items()},
            "test": summaries, "saving_vs_main_baseline": saving, "daily_load": load,
            "monthly_value_eur": economics.monthly_value(saving["saving_per_1000"], a=a),
            "saving_vs_b0": float((c_b0.mean() - c_maint.mean()) * 1000)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--llm-eur", type=float, default=0.0,
                    help="measured AI cost per request in EUR (default 0 = NOT measured yet, stated in the output)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()

    val, test = load_split("validation"), load_split("test")
    p_val, version = risk_scores(val)
    p_test, _ = risk_scores(test)
    demo = version.startswith("stub")
    base = economics.DEFAULT

    headline = evaluate_setting(base, val, test, p_val, p_test, args.llm_eur, args.boot)

    rows = []

    def add(sweep, label, a, volume=None):
        res = evaluate_setting(a, val, test, p_val, p_test, args.llm_eur, args.boot)
        s = res["saving_vs_main_baseline"]
        v = volume or a.processes_per_day
        rows.append({"sweep": sweep, "setting": label, "cf_over_ci": round(a.failure_eur / a.intervention_eur, 2),
                     "effectiveness": a.prevention_effectiveness, "review_eur": a.review_eur, "volume_per_day": v,
                     "threshold": res["threshold"], "main_baseline": res["main_baseline"],
                     "maintai_cost_per_1000": res["test"]["MaintAI"]["cost_per_1000"],
                     "baseline_cost_per_1000": res["test"][res["main_baseline"]]["cost_per_1000"],
                     "saving_per_1000": s["saving_per_1000"], "ci_low": s["ci_low"], "ci_high": s["ci_high"],
                     "relative_saving": s["relative_saving"], "ci_above_zero": s["ci_low"] > 0,
                     "recall": res["test"]["MaintAI"]["recall"],
                     "reviews_per_day": economics.daily_load(res["test"]["MaintAI"]["reviews_per_1000"] / 1000, 0,
                                                             v, a)["reviews_per_day"],
                     "monthly_value_eur": economics.monthly_value(s["saving_per_1000"], v, a=a)})

    for r in RATIOS:
        add("ratio", f"CF/CI={r}", replace(base, failure_eur=base.intervention_eur * r))
    for e in EFFECTIVENESS:
        add("effectiveness", f"e={e}", replace(base, prevention_effectiveness=e))
    for c in REVIEW_COSTS:
        add("review", f"CR={c}", replace(base, review_eur=float(c)))
    for v in VOLUMES:
        add("volume", f"V={v}", replace(base, processes_per_day=v), volume=v)
    for r in RATIOS:
        for e in EFFECTIVENESS:
            add("grid", f"CF/CI={r}, e={e}", replace(base, failure_eur=base.intervention_eur * r,
                                                    prevention_effectiveness=e))

    grid = [r for r in rows if r["sweep"] == "grid" and r["effectiveness"] >= 0.6 and r["cf_over_ci"] >= 10]
    criterion = {"cells": len(grid), "cells_with_ci_above_zero": sum(r["ci_above_zero"] for r in grid),
                 "rule": "criterion 1 asks for most cells with e >= 0.6 and CF/CI >= 10 (REPORT.md)"}
    summary = {
        "status": "DEMO: component A is a stub, these numbers mean nothing" if demo else "model outputs under assumptions",
        "label": "Simulated savings under assumptions. NOT observed savings.",
        "component_a_version": version,
        "assumptions": base.__dict__, "operating_days_per_month": economics.OPERATING_DAYS_PER_MONTH,
        "review_hours_per_review": economics.REVIEW_HOURS,
        "llm_eur_per_request": args.llm_eur,
        "llm_cost_note": "measured value" if args.llm_eur else "NOT measured yet: AI cost set to 0 in this run",
        "fx": economics.FX_SOURCE, "gemini_prices": economics.PRICE_SOURCE,
        "tuning_data": "data/validation.csv (threshold and baseline limits)", "reporting_data": "data/test.csv",
        "test_n": len(test), "test_failures": int(test["Machine failure"].sum()),
        "headline": headline, "criterion_1_grid": criterion,
        "limits": ["Process-level bootstrap treats rows as independent; neighbouring rows are correlated.",
                   "Baselines are assumed to need no review and no AI cost.",
                   "Test split has few failures, so intervals are wide.",
                   "Review rule mirrors component A's bands; WP1 owns the real rule."],
    }
    (ROOT / "eval" / "economics_results.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    with open(ROOT / "eval" / "economics_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    s = headline["saving_vs_main_baseline"]
    t = headline["test"]
    print(f"[{summary['status']}]  A = {version}, threshold {headline['threshold']:.2f}, main baseline {headline['main_baseline']}")
    for name in ("B0", "B1", "B2", "MaintAI"):
        x = t[name]
        print(f"  {name:8s} cost/1000 = {x['cost_per_1000']:9.0f} EUR   recall = {x['recall'] if x['recall'] is None else round(x['recall'], 3)}")
    print(f"  saving vs {headline['main_baseline']} = {s['saving_per_1000']:.0f} EUR per 1,000 "
          f"(95% CI {s['ci_low']:.0f} to {s['ci_high']:.0f}), {s['relative_saving']:.1%}")
    print(f"  reviews/day = {headline['daily_load']['reviews_per_day']:.0f}; monthly value = {headline['monthly_value_eur']:.0f} EUR (simulated)")
    print(f"  grid check (e>=0.6, CF/CI>=10): CI above zero in {criterion['cells_with_ci_above_zero']}/{criterion['cells']} cells")


if __name__ == "__main__":
    main()
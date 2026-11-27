"""
Evaluation. `uv run python main.py evaluate`  OWNER: WP4.

What it measures, and which go / no-go criterion of REPORT.md it feeds:

  1. Component A (tabular ML) on the TEST split of AI4I 2020: recall, precision,
     PR-AUC, Brier score, and whether it names the right failure mode.   -> 2
  2. Economics on the test split with the cost model of economics.py: MaintAI's
     decision rule against B0, B1 and B2 (eval/baselines.py, tuned on train +
     validation only), with a paired 95% bootstrap interval of the saving.  -> 1
  3. Component B (work-order agent) on eval/cases.csv, checked in code: valid
     order, grounded ids, action consistent with the failure mode, priority
     consistent with the risk band, review rule applied.                   -> 3, 4
  4. End to end on eval/cases.csv: is_correct() against `expected`, by kind.  -> 3, 6
  5. The free text of the orders graded by an LLM judge (eval/judge.py), for the
     agent AND for the rule-based template (the no-LLM baseline), and the judge's
     agreement with our own labels in eval/human_labels.csv (Cohen's kappa).  -> 5
  6. Tokens, euros and latency (p50 / p95) per request.                     -> 7
  7. Errors by component and by kind of case (eval/errors.csv).

Everything is written to eval/ and committed, so every number in REPORT.md can be
traced back. Results from STUB components (a version containing "stub") are
written but labelled as NOT evidence of the product's performance.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import env
import pipeline

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval"
sys.path.insert(0, str(EVAL))

from baselines import b0_decide, confusion, cost_per_1000, load, tuned
from judge import CRITERIA, RUBRIC_VERSION, Judge, build_prompt

from maintai import agent, component_a, guardrails, inputs
from maintai.schemas import (
    Action,
    FailureMode,
    MachineInput,
    OutOfScope,
    Priority,
    RiskBand,
)

CASES = EVAL / "cases.csv"
RESULTS = EVAL / "results.csv"
HUMAN_LABELS = EVAL / "human_labels.csv"
TO_LABEL = EVAL / "to_label.csv"
BOOTSTRAP = 2000
SEED = 0

# Safe actions for each failure mode, and acceptable priorities for each band.
# The template (agent.rule_based_plan) uses one of each; the agent may choose
# another safe one, but must not contradict them.
ACTION_FOR_MODE = {
    FailureMode.TWF: {Action.REPLACE_TOOL, Action.STOP_AND_INSPECT},
    FailureMode.OSF: {Action.REDUCE_LOAD, Action.REPLACE_TOOL, Action.STOP_AND_INSPECT},
    FailureMode.HDF: {Action.COOLING_CHECK, Action.STOP_AND_INSPECT},
    FailureMode.PWF: {Action.POWER_CHECK, Action.REDUCE_LOAD, Action.STOP_AND_INSPECT},
}
PRIORITY_FOR_BAND = {
    RiskBand.LOW: {Priority.P4, Priority.P3},
    RiskBand.MEDIUM: {Priority.P3, Priority.P2},
    RiskBand.HIGH: {Priority.P2, Priority.P1},
    RiskBand.CRITICAL: {Priority.P1},
}


# ------------------------------------------------------------ expected ---

def parse_expected(text: str) -> dict:
    """'decision=intervene; mode=HDF; review=any' -> {'decision': 'intervene', 'mode': 'HDF', 'review': 'any'}."""
    text = (text or "").strip()
    if text == "refuse":
        return {"decision": "refuse"}
    out = {}
    for part in text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def work_order(result: dict) -> dict:
    return result["outputs"].get("component_b", {}).get("work_order", {})


def decision_of(result: dict) -> str:
    """refuse | monitor | intervene, read from the pipeline's output."""
    if result.get("status") == "out_of_scope":
        return "refuse"
    return "monitor" if work_order(result)["draft"]["action"] == Action.MONITOR.value else "intervene"


def is_correct(case: dict, result: dict) -> bool:
    """Is this end-to-end result correct for this case? Decided fully in code.

    - out_of_scope: refused before any model call (status out_of_scope and 0 tokens);
    - decision: intervene / monitor as expected (any = either is fine);
    - mode: the order addresses the expected failure mode, when one is given;
    - review: sent to a person when expected (yes), not sent when not expected (no).
    """
    exp = parse_expected(case["expected"])
    got = decision_of(result)
    if exp.get("decision") == "refuse":
        return got == "refuse" and result.get("tokens", 0) == 0
    if got == "refuse":
        return False
    if exp.get("decision", "any") not in ("any", got):
        return False
    mode = exp.get("mode", "any")
    if mode != "any" and work_order(result)["draft"]["failure_mode_addressed"] != mode:
        return False
    review = exp.get("review", "any")
    if review == "yes" and not result["review"]:
        return False
    if review == "no" and result["review"]:
        return False
    return True


# -------------------------------------------------- component B (code) ---

def component_b_checks(result: dict) -> dict:
    """Checks on one work order that code can decide (criteria 3 and 4)."""
    if result.get("status") == "out_of_scope":
        return {}
    risk = result["outputs"]["component_a"]
    order = work_order(result)
    draft = order["draft"]
    band = RiskBand(risk["risk_band"])
    mode = FailureMode(draft["failure_mode_addressed"])
    action = Action(draft["action"])
    errors = order.get("validation_errors") or []

    must_review = (band == RiskBand.CRITICAL or bool(risk.get("ood_flags"))
                   or bool(risk.get("in_uncertainty_band")) or bool(errors))
    if action == Action.MONITOR:
        action_ok = band == RiskBand.LOW
    else:
        action_ok = action == Action.SCHEDULE_INSPECTION or action in ACTION_FOR_MODE.get(mode, set(Action))
    predicted = (risk.get("likely_modes") or [{}])[0].get("mode")
    return {
        "valid_order": not errors,
        "grounded": not any(e.startswith("unknown") for e in errors),
        "action_matches_mode": action_ok,
        "addresses_predicted_mode": action == Action.MONITOR or draft["failure_mode_addressed"] == predicted,
        "priority_matches_band": Priority(draft["priority"]) in PRIORITY_FOR_BAND[band],
        "cost_from_tool": mode == FailureMode.NONE or order.get("cost") is not None,
        "review_rule_required": must_review,
        "review_rule_applied": (not must_review) or bool(result["review"]),
    }


def template_result(raw_input: str) -> dict | None:
    """The rule-based template (no LLM) on the same input: the baseline for the work order."""
    parsed = inputs.parse(raw_input)
    if isinstance(parsed, OutOfScope):
        return None
    risk = component_a.assess(parsed)
    plan = agent.rule_based_plan(parsed, risk, reason="template baseline")
    order = guardrails.finalise(parsed, risk, plan)
    return {"risk": risk.model_dump(mode="json"), "tool_calls": [c.model_dump(mode="json") for c in plan.tool_calls],
            "draft": order.draft.model_dump(mode="json"), "review": order.review}


# -------------------------------------------------------- component A ---

def machine_from_row(row: pd.Series) -> MachineInput:
    return MachineInput(type=row["type"], air_temperature_k=row["air_temperature_k"],
                        process_temperature_k=row["process_temperature_k"],
                        rotational_speed_rpm=row["rotational_speed_rpm"], torque_nm=row["torque_nm"],
                        tool_wear_min=row["tool_wear_min"])


def component_a_on_test() -> tuple[dict, pd.DataFrame]:
    """Component A on every row of the test split, and MaintAI's decision on it."""
    from sklearn.metrics import average_precision_score, brier_score_loss

    test = load("test")
    rows = []
    for _, row in test.iterrows():
        risk = component_a.assess(machine_from_row(row))
        true_modes = [m for m in ("TWF", "HDF", "PWF", "OSF") if row.get(m) == 1]
        rows.append({
            "UDI": int(row["UDI"]), "failure": int(row["failure"]), "true_modes": ";".join(true_modes),
            "p_fail": risk.p_fail, "threshold": risk.threshold, "risk_band": risk.risk_band.value,
            "predicted_mode": risk.likely_modes[0].mode.value if risk.likely_modes else "none",
            # the deterministic part of the review rule (guardrails.finalise)
            "review": risk.risk_band == RiskBand.CRITICAL or bool(risk.ood_flags) or risk.in_uncertainty_band,
            "model_version": risk.model_version,
        })
    df = pd.DataFrame(rows)
    y, p = df["failure"].to_numpy(), df["p_fail"].to_numpy()
    df["intervene"] = p >= df["threshold"].to_numpy()
    failures = df[(df["failure"] == 1) & (df["true_modes"] != "")]
    mode_hits = [m in t.split(";") for m, t in zip(failures["predicted_mode"], failures["true_modes"])]
    metrics = {
        "rows": len(df), "failures": int(y.sum()), "threshold": float(df["threshold"].iloc[0]),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        **confusion(df["intervene"].to_numpy(), y.astype(bool)),
        "mode_accuracy_on_failures": round(float(np.mean(mode_hits)), 3) if mode_hits else None,
        "review_per_1000": round(1000 * float(df["review"].mean()), 1),
        "model_version": df["model_version"].iloc[0],
    }
    return metrics, df


def economics_on_test(a_rows: pd.DataFrame, llm_eur_per_request: float) -> dict:
    """Cost per 1,000 processes on the test split: MaintAI against the baselines (criterion 1)."""
    b1, b2, main = tuned()
    test = load("test")
    failed = test["failure"].to_numpy(bool)
    policies = {
        "B0": (b0_decide(test), None, 0.0),
        "B1": (b1.decide(test), None, 0.0),
        "B2": (b2.decide(test), None, 0.0),
        "MaintAI": (a_rows["intervene"].to_numpy(bool), a_rows["review"].to_numpy(bool), llm_eur_per_request),
    }
    out = {"main_baseline": main, "B1_rule": b1.describe(), "B2_rule": b2.describe(),
           "llm_eur_per_request_used": llm_eur_per_request, "policies": {}}
    for name, (intervene, review, llm) in policies.items():
        out["policies"][name] = {"cost_per_1000_eur": round(cost_per_1000(intervene, failed, review, llm), 1),
                                 **confusion(intervene, failed)}

    # Paired bootstrap: the same resampled rows for both policies.
    rng = np.random.default_rng(SEED)
    base = policies[main][0]
    m_int, m_rev = policies["MaintAI"][0], policies["MaintAI"][1]
    savings = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(failed), len(failed))
        savings.append(cost_per_1000(base[idx], failed[idx])
                       - cost_per_1000(m_int[idx], failed[idx], m_rev[idx], llm_eur_per_request))
    base_cost = out["policies"][main]["cost_per_1000_eur"]
    saving = base_cost - out["policies"]["MaintAI"]["cost_per_1000_eur"]
    out["saving_vs_main_per_1000_eur"] = round(saving, 1)
    out["saving_vs_main_pct"] = round(100 * saving / base_cost, 1) if base_cost else None
    out["saving_95ci_per_1000_eur"] = [round(float(np.percentile(savings, 2.5)), 1),
                                       round(float(np.percentile(savings, 97.5)), 1)]
    return out


# ----------------------------------------------------------- judge, kappa ---

def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    if not a:
        return None
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 3)


def agreement_with_humans(judged: pd.DataFrame) -> dict:
    """Judge pass/fail on the agent's orders against eval/human_labels.csv (label 1 = pass, 0 = fail)."""
    if not HUMAN_LABELS.is_file() or judged.empty:
        return {"labels": 0}
    labels = pd.read_csv(HUMAN_LABELS, dtype={"id": str, "label": str})
    labels = labels[labels["label"].fillna("").str.strip().isin(["0", "1"])]
    agent_rows = judged[(judged["writer"] == "agent") & judged["judge_pass"].notna()]
    merged = labels.merge(agent_rows, on="id")
    if merged.empty:
        return {"labels": len(labels), "compared": 0}
    human = merged["label"].astype(int).tolist()
    judge = merged["judge_pass"].astype(int).tolist()
    return {"labels": len(labels), "compared": len(merged),
            "agreement": round(float(np.mean(np.array(human) == np.array(judge))), 3),
            "cohen_kappa": cohen_kappa(human, judge),
            "human_pass_rate": round(float(np.mean(human)), 3), "judge_pass_rate": round(float(np.mean(judge)), 3),
            "disagreements": [i for i, h, j in zip(merged["id"], human, judge) if h != j]}


# ------------------------------------------------------------- errors ---

def error_cause(case: dict, result: dict, checks: dict) -> str:
    """Which part of the product a wrong end-to-end result comes from."""
    exp = parse_expected(case["expected"])
    got = decision_of(result)
    if exp.get("decision") == "refuse":
        return "input check: out-of-scope input got an order" if got != "refuse" else "input check: model called before refusing"
    if got == "refuse":
        return "input check: valid reading refused"
    risk = result["outputs"]["component_a"]
    if exp.get("review") == "yes" and not result["review"]:
        return ("guardrails: required review not sent" if checks.get("review_rule_required")
                else "component A: case not flagged (no out-of-range, critical or uncertain signal)")
    if exp.get("review") == "no" and result["review"]:
        return "review sent without need"
    want = exp.get("decision", "any")
    if want not in ("any", got):
        if (want == "intervene") != (risk["p_fail"] >= risk["threshold"]):
            return "component A: risk on the wrong side of the threshold"
        return "component B: action contradicts the risk"
    mode = exp.get("mode", "any")
    if mode != "any":
        predicted = (risk.get("likely_modes") or [{}])[0].get("mode")
        return "component A: wrong failure mode" if predicted != mode else "component B: wrong failure mode in the order"
    return "other"


# --------------------------------------------------------------- runs ---

def run_cases() -> pd.DataFrame:
    with CASES.open(newline="", encoding="utf-8") as f:
        cases = [r for r in csv.DictReader(f) if (r.get("input") or "").strip()]
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"  case {i}/{len(cases)}", end="\r", flush=True)
        started = time.perf_counter()
        result = pipeline.run(case["input"])
        latency = time.perf_counter() - started
        checks = component_b_checks(result)
        correct = is_correct(case, result)
        order = work_order(result)
        rows.append({
            "id": case["id"], "kind": case["kind"].strip().lower(), "expected": case["expected"],
            "decision": decision_of(result), "review": result["review"], "correct": correct,
            "error_cause": "" if correct else error_cause(case, result, checks),
            "p_fail": result["outputs"]["component_a"].get("p_fail"),
            "risk_band": result["outputs"]["component_a"].get("risk_band"),
            "mode_addressed": order.get("draft", {}).get("failure_mode_addressed"),
            "review_reasons": ";".join(order.get("review_reasons", [])),
            **{f"check_{k}": v for k, v in checks.items()},
            "latency_s": round(latency, 3), "tokens": result["tokens"], "cost_eur": result["cost_eur"],
            "degraded": bool(result.get("degraded", False)),
            "version_a": result["versions"].get("component_a"), "version_b": result["versions"].get("component_b"),
            "input": case["input"], "outputs": json.dumps(result["outputs"], ensure_ascii=False),
        })
    print()
    return pd.DataFrame(rows)


def run_judge(results: pd.DataFrame) -> tuple[pd.DataFrame, Judge]:
    """Judge the agent's orders and the template's orders on the same in-scope cases."""
    judge = Judge()
    rows = []
    for _, r in results[results["decision"] != "refuse"].iterrows():
        outputs = json.loads(r["outputs"])
        order = outputs["component_b"]["work_order"]
        candidates = {"agent": (outputs["component_a"], outputs["component_b"].get("tool_calls", []),
                                order["draft"], bool(r["review"]))}
        template = template_result(r["input"])
        if template:
            candidates["template"] = (template["risk"], template["tool_calls"], template["draft"], template["review"])
        for writer, (risk, calls, draft, review) in candidates.items():
            verdict = judge.grade(build_prompt(r["input"], risk, calls, draft, review))
            rows.append({"id": r["id"], "kind": r["kind"], "writer": writer,
                         "judge_pass": None if verdict is None else int(verdict["pass"]),
                         **{c: None if verdict is None else int(verdict[c]) for c in CRITERIA},
                         "reason": "" if verdict is None else verdict["reason"],
                         "steps": " | ".join(draft.get("steps") or []),
                         "justification": draft.get("justification", "")})
    return pd.DataFrame(rows), judge


def write_to_label(results: pd.DataFrame) -> None:
    """The agent's orders WITHOUT any verdict, for our own blind labels (eval/label.py)."""
    results[results["decision"] != "refuse"][["id", "kind", "input", "outputs"]].to_csv(TO_LABEL, index=False)


def rate(series: pd.Series) -> float | None:
    return round(float(series.mean()), 3) if len(series) else None


# --------------------------------------------------------------- main ---

def main() -> None:
    env.load_env()
    print("1/5 End-to-end cases (eval/cases.csv)")
    results = run_cases()
    results.to_csv(RESULTS, index=False)
    write_to_label(results)
    stub = results[["version_a", "version_b"]].astype(str).apply(lambda c: c.str.contains("stub")).any().any()

    by_kind = results.groupby("kind").agg(cases=("id", "count"), correct=("correct", "mean"),
                                          sent_to_review=("review", "mean"), median_latency_s=("latency_s", "median"),
                                          mean_cost_eur=("cost_eur", "mean")).round(3)
    print(by_kind.to_string())

    in_scope = results[results["decision"] != "refuse"]
    oos = results[results["kind"] == "out_of_scope"]
    oos_ok = bool(((oos["decision"] == "refuse") & (oos["tokens"] == 0)).all())
    check_cols = [c for c in in_scope.columns if c.startswith("check_") and c != "check_review_rule_required"]
    component_b = {
        "orders": len(in_scope),
        **{c.removeprefix("check_"): rate(in_scope[c].astype(bool)) for c in check_cols},
        "review_rule_cases": int(in_scope["check_review_rule_required"].astype(bool).sum()),
        "review_rule_violations": in_scope.loc[~in_scope["check_review_rule_applied"].astype(bool), "id"].tolist(),
        "ungrounded_orders": in_scope.loc[~in_scope["check_grounded"].astype(bool), "id"].tolist(),
    }
    lat = results["latency_s"]
    operations = {
        "latency_p50_s": round(float(lat.quantile(0.5)), 3),
        "latency_p95_s": round(float(lat.quantile(0.95)), 3),
        "tokens_per_request_mean": round(float(in_scope["tokens"].mean()), 1) if len(in_scope) else 0.0,
        "eur_per_request_mean": round(float(in_scope["cost_eur"].mean()), 6) if len(in_scope) else 0.0,
        "eur_per_request_max": round(float(in_scope["cost_eur"].max()), 6) if len(in_scope) else 0.0,
        "requests_with_llm_calls": int((in_scope["tokens"] > 0).sum()),
        "degraded_share": rate(in_scope["degraded"]),
        "run_cost_eur": round(float(results["cost_eur"].sum()), 4),
        "run_tokens": int(results["tokens"].sum()),
    }
    print(json.dumps({"component_b": component_b, "operations": operations}, indent=2))

    print("\n2/5 Component A on the test split")
    a_metrics, a_rows = component_a_on_test()
    a_rows.to_csv(EVAL / "component_a_test.csv", index=False)
    print(json.dumps(a_metrics, indent=2))

    print("\n3/5 Economics on the test split (baselines tuned on train + validation)")
    economics = economics_on_test(a_rows, operations["eur_per_request_mean"])
    print(pd.DataFrame(economics["policies"]).T.to_string())
    print(f"Saving vs {economics['main_baseline']}: €{economics['saving_vs_main_per_1000_eur']} per 1,000 "
          f"({economics['saving_vs_main_pct']}%), 95% CI {economics['saving_95ci_per_1000_eur']}")

    print("\n4/5 LLM judge on the work orders (agent and no-LLM template)")
    judged, judge = run_judge(results)
    judged.to_csv(EVAL / "judge.csv", index=False)
    graded = judged.dropna(subset=["judge_pass"]) if len(judged) else judged
    judge_summary = {"model": judge.model, "rubric": RUBRIC_VERSION, "key_available": judge.available,
                     "graded": len(graded), "calls_this_run": judge.calls, "tokens_this_run": judge.tokens}
    for writer in ("agent", "template"):
        part = graded[graded["writer"] == writer] if len(graded) else graded
        judge_summary[f"{writer}_pass_rate"] = rate(part["judge_pass"]) if len(part) else None
        for c in CRITERIA:
            judge_summary[f"{writer}_{c}"] = rate(part[c]) if len(part) else None
    if not judge.available and not len(graded):
        print("No GEMINI_API_KEY and no cached verdicts: judge skipped.")
    human = agreement_with_humans(judged)
    print(json.dumps({**judge_summary, "human_agreement": human}, indent=2))

    print("\n5/5 Errors by cause and kind")
    errors = results[~results["correct"].astype(bool)][["id", "kind", "expected", "decision", "review", "error_cause"]]
    errors.to_csv(EVAL / "errors.csv", index=False)
    print(errors.groupby(["error_cause", "kind"]).size().to_string() if len(errors) else "No end-to-end errors.")

    main_b = economics["main_baseline"]
    summary = {
        "evidence": ("STUB components: these numbers test the evaluation code, NOT the product. "
                     "Re-run after WP1 and WP3 replace the stubs.") if stub else "real components",
        "versions": {"component_a": results["version_a"].iloc[0], "component_b": results["version_b"].iloc[0],
                     "judge_rubric": RUBRIC_VERSION},
        "cases": {"total": len(results), **results["kind"].value_counts().to_dict()},
        "end_to_end_correct": rate(results["correct"]),
        "by_kind": by_kind.reset_index().to_dict(orient="records"),
        "component_a": a_metrics, "economics_test_split": economics, "component_b": component_b,
        "judge": judge_summary, "human_agreement": human, "operations": operations,
        "errors_by_cause": errors["error_cause"].value_counts().to_dict(),
        "criteria": {
            "1_cost": {"saving_pct": economics["saving_vs_main_pct"], "ci_eur": economics["saving_95ci_per_1000_eur"],
                       "target": ">= 10% cheaper than the main baseline, CI above 0"},
            "2_recall": {"recall": a_metrics["recall"], "main_baseline_recall": economics["policies"][main_b]["recall"],
                         "target": ">= 0.80"},
            "3_review_rule": {"violations": component_b["review_rule_violations"], "target": "none"},
            "4_grounding": {"ungrounded": component_b["ungrounded_orders"], "target": "none"},
            "5_quality": {"agent_pass": judge_summary["agent_pass_rate"],
                          "template_pass": judge_summary["template_pass_rate"],
                          "kappa": human.get("cohen_kappa"), "human_labels": human.get("labels", 0),
                          "target": "agent pass >= 0.80 and above the template, kappa >= 0.6 on >= 20 labels"},
            "6_out_of_scope": {"all_refused_before_model": oos_ok, "target": "all"},
            "7_operations": {"p95_s": operations["latency_p95_s"], "eur_per_request": operations["eur_per_request_mean"],
                             "target": "p95 < 15 s and < €0.01 per request"},
        },
    }
    (EVAL / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\nWrote eval/results.csv, component_a_test.csv, judge.csv, errors.csv, to_label.csv, summary.json.")
    if stub:
        print("NOTE: stub components detected: these results are NOT evidence of the product's performance.")


if __name__ == "__main__":
    main()

"""
Tests for the WP6 bandit. `uv run python tests/test_bandit.py`
(pytest-compatible too: `pytest tests/test_bandit.py`, if pytest is installed).

All data here is SYNTHETIC test data built in memory or in a temporary folder.
No API calls: GEMINI_API_KEY is blanked for this process, so any agent falls
back to the rule-based template instead of calling Gemini.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

LOG_DIR = tempfile.mkdtemp(prefix="bandit-test-logs-")
os.environ["LOG_DIR"] = LOG_DIR  # before importing logbook: files in a temp folder, never Postgres
os.environ["GEMINI_API_KEY"] = ""  # never call Gemini from tests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import inspect  # noqa: E402

import pandas as pd  # noqa: E402

import logbook  # noqa: E402
import pipeline  # noqa: E402
from maintai import bandit, guardrails  # noqa: E402
from maintai.schemas import ARMS  # noqa: E402

PV = "agent-v1"  # a non-stub prompt version for the synthetic records
UNIFORM = {arm: (1.0, 1.0) for arm in ARMS}


def rec(i, arm, feedback=None, source="real", status="ok", degraded=False, versions=None, probs=True, minute=0):
    """One row shaped like logbook.load() output."""
    return {
        "id": f"r{i}", "timestamp": pd.Timestamp("2026-11-28 10:00", tz="UTC") + pd.Timedelta(minutes=minute or i),
        "arm": arm, "source": source, "status": status, "degraded": degraded,
        "versions": versions or {"component_a": "a-v1", "component_b": f"gemini|{PV}", "strategy": arm, "bandit": "ts-v1"},
        "arm_probs": {a: 1 / 3 for a in ARMS} if probs else None, "feedback": feedback,
    }


def logs_from(rows):
    return pd.DataFrame(rows)


# --- Posterior updates ---------------------------------------------------------------

def test_alpha_beta_update():
    rows = [rec(1, "cost", 1), rec(2, "cost", 1), rec(3, "cost", -1), rec(4, "checklist", -1)]
    post = bandit.posterior(logs_from(rows), prior=UNIFORM, prompt_version=PV)
    assert (post["cost"]["alpha"], post["cost"]["beta"]) == (3, 2)  # 1 + 2 successes, 1 + 1 failure
    assert (post["checklist"]["alpha"], post["checklist"]["beta"]) == (1, 2)
    assert (post["root_cause"]["alpha"], post["root_cause"]["beta"]) == (1, 1)
    assert post["cost"]["n_real"] == 3 and post["cost"]["n_prior"] == 0


def test_prior_is_added_and_kept_separate():
    prior = {"checklist": (7.0, 5.0), "cost": (6.0, 6.0), "root_cause": (6.5, 5.5)}
    post = bandit.posterior(logs_from([rec(1, "checklist", 1)]), prior=prior, prompt_version=PV)
    assert post["checklist"]["alpha"] == 8 and post["checklist"]["beta"] == 5
    assert post["checklist"]["n_prior"] == 10 and post["checklist"]["n_real"] == 1


def test_prior_file_is_capped(tmp=None):
    path = Path(tempfile.mkdtemp()) / "prior.json"
    path.write_text(json.dumps({"arms": {"cost": {"alpha": 51, "beta": 31}}}))  # 80 pseudo-observations
    prior = bandit.load_prior(path)
    a, b = prior["cost"]
    assert abs((a - 1) + (b - 1) - bandit.MAX_PRIOR_PSEUDO) < 1e-9  # capped at 10
    assert abs((a - 1) / ((a - 1) + (b - 1)) - 50 / 80) < 1e-9  # same pseudo-success rate
    assert prior["checklist"] == (1.0, 1.0)  # missing arm: uniform
    assert bandit.load_prior(Path(tempfile.mkdtemp()) / "missing.json") == UNIFORM


def test_production_prior_is_neutral():
    assert bandit.load_prior() == UNIFORM  # Beta(1, 1) for every arm, no pseudo-observations
    snap = bandit.policy_snapshot(None)
    assert (snap["n_prior"] == 0).all() and (snap["n_real"] == 0).all()


# --- No feedback, exploration, learning -------------------------------------------------

def test_no_feedback_uses_prior_only():
    for logs in (None, pd.DataFrame(), logs_from([rec(1, "cost", None)])):
        post = bandit.posterior(logs, prior=UNIFORM, prompt_version=PV)
        assert all(post[a]["n_real"] == 0 and post[a]["alpha"] == 1 and post[a]["beta"] == 1 for a in ARMS)
        arm, probs = bandit.choose_arm(logs, seed=1, prior=UNIFORM, prompt_version=PV)
        assert arm in ARMS and abs(sum(probs.values()) - 1) < 0.01
        assert all(0.25 < p < 0.42 for p in probs.values())  # no data, no preference


def test_exploration_every_arm_gets_chosen():
    chosen = [bandit.choose_arm(None, seed=s, prior=UNIFORM)[0] for s in range(600)]
    shares = {a: chosen.count(a) / len(chosen) for a in ARMS}
    assert all(0.25 < s < 0.42 for s in shares.values()), shares
    # With some evidence for one arm, the others are still explored sometimes.
    rows = [rec(i, "root_cause", 1) for i in range(6)] + [rec(10 + i, "cost", -1) for i in range(2)]
    logs = logs_from(rows)
    picks = [bandit.choose_arm(logs, seed=s, prior=UNIFORM, prompt_version=PV)[0] for s in range(600)]
    assert picks.count("root_cause") / 600 > 0.6
    assert picks.count("checklist") > 0 and picks.count("cost") > 0


def test_learns_from_logged_feedback():
    _, before = bandit.choose_arm(None, seed=3, prior=UNIFORM)
    rows = []
    for i in range(30):
        rows.append(rec(i, "checklist", 1 if i % 10 else -1))  # 90% positive
        rows.append(rec(100 + i, "cost", 1 if i % 3 == 0 else -1))  # 33% positive
        rows.append(rec(200 + i, "root_cause", 1 if i % 2 == 0 else -1))  # 50% positive
    _, after = bandit.choose_arm(logs_from(rows), seed=3, prior=UNIFORM, prompt_version=PV)
    assert after["checklist"] > 0.9 > before["checklist"]


def test_simulation_converges_to_best_arm():
    sys.path.insert(0, str(ROOT / "eval"))
    import simulate_bandit as sim

    rates = {"checklist": 0.45, "cost": 0.45, "root_cause": 0.75}
    last = sim.simulate(rates, sim.PRIORS["neutral"], runs=20, rounds=200).iloc[-1]
    assert last["p_root_cause"] > 0.9  # learns the best arm from a neutral start
    assert last["regret_ts"] < 0.4 * last["regret_random"]
    wrong = sim.simulate(rates, sim.PRIORS["wrong_belief"], runs=20, rounds=300).iloc[-1]
    assert wrong["p_root_cause"] > 0.9  # corrects a prior that strongly favoured checklist
    tie = sim.simulate({a: 0.5 for a in ARMS}, sim.PRIORS["neutral"], runs=20, rounds=200).iloc[-1]
    assert max(tie[f"p_{a}"] for a in ARMS) < 0.7  # no false convergence when the arms are equal


# --- Which records count ----------------------------------------------------------------

def test_invalid_records_are_excluded():
    stub_versions = {"component_a": "stub-a-v0", "component_b": "none|stub-agent-v0", "bandit": "stub-bandit-v0"}
    rows = [
        rec(1, "cost", 1),  # the only valid one
        rec(2, "cost", 1, source="simulated"),
        rec(3, "cost", 1, source="evaluation"),
        rec(4, "cost", 1, status="out_of_scope"),
        rec(5, "cost", 1, degraded=True),
        rec(6, "cost", 1, versions=stub_versions),
        rec(7, "cost", 1, versions={"component_a": "a-v1", "component_b": "gemini|agent-v0", "bandit": "ts-v1"}),
        rec(8, "cost", 1, versions={"component_a": "a-v1", "component_b": f"gemini|{PV}", "bandit": "stub-bandit-v0"}),
        rec(9, "cost", 1, probs=False),  # not chosen by the live bandit
        rec(13, "cost", 1, versions={"component_a": "a-v1", "component_b": f"gemini|{PV}", "bandit": "ts-v0"}),  # incompatible policy
        rec(10, "not_an_arm", 1),
        rec(11, "cost", None),  # no rating
        rec(12, "cost", 0),  # not a valid rating
    ]
    fb = bandit.eligible_feedback(logs_from(rows), prompt_version=PV)
    assert list(fb["id"]) == ["r1"], list(fb["id"])


def test_current_stub_prompt_version_means_no_eligible_feedback():
    rows = [rec(1, "cost", 1)]
    assert bandit.eligible_feedback(logs_from(rows)).empty  # current agent is a stub: nothing counts yet


def test_prompt_version_change_resets_policy():
    rows = [rec(i, "cost", 1) for i in range(5)]
    assert len(bandit.eligible_feedback(logs_from(rows), prompt_version=PV)) == 5
    assert bandit.eligible_feedback(logs_from(rows), prompt_version="agent-v2").empty


# --- Real logbook: duplicates, persistence --------------------------------------------------

def _log_real_request(arm):
    record = {"input_summary": "synthetic test", "features": {"x": 1.0}, "outputs": {"a": {}, "b": {}},
              "versions": {"component_a": "a-v1", "component_b": f"gemini|{PV}", "strategy": arm, "bandit": "ts-v1"},
              "review": False, "latency_s": 0.1, "tokens": 0, "cost_eur": 0.0,
              "status": "ok", "arm": arm, "degraded": False, "source": "real", "arm_probs": {a: 1 / 3 for a in ARMS}}
    return logbook.log_request(record)["id"]


def test_duplicate_or_changed_feedback_counts_once_latest_wins():
    rid = _log_real_request("root_cause")
    logbook.log_feedback(rid, +1)
    logbook.log_feedback(rid, +1)  # duplicate
    logbook.log_feedback(rid, -1)  # changed mind: the latest wins
    fb = bandit.eligible_feedback(logbook.load(), prompt_version=PV)
    mine = fb[fb["id"] == rid]
    assert len(mine) == 1 and int(mine["reward"].iloc[0]) == 0


def test_policy_survives_restart():
    for arm, value in [("checklist", 1), ("checklist", 1), ("cost", -1)]:
        logbook.log_feedback(_log_real_request(arm), value)
    here = bandit.policy_snapshot(logbook.load(), prior=UNIFORM, prompt_version=PV)[["arm", "alpha", "beta"]]
    code = ("import sys, json; sys.path.insert(0, %r); from maintai import bandit; import logbook; "
            "s = bandit.policy_snapshot(logbook.load(), prior={a: (1.0, 1.0) for a in bandit.ARMS}, prompt_version=%r); "
            "print(json.dumps(s[['arm','alpha','beta']].to_dict('records')))") % (str(ROOT), PV)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "LOG_DIR": LOG_DIR, "GEMINI_API_KEY": ""}).stdout
    assert json.loads(out.strip().splitlines()[-1]) == here.to_dict("records")  # a new process rebuilds the same policy


# --- Reproducibility and safety ------------------------------------------------------------

def test_offline_arm_is_reproducible():
    raw = '{"type": "L", "air_temperature_k": 300.1, "process_temperature_k": 310.2, "rotational_speed_rpm": 1450, "torque_nm": 62.0, "tool_wear_min": 215}'
    assert bandit.offline_arm(raw) == bandit.offline_arm(raw) in ARMS
    code = "import sys; sys.path.insert(0, %r); from maintai import bandit; print(bandit.offline_arm(%r))" % (str(ROOT), raw)
    other = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert other == bandit.offline_arm(raw)  # same arm in another process (no random seed involved)
    assert bandit.choose_arm(None, seed=7, prior=UNIFORM) == bandit.choose_arm(None, seed=7, prior=UNIFORM)


def test_arm_never_changes_safety_decisions():
    assert "arm" not in inspect.signature(guardrails.finalise).parameters  # the safety layer never sees the arm
    readings = [
        {"type": "L", "air_temperature_k": 300.1, "process_temperature_k": 310.2, "rotational_speed_rpm": 1450, "torque_nm": 62.0, "tool_wear_min": 215},
        {"type": "M", "air_temperature_k": 298.4, "process_temperature_k": 308.7, "rotational_speed_rpm": 1520, "torque_nm": 38.0, "tool_wear_min": 40},
        {"type": "L", "air_temperature_k": 300.1, "process_temperature_k": 310.2, "rotational_speed_rpm": 1450, "torque_nm": 42.0, "tool_wear_min": 228},
    ]
    for reading in readings:
        decisions = set()
        for arm in ARMS:
            r = pipeline.run(json.dumps(reading), arm=arm)
            order = r["outputs"]["component_b"]["work_order"]
            decisions.add((order["review"], tuple(order["review_reasons"]), r["outputs"]["component_a"]["risk_band"]))
        assert len(decisions) == 1, decisions  # same review flag, reasons and risk for every arm


def test_snapshot_and_history_shape():
    rows = [rec(i, ARMS[i % 3], 1 if i % 2 else -1) for i in range(12)]
    snap = bandit.policy_snapshot(logs_from(rows), prior=UNIFORM, prompt_version=PV)
    for c in ["arm", "alpha", "beta", "mean", "ci_low", "ci_high", "n_real", "n_prior", "p_select"]:
        assert c in snap.columns, c
    assert ((snap["ci_low"] <= snap["mean"]) & (snap["mean"] <= snap["ci_high"])).all()
    assert abs(snap["p_select"].sum() - 1) < 0.01 and snap["n_real"].sum() == 12
    events = pd.DataFrame([{"id": r["id"], "timestamp": r["timestamp"], "feedback": r["feedback"]} for r in rows])
    hist = bandit.policy_history(logs_from(rows), events, prior=UNIFORM, prompt_version=PV)
    assert hist["step"].min() == 0 and hist["step"].max() == 12
    assert (hist[hist["step"] == 0]["n_real"] == 0).all()  # step 0 is the prior alone


# --- Methodology checks (round 2) ------------------------------------------------------

def test_selection_probabilities_sum_to_one():
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(300):
        post = bandit.posterior_from_counts({a: int(rng.integers(0, 40)) for a in ARMS},
                                            {a: int(rng.integers(0, 40)) for a in ARMS}, UNIFORM)
        probs = bandit.selection_probabilities(post, rng, n=int(rng.integers(50, 3000)))
        assert abs(sum(probs.values()) - 1) < 1e-9 and all(0 <= p <= 1 for p in probs.values())


def test_exploration_is_an_explicit_operational_label():
    tie = bandit.posterior_from_counts({}, {}, UNIFORM)
    assert bandit.greedy_arm(tie) is None and bandit.is_exploration("cost", tie) is None  # undefined, not "exploration"
    post = bandit.posterior_from_counts({"cost": 5}, {"checklist": 3}, UNIFORM)
    assert bandit.greedy_arm(post) == "cost"
    assert bandit.is_exploration("cost", post) is False and bandit.is_exploration("root_cause", post) is True
    d = bandit.decide(logs_from([rec(1, "cost", 1)]), seed=4, prior=UNIFORM, prompt_version=PV)
    assert d.explore == bandit.is_exploration(d.arm, d.posterior) and d.n_feedback == 1


def test_history_uses_rating_time_and_never_future_information():
    rows = [rec(1, "cost", -1, minute=5), rec(2, "checklist", 1, minute=1)]  # r2 was served first...
    t = pd.Timestamp("2026-11-28 12:00", tz="UTC")
    events = pd.DataFrame([
        {"id": "r1", "timestamp": t, "feedback": 1},  # ...but r1 was rated first
        {"id": "r2", "timestamp": t + pd.Timedelta(minutes=1), "feedback": 1},
        {"id": "r1", "timestamp": t + pd.Timedelta(minutes=2), "feedback": -1},  # r1 changed to negative
    ])
    hist = bandit.policy_history(logs_from(rows), events, prior=UNIFORM, prompt_version=PV, points=10)
    at = {k: g.set_index("arm") for k, g in hist.groupby("step")}
    assert at[1].loc["cost", "successes"] == 1 and at[1].loc["checklist", "n_real"] == 0  # only what was known at step 1
    assert at[2].loc["checklist", "successes"] == 1
    assert at[3].loc["cost", "successes"] == 0 and at[3].loc["cost", "failures"] == 1  # replaced, not added
    assert int(at[3]["n_feedback"].iloc[0]) == 2 and int(at[3]["n_real"].sum()) == 2
    final = bandit.policy_snapshot(logs_from(rows), prior=UNIFORM, prompt_version=PV).set_index("arm")
    assert (final["alpha"] == at[3]["alpha"]).all() and (final["beta"] == at[3]["beta"]).all()  # history ends at the live policy


def test_log_read_failure_falls_back_to_prior_and_is_logged():
    import app

    original = logbook.load
    logbook.load = lambda: (_ for _ in ()).throw(RuntimeError("database unreachable"))
    try:
        record = app.serve(json.dumps({"type": "M", "air_temperature_k": 298.4, "process_temperature_k": 308.7,
                                       "rotational_speed_rpm": 1520, "torque_nm": 38.0, "tool_wear_min": 40}))
    finally:
        logbook.load = original
    assert record["bandit_policy_from"] == "prior_only" and record["arm"] in ARMS
    assert abs(sum(record["arm_probs"].values()) - 1) < 1e-9
    from ui import rl_dashboard

    assert "Prior-only decisions" in rl_dashboard.render(logbook.load())


def test_adversarial_agent_cannot_change_rule_based_safety():
    """A fake agent that writes a different order for every arm (worst case for the real LLM agent).

    Protected by design today: the risk assessment and the deterministic review rules
    (critical risk, out of training range, uncertain prediction, grounding), because
    component A and guardrails never receive the arm. NOT protected by the current
    guardrails: the action, priority and the agent's own review request, which come from
    the draft. That gap is a requirement for WP3 (docs/WP6_BANDIT.md); this test records it.
    """
    from maintai import agent
    from maintai.schemas import Action, AgentResult, FailureMode, Priority, WorkOrderDraft

    def fake_plan(machine, risk, arm):
        i = ARMS.index(arm)
        draft = WorkOrderDraft(priority=list(Priority)[i], action=[Action.MONITOR, Action.REPLACE_TOOL, Action.REDUCE_LOAD][i],
                               failure_mode_addressed=FailureMode.TWF, steps=[f"style {arm}"], justification=arm,
                               requires_human_review=(arm == "cost"))
        return AgentResult(draft=draft, prompt_version="fake-v1")

    original = agent.plan
    agent.plan = fake_plan
    try:
        reading = json.dumps({"type": "L", "air_temperature_k": 300.1, "process_temperature_k": 310.2,
                              "rotational_speed_rpm": 1450, "torque_nm": 62.0, "tool_wear_min": 215})
        records = {arm: pipeline.run(reading, arm=arm) for arm in ARMS}
    finally:
        agent.plan = original
    rule_reasons = {"critical_risk", "out_of_training_range", "uncertain_prediction", "failed_validation"}
    risk = {json.dumps(r["outputs"]["component_a"], sort_keys=True) for r in records.values()}
    rules = {tuple(x for x in r["outputs"]["component_b"]["work_order"]["review_reasons"] if x in rule_reasons) for r in records.values()}
    assert len(risk) == 1 and len(rules) == 1 and "critical_risk" in next(iter(rules))
    assert all(r["review"] for r in records.values())  # a critical case is reviewed whatever the arm
    actions = {r["outputs"]["component_b"]["work_order"]["draft"]["action"] for r in records.values()}
    assert len(actions) == 3  # documents the WP3 gap: today the agent's draft decides the action


def test_dashboard_without_real_feedback_shows_empty_states_only():
    from ui import rl_dashboard

    page = rl_dashboard.render(logs_from([rec(1, "cost", 1, versions={"component_a": "stub-a-v0", "component_b": "none|stub-agent-v0", "bandit": "ts-v1"})]))
    assert "Waiting for eligible feedback" in page and "No real learning curve yet" in page
    live, _, sim = page.partition('class="mai-card mai-sim"')
    assert "Simulated users" not in live and "Simulated users" in sim  # simulation kept in its own section
    assert "Stub component versions" in live


def test_dashboard_with_synthetic_eligible_feedback():
    from ui import rl_dashboard

    rows = [rec(i, ARMS[i % 3], 1 if i % 3 == 2 or i % 4 == 0 else -1) for i in range(30)]
    original = bandit.current_prompt_version
    bandit.current_prompt_version = lambda: PV
    try:
        page = rl_dashboard.render(logs_from(rows))
    finally:
        bandit.current_prompt_version = original
    assert "Learning from 30 real rating(s)" in page and "Waiting for eligible feedback" not in page
    assert "No real learning curve yet" in page  # synthetic rows here have no feedback events file: history stays empty
    assert "<polyline" in page  # Beta density curves are drawn


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

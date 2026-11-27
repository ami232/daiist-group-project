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


def test_eligibility_follows_the_current_prompt_version_whatever_it_is():
    """Version-independent: works with today's stub agent and with WP3's real agent."""
    current = bandit.current_prompt_version()
    mine = {"component_a": "a-v1", "component_b": f"gemini|{current}", "bandit": "ts-v1"}
    rows = [rec(1, "cost", 1, versions=mine), rec(2, "cost", 1)]  # r2 is written with PV = "agent-v1"
    fb = bandit.eligible_feedback(logs_from(rows))
    if current.startswith("stub"):
        assert fb.empty  # a stub agent never trains the policy
    else:
        assert "r1" in set(fb["id"])  # the real agent's own records count
        assert ("r2" in set(fb["id"])) == (current == PV)  # other prompt versions never do


def test_future_agent_version_is_supported():
    """Simulates WP3 publishing a real agent: only the new version's real ratings count."""
    from maintai import agent

    original = agent.PROMPT_VERSION
    agent.PROMPT_VERSION = "agent-v7"
    try:
        v7 = {"component_a": "a-v1", "component_b": "gemini-x|agent-v7", "bandit": "ts-v1"}
        rows = [rec(1, "cost", 1, versions=v7), rec(2, "root_cause", -1, versions=v7),
                rec(3, "cost", 1),  # previous prompt version (agent-v1): excluded
                rec(4, "cost", 1, versions={**v7, "component_a": "stub-a-v0"})]  # stub model: excluded
        assert bandit.current_prompt_version() == "agent-v7"
        assert sorted(bandit.eligible_feedback(logs_from(rows))["id"]) == ["r1", "r2"]
        assert bandit.decide(logs_from(rows), seed=1).n_feedback == 2
    finally:
        agent.PROMPT_VERSION = original


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



# --- Hardening (WP6/WP7): sessions, missing data, events, metrics ------------------------------

def _tagged(i, arm, fb, tag, minute=None):
    return {**rec(i, arm, fb, minute=minute or i), "session_tag": tag}


def test_session_cap_keeps_first_ratings_per_session():
    cap = bandit.MAX_RATINGS_PER_SESSION
    rows = [_tagged(i, "cost", 1, "spam") for i in range(cap + 10)]  # one session floods with positives
    rows += [_tagged(100 + i, "checklist", -1, f"user{i}") for i in range(3)]
    rows += [rec(200, "root_cause", 1)]  # untagged (served outside the UI): not capped
    fb = bandit.eligible_feedback(logs_from(rows), prompt_version=PV)
    assert (fb["arm"] == "cost").sum() == cap  # the flood is cut at the cap...
    kept = set(fb[fb["arm"] == "cost"]["id"])
    assert kept == {f"r{i}" for i in range(cap)}  # ...keeping the first ones by request time
    assert (fb["arm"] == "checklist").sum() == 3 and (fb["arm"] == "root_cause").sum() == 1
    s = __import__("maintai.rl_monitor", fromlist=["x"]).sessions(logs_from(rows), PV)
    assert s["n_over_cap"] == 10 and s["n_sessions"] == 4 and s["n_untagged"] == 1 and s["max_per_session"] == cap + 10


def test_history_applies_the_same_session_cap_and_ends_at_live_policy():
    cap = bandit.MAX_RATINGS_PER_SESSION
    rows = [_tagged(i, "cost", 1, "spam") for i in range(cap + 5)] + [_tagged(50, "checklist", 1, "other")]
    t = pd.Timestamp("2026-11-28 12:00", tz="UTC")
    events = pd.DataFrame([{"id": r["id"], "timestamp": t + pd.Timedelta(seconds=k), "feedback": r["feedback"]}
                           for k, r in enumerate(rows)])
    hist = bandit.policy_history(logs_from(rows), events, prior=UNIFORM, prompt_version=PV, points=100)
    last = hist[hist["step"] == hist["step"].max()].set_index("arm")
    live = bandit.policy_snapshot(logs_from(rows), prior=UNIFORM, prompt_version=PV).set_index("arm")
    assert (last["alpha"] == live["alpha"]).all() and (last["beta"] == live["beta"]).all()
    assert last.loc["cost", "successes"] == cap and int(last["n_feedback"].iloc[0]) == cap + 1


def test_feedback_guard_limits_and_privacy():
    from maintai import feedback_guard as fg

    guard = fg.FeedbackGuard(per_request=2, per_session=3)
    assert guard.allow("s1", "a") == (True, "ok") and guard.allow("s1", "a") == (True, "ok")
    assert guard.allow("s1", "a") == (False, "request_limit")  # third click on the same order
    assert guard.allow("s1", "b") == (True, "ok")
    assert guard.allow("s1", "c") == (False, "session_limit")  # session total reached
    assert guard.allow("s2", "c") == (True, "ok")  # another session is independent
    assert guard.allow(None, "x") == (True, "no session")  # check.py / tests: no session, not blocked
    tag = fg.session_tag("abc123")
    assert tag == fg.session_tag("abc123") and len(tag) == 16 and "abc123" not in tag  # stable within the process
    assert fg.session_tag("") is None and fg.session_tag(None) is None

    class FakeRequest:  # has client/headers like gr.Request; only session_hash may be read
        session_hash = "abc123"
        client = type("C", (), {"host": "203.0.113.7"})()
        headers = {"user-agent": "secret-browser"}

    assert fg.tag_from_request(FakeRequest()) == tag and fg.tag_from_request(None) is None


def test_feedback_guard_expires_and_bounds_memory():
    from maintai import feedback_guard as fg

    now = [0.0]
    guard = fg.FeedbackGuard(per_request=1, per_session=1, ttl_s=10, max_sessions=50, clock=lambda: now[0])
    assert guard.allow("s", "a")[0] and not guard.allow("s", "b")[0]
    now[0] = 11.0  # session expired: counters reset
    assert guard.allow("s", "b")[0]
    for i in range(500):
        guard.allow(f"x{i}", "a")
    assert guard.tracked_sessions() <= 50


def test_ui_feedback_is_limited_per_session():
    import app
    from maintai import feedback_guard as fg

    class Req:
        session_hash = "ui-session-1"

    reading = json.dumps({"type": "M", "air_temperature_k": 298.4, "process_temperature_k": 308.7,
                          "rotational_speed_rpm": 1520, "torque_nm": 38.0, "tool_wear_min": 40})
    record = app.serve(reading, session_tag=fg.tag_from_request(Req()))
    assert record["session_tag"] == fg.tag_from_request(Req())
    before = len(logbook._read("feedback"))
    results = [app.on_feedback(record["id"], record["arm"], +1, Req()) for _ in range(fg.MAX_EVENTS_PER_REQUEST + 2)]
    written = len(logbook._read("feedback")) - before
    assert written == fg.MAX_EVENTS_PER_REQUEST  # extra clicks on the same order are refused, not logged
    assert all(isinstance(r, str) for r in results[:fg.MAX_EVENTS_PER_REQUEST])


def test_serve_and_run_stay_compatible():
    import app

    assert list(inspect.signature(app.serve).parameters)[0] == "raw_input"
    assert inspect.signature(app.serve).parameters["session_tag"].default is None  # check.py: serve(raw_input)
    record = app.serve(json.dumps({"type": "L", "air_temperature_k": 300.1, "process_temperature_k": 310.2,
                                   "rotational_speed_rpm": 1450, "torque_nm": 42.0, "tool_wear_min": 100}))
    assert logbook.validate(record) == [] and record["session_tag"] is None
    api = app.demo.get_api_info()["named_endpoints"]
    assert list(api) == ["/run"], list(api)  # the only public endpoint
    assert [p["parameter_name"] for p in api["/run"]["parameters"]] == ["raw_input"]  # still one text argument
    fn = next(f for f in app.demo.fns.values() if f.api_name == "run")
    assert fn.concurrency_limit == app.RUN_CONCURRENCY >= 1 and app.demo._queue.max_size == app.QUEUE_MAX_SIZE


def test_feedback_events_reader_is_guarded():
    original = logbook._read
    try:
        logbook._read = lambda table: (_ for _ in ()).throw(RuntimeError("db down"))
        ev = bandit.load_feedback_events()
        assert ev.empty and "unreadable" in ev.attrs["error"]
        logbook._read = lambda table: [{"id": "a", "timestamp": "2026-11-28T10:00:00+00:00", "feedback": 1},
                                       {"id": "b", "timestamp": "not a date", "feedback": 1},
                                       {"id": "c", "feedback": -1}, "garbage",
                                       {"id": "d", "timestamp": "2026-11-28T09:00:00+00:00", "feedback": "x"}]
        ev = bandit.load_feedback_events()
        assert list(ev["id"]) == ["a"] and "malformed" in ev.attrs["error"]
        del logbook._read  # the private helper disappears in a future logbook.py
        ev = bandit.load_feedback_events()
        assert ev.empty and "not available" in ev.attrs["error"]
    finally:
        logbook._read = original
    from ui import rl_dashboard

    logs = logbook.load()
    logbook._read = lambda table: (_ for _ in ()).throw(RuntimeError("db down")) if table == "feedback" else original(table)
    try:
        page = rl_dashboard.render(logs)  # the dashboard still renders when only the rating events fail
    finally:
        logbook._read = original
    assert "Adaptive intelligence" in page


def test_missing_data_is_not_a_false_zero():
    from maintai import rl_monitor as m
    from ui import dashboard, rl_dashboard

    for logs in (None, pd.DataFrame(), logs_from([rec(1, "cost", None)])):
        assert m.response_rate(logs, PV)["rate"] in (None, 0.0)
        d = m.reward_delay(logs, None, PV)
        assert d["median_s"] is None and d["p95_s"] is None and d["n"] == 0
        comp = m.belief_comparison(logs, PV)
        assert comp["observed"].isna().all() and comp["ips"].isna().all()
    assert m.response_rate(pd.DataFrame(), PV)["rate"] is None
    page = rl_dashboard.render(pd.DataFrame())
    assert "N/A" in page and "Pending" in page
    # Monitoring KPIs: if monitor.summary() changes shape or fails, show N/A, never 0 / €0.0000.
    logs = logs_from([{**rec(1, "cost", None), "review": False, "latency_s": 0.2, "cost_eur": 0.0, "features": {}}])
    original = dashboard.monitor.summary
    try:
        dashboard.monitor.summary = lambda df: pd.DataFrame({"metric": ["requests"], "value": ["1"]})  # renamed metrics
        page = dashboard.render(logs, None, "local files")
        assert "€0.0000" not in page and page.count(">N/A<") >= 4
        dashboard.monitor.summary = lambda df: (_ for _ in ()).throw(KeyError("feedback"))
        assert ">N/A<" in dashboard.render(logs, None, "local files")
    finally:
        dashboard.monitor.summary = original
    assert dashboard._number({"x": "-"}, "x") is None and dashboard._number({"x": "40%"}, "x") == 0.4
    assert dashboard._number({"x": "nan"}, "x") is None and dashboard._number({"x": "0.0"}, "x") == 0.0  # a real zero stays


def test_reward_delay_uses_first_rating_and_counts_changes():
    from maintai import rl_monitor as m

    t = pd.Timestamp("2026-11-28 10:00", tz="UTC")
    rows = [rec(1, "cost", -1, minute=1), rec(2, "checklist", 1, minute=2), rec(3, "root_cause", 1, minute=3),
            rec(4, "cost", 1, source="simulated", minute=4)]
    events = pd.DataFrame([
        {"id": "r1", "timestamp": t + pd.Timedelta(minutes=1, seconds=30), "feedback": 1},
        {"id": "r1", "timestamp": t + pd.Timedelta(minutes=5), "feedback": -1},  # changed later: delay uses the first
        {"id": "r2", "timestamp": t + pd.Timedelta(minutes=2, seconds=2), "feedback": 1},  # very fast (2 s)
        {"id": "r3", "timestamp": t + pd.Timedelta(minutes=2), "feedback": 1},  # before its request: clock problem
        {"id": "r4", "timestamp": t + pd.Timedelta(minutes=9), "feedback": 1},  # simulated: not eligible
    ])
    d = m.reward_delay(logs_from(rows), events, PV)
    assert d["n"] == 2 and d["n_changed"] == 1 and d["n_fast"] == 1 and d["n_negative_dropped"] == 1
    assert d["median_s"] == 16.0 and abs(d["p95_s"] - 28.6) < 1e-9


def test_response_rate_coverage_and_beliefs():
    from maintai import rl_monitor as m

    rows = [rec(i, ARMS[i % 3], (1 if i % 2 else -1) if i % 4 else None) for i in range(1, 25)]
    rows += [rec(90, "cost", 1, source="simulated")]  # never in the denominators
    logs = logs_from(rows)
    rr = m.response_rate(logs, PV)
    assert rr["n_eligible"] == 24 and rr["n_rated"] == 18 and abs(rr["rate"] - 0.75) < 1e-12
    cov = m.coverage_by_arm(logs, PV).set_index("arm")
    assert cov["eligible"].sum() == 24 and cov["counted"].sum() == 18 and abs(cov["share_of_counted"].sum() - 1) < 1e-12
    comp = m.belief_comparison(logs, PV).set_index("arm")
    fb = bandit.eligible_feedback(logs, PV)
    for arm in ARMS:
        mine = fb[fb["arm"] == arm]
        assert abs(comp.loc[arm, "observed"] - mine["reward"].mean()) < 1e-12
        assert abs(comp.loc[arm, "posterior_mean"] - (1 + mine["reward"].sum()) / (2 + len(mine))) < 1e-12
        assert abs(comp.loc[arm, "ips"] - mine["reward"].mean()) < 1e-12  # uniform logged probs: IPS = observed
    assert abs(comp["p_select"].sum() - 1) < 1e-9


def test_ips_corrects_for_uneven_serving_probabilities():
    from maintai import rl_monitor as m

    rows = []
    for i in range(20):  # early on cost was served with p = 0.1 (rated well), later with p = 0.9 (rated badly)
        early = i < 10
        r = rec(i, "cost", 1 if early else -1)
        r["arm_probs"] = {"checklist": 0.45 if early else 0.05, "cost": 0.1 if early else 0.9,
                          "root_cause": 0.45 if early else 0.05}
        rows.append(r)
    comp = m.belief_comparison(logs_from(rows), PV).set_index("arm")
    assert comp.loc["cost", "observed"] == 0.5
    assert abs(comp.loc["cost", "ips"] - 0.9) < 1e-9  # (10/0.1) / (10/0.1 + 10/0.9)


def test_served_probabilities_are_the_logged_ones_in_time_order():
    from maintai import rl_monitor as m

    rows = []
    for i in range(1, 7):
        r = rec(i, "cost", None)
        r["arm_probs"] = {"checklist": 0.1 * i, "cost": 1 - 0.1 * i, "root_cause": 0.0}
        rows.append(r)
    rows.reverse()  # given out of order
    served = m.served_probabilities(logs_from(rows))
    assert list(served["p_checklist"].round(3)) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    blocks = m.served_probabilities(logs_from(rows), max_points=3)
    assert list(blocks["n"]) == [2, 2, 2] and abs(blocks["p_checklist"].iloc[0] - 0.15) < 1e-12


def test_non_response_flags_large_group_gaps_only_with_enough_data():
    from maintai import rl_monitor as m

    rows = []
    for i in range(40):  # critical orders always rated, low-risk orders rarely
        critical = i % 2 == 0
        r = rec(i, ARMS[i % 3], 1 if critical or i % 10 == 1 else None)
        r["risk_band"] = "critical" if critical else "low"
        r["review"] = critical
        rows.append(r)
    nr = m.non_response(logs_from(rows), PV)
    dims = {f["dimension"] for f in nr["flags"]}
    assert "risk_band" in dims and "review" in dims
    small = m.non_response(logs_from(rows[:8]), PV)
    assert small["flags"] == [] and small["dimensions_with_enough_data"] == 0  # too few: no flag either way


def test_evidence_export_is_aggregated_and_labelled():
    sys.path.insert(0, str(ROOT / "eval"))
    import export_rl_evidence as exporter

    out = Path(tempfile.mkdtemp(prefix="rl-evidence-"))
    from maintai import feedback_guard as fg

    rid = _log_real_request("checklist")
    logbook.log_feedback(rid, +1)
    manifest = exporter.export(out)
    files = {p.name for p in out.iterdir()}
    for name in ["manifest.json", "metrics.json", "policy_snapshot.csv", "policy_history.csv", "served_probabilities.csv",
                 "coverage_by_arm.csv", "belief_comparison.csv", "non_response.csv", "exclusions.csv", "simulation_summary.json"]:
        assert name in files, name
    assert set(manifest["provenance"]) == {"measured", "simulated", "assumption"}
    text = "".join((out / f).read_text(encoding="utf-8") for f in files if f != "simulation_summary.json")
    ids = set(logbook.load()["id"])
    assert not any(i in text for i in ids)  # no request id leaves the logs
    assert "session_tag" not in text  # no tag column or key is exported
    sim = json.loads((out / "simulation_summary.json").read_text())
    assert sim["data_type"] == "simulated"
    for name in files - {"simulation_summary.json", "manifest.json", "metrics.json"}:
        assert set(pd.read_csv(out / name)["data_type"]) <= {"measured"}, name
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["data_type"] == "measured"
    if not manifest["n_eligible_ratings"]:
        assert "not evidence" in manifest["warning"]  # with the stub agent nothing is eligible: said explicitly


def test_simulation_and_calibration_never_reach_production():
    import ast

    for path in ["maintai/bandit.py", "maintai/rl_monitor.py", "app.py", "pipeline.py"]:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        names = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not ({"simulate_bandit", "bandit_calibration", "export_rl_evidence"} & (names | mods)), path
        assert "bandit_simulation" not in (ROOT / path).read_text() and "bandit_calibration" not in (ROOT / path).read_text(), path
    assert bandit.load_prior() == UNIFORM  # production prior untouched by any simulated prior
    from ui import rl_dashboard

    page = rl_dashboard.render(pd.DataFrame())
    live, _, sim = page.partition('class="mai-card mai-sim"')
    assert "Coverage of the 95% credible intervals" in sim and "Coverage of the 95% credible intervals" not in live
    sim_ids = {r["id"] for r in logbook._read("requests") if r.get("source") == "simulated"}
    assert not sim_ids  # no test or tool wrote simulated requests into the logs


def test_calibration_is_reproducible_and_sane():
    sys.path.insert(0, str(ROOT / "eval"))
    import bandit_calibration as cal

    a = cal.run_scenario("prior_draw", None, runs=30, rounds=60, checkpoints=(20, 60))
    b = cal.run_scenario("prior_draw", None, runs=30, rounds=60, checkpoints=(20, 60))
    pd.testing.assert_frame_equal(a, b)  # same seed, same intervals
    assert ((a["ci_low"] <= a["ci_high"]) & a["ci_low"].between(0, 1) & a["ci_high"].between(0, 1)).all()
    table = cal.summarise(a)
    overall = table[table["group"] == "all styles"]
    assert overall["coverage"].between(0.85, 1.0).all()  # rates drawn from the prior: close to the nominal 95%
    saved = json.loads((ROOT / "eval" / "bandit_calibration_summary.json").read_text())
    assert saved["source"] == "simulated" and saved["nominal"] == 0.95


def test_performance_and_memory_budget():
    import time
    import tracemalloc

    from maintai import rl_monitor as m
    from ui import rl_dashboard

    rows = [{**rec(i, ARMS[i % 3], (1 if i % 3 else -1) if i % 2 else None), "session_tag": f"s{i % 200}",
             "risk_band": ["low", "medium", "high", "critical"][i % 4], "review": i % 4 == 3} for i in range(5000)]
    logs = logs_from(rows)
    t = pd.Timestamp("2026-11-28 10:00", tz="UTC")
    events = pd.DataFrame([{"id": r["id"], "timestamp": t + pd.Timedelta(minutes=i, seconds=40), "feedback": r["feedback"]}
                           for i, r in enumerate(rows) if r["feedback"] is not None])
    tracemalloc.start()
    start = time.perf_counter()
    bandit.decide(logs, seed=1, prompt_version=PV)
    decide_s = time.perf_counter() - start
    start = time.perf_counter()
    m.evidence(logs, events, PV)
    metrics_s = time.perf_counter() - start
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    original = bandit.current_prompt_version
    bandit.current_prompt_version = lambda: PV
    try:
        start = time.perf_counter()
        rl_dashboard.render(logs)
        render_s = time.perf_counter() - start
    finally:
        bandit.current_prompt_version = original
    print(f"      5,000 requests: decide {decide_s * 1000:.0f} ms, metrics {metrics_s * 1000:.0f} ms, "
          f"dashboard {render_s * 1000:.0f} ms, metrics peak {peak_mb:.0f} MB (Python allocations)")
    assert decide_s < 1.0 and metrics_s < 5.0 and render_s < 10.0 and peak_mb < 150  # local budget; Render is slower

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

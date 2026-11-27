"""Thompson-sampling bandit over the three work-order writing strategies.  OWNER: WP6.

What it learns: which writing style (arm) maintenance users rate as useful.

    checklist   short numbered steps, minimum text          (technician on the floor)
    cost        opens with preventive vs breakdown cost     (manager with a budget)
    root_cause  explains the indicators behind the failure  (reliability engineer)

Scope: the arm only selects the writing strategy given to the agent. The safety
layer (guardrails.finalise) never receives it. Which decision fields are protected
from the arm, and what WP3 must still guarantee for the real agent, is described in
docs/WP6_BANDIT.md and checked in tests/test_bandit.py.

The cycle, rebuilt from the logs on every request (decision D6, prior revised by
Manuela on 2026-11-26):

1. Posterior per arm: Beta(alpha, beta) = prior + real feedback.
   - Production prior: neutral Beta(1, 1) for every arm (no pseudo-observations).
     The capped simulated prior of the master plan (at most MAX_PRIOR_PSEUDO
     pseudo-observations per arm) was studied in eval/simulate_bandit.py and is
     NOT used live; load_prior(path) can still read one for experiments.
   - Real feedback: eligible_feedback(logs). Reward 1 for a thumbs-up, 0 for a
     thumbs-down. A request without a rating does not count. At most
     MAX_RATINGS_PER_SESSION rated requests per random session tag count.
2. Selection (Thompson sampling): one sample from each posterior; the highest wins.
3. Logging (app.serve): arm, arm_probs (Monte Carlo probability that each arm is
   chosen, summing to 1), bandit_explore (see is_exploration), bandit_policy_from
   ("logs" or "prior_only" when the logs could not be read), versions["bandit"].
4. Persistence: no state file. logbook.load() reads Postgres on Render.

Offline (evaluate.py, main.py, check.py) the arm is a hash of the input
(offline_arm, decision D1), so results and the LLM cache are reproducible.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from maintai.schemas import ARMS

POLICY_VERSION = "ts-v1"
COMPATIBLE_POLICIES = {"ts-v1"}  # feedback logged under these policy versions is reused
DEFAULT_ARM = "checklist"
NEUTRAL_PRIOR = {arm: (1.0, 1.0) for arm in ARMS}
MAX_PRIOR_PSEUDO = 10  # cap for any simulated prior (decision D6)
MC_SAMPLES = 4000  # draws used to estimate the probability that each arm is chosen
CI_LEVEL = 0.95
SNAPSHOT_SEED = 0  # fixed, so the dashboard shows the same probabilities on every refresh
# Abuse limit (decided by Manuela, 2026-11-26): at most this many rated requests per
# browser session train the policy, the first ones by request time. A session is a
# random, temporary tag (maintai/feedback_guard.py), NOT an identity: a new page load
# is a new session, so this bounds casual abuse, it does not prove who rated.
MAX_RATINGS_PER_SESSION = 15

FEEDBACK_COLUMNS = ["id", "timestamp", "arm", "reward"]
EVENT_COLUMNS = ["id", "timestamp", "feedback"]


# --- Prior ---------------------------------------------------------------------

def load_prior(path: Path | str | None = None) -> dict[str, tuple[float, float]]:
    """(alpha0, beta0) per arm.

    Without a path: the production prior, neutral Beta(1, 1). With a path: a
    simulated prior file (experiments only), capped at MAX_PRIOR_PSEUDO
    pseudo-observations per arm; arms missing from the file stay Beta(1, 1).
    """
    prior = dict(NEUTRAL_PRIOR)
    if path is None or not Path(path).is_file():
        return prior
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for arm, values in data.get("arms", {}).items():
        if arm not in prior:
            continue
        a, b = max(float(values["alpha"]), 1.0), max(float(values["beta"]), 1.0)
        pseudo = (a - 1) + (b - 1)
        if pseudo > MAX_PRIOR_PSEUDO:  # shrink towards Beta(1, 1), keeping the pseudo-success rate
            scale = MAX_PRIOR_PSEUDO / pseudo
            a, b = 1 + (a - 1) * scale, 1 + (b - 1) * scale
        prior[arm] = (a, b)
    return prior


# --- Real feedback ---------------------------------------------------------------

def current_prompt_version() -> str:
    """The prompt version the policy belongs to. A new prompt version restarts the policy (D6)."""
    from maintai import agent  # local import: agent is heavy once WP3 adds the Gemini client

    return agent.PROMPT_VERSION


def _prompt_version(versions) -> str:
    if not isinstance(versions, dict):
        return ""
    return str(versions.get("component_b", "")).split("|")[-1]


def _has_stub(versions) -> bool:
    """True if any component version starts with "stub" (component_b is "model|prompt")."""
    return isinstance(versions, dict) and any(part.strip().startswith("stub")
                                              for v in versions.values() for part in str(v).split("|"))


def _col(df: pd.DataFrame, name: str, default) -> pd.Series:
    return df[name] if name in df else pd.Series([default] * len(df), index=df.index, dtype=object)


def eligible_requests(logs: pd.DataFrame | None, prompt_version: str | None = None) -> pd.DataFrame:
    """Requests whose rating may train the policy (rating not checked here).

    - served live by the bandit: source == "real" and it has `arm_probs`
      (simulations, evaluation runs and check.py are excluded);
    - handled (not out of scope) and not degraded (template order, no agent);
    - no component version starts with "stub" (smoke tests and development);
    - same prompt version as now, and a compatible bandit policy version;
    - its arm is one of schemas.ARMS.
    """
    if logs is None or logs.empty:
        return pd.DataFrame(columns=["id", "timestamp", "arm"])
    prompt_version = prompt_version or current_prompt_version()
    df = logs
    versions = _col(df, "versions", None)
    keep = (
        (_col(df, "source", "") == "real")
        & _col(df, "arm_probs", None).apply(lambda v: isinstance(v, dict) and len(v) > 0)
        & (_col(df, "status", "ok").fillna("ok") == "ok")
        & ~_col(df, "degraded", False).fillna(False).astype(bool)
        & ~versions.apply(_has_stub)
        & (versions.apply(_prompt_version) == prompt_version)
        & versions.apply(lambda v: isinstance(v, dict) and v.get("bandit") in COMPATIBLE_POLICIES)
        & _col(df, "arm", "").isin(ARMS)
    )
    return df[keep]


def apply_session_cap(df: pd.DataFrame, cap: int = MAX_RATINGS_PER_SESSION) -> pd.DataFrame:
    """Keep at most `cap` rated requests per session tag: the first ones by request time.

    `df` has one row per rated request with `timestamp` (request time) and, when
    logged, `session_tag`. Rows without a tag (records logged before the tag existed,
    or requests served outside the UI, e.g. check.py) are not capped.
    """
    if df.empty or "session_tag" not in df:
        return df
    tag = df["session_tag"]
    tagged = tag.notna() & (tag.astype(str).str.len() > 0)
    ordered = df[tagged].sort_values("timestamp", kind="stable")
    rank = ordered.groupby("session_tag", sort=False).cumcount()
    keep = pd.Series(True, index=df.index)
    keep.loc[rank.index] = rank < cap
    return df[keep]


def _rated(logs: pd.DataFrame | None, prompt_version: str | None = None) -> pd.DataFrame:
    """Eligible requests with a valid rating (latest per request), before the session cap."""
    if logs is None or logs.empty or "feedback" not in logs:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    df = eligible_requests(logs, prompt_version)
    feedback = pd.to_numeric(df["feedback"], errors="coerce") if not df.empty else pd.Series(dtype=float)
    df = df[feedback.isin([1, -1])]
    if df.empty:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    df = df.assign(reward=(pd.to_numeric(df["feedback"]) > 0).astype(int))
    return df.sort_values("timestamp").drop_duplicates("id", keep="last")  # logbook.load() already keeps one row per id


def eligible_feedback(logs: pd.DataFrame | None, prompt_version: str | None = None) -> pd.DataFrame:
    """The ratings the policy learns from: one row per request (the latest rating), reward 1 or 0,
    at most MAX_RATINGS_PER_SESSION per session tag."""
    df = apply_session_cap(_rated(logs, prompt_version))
    if df.empty:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    return df[FEEDBACK_COLUMNS].reset_index(drop=True)


def load_feedback_events() -> pd.DataFrame:
    """Every rating event with its own timestamp (id, timestamp, feedback), oldest first.

    logbook.load() keeps only the latest rating per request and drops when it was
    given; policy_history() and the reward-delay metric need the events themselves.

    INTERNAL DEPENDENCY: this reads `logbook._read("feedback")`, a private helper of
    the fixed template module (files or Postgres, whichever logbook uses). It is
    read-only and guarded: if the helper disappears, raises (database unreachable)
    or returns malformed rows, the result is an empty or partial frame and
    `result.attrs["error"]` says why. Callers then show "not available" instead of
    failing. Nothing on the request path (/run) calls this function.
    """
    empty = pd.DataFrame(columns=EVENT_COLUMNS)
    try:
        import logbook

        reader = getattr(logbook, "_read", None)
        if not callable(reader):
            empty.attrs["error"] = "logbook._read is not available in this version of logbook.py"
            return empty
        rows = reader("feedback")
    except Exception as error:  # noqa: BLE001 - the dashboard must not fail because of the logs
        empty.attrs["error"] = f"feedback events unreadable ({type(error).__name__})"
        return empty
    rows = [r for r in (rows or []) if isinstance(r, dict) and all(k in r for k in EVENT_COLUMNS)]
    if not rows:
        return empty
    events = pd.DataFrame(rows)[EVENT_COLUMNS]
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, format="ISO8601", errors="coerce")
    events["feedback"] = pd.to_numeric(events["feedback"], errors="coerce")
    bad = events["timestamp"].isna() | events["feedback"].isna()
    events = events[~bad].sort_values("timestamp", kind="stable").reset_index(drop=True)
    if bad.any():
        events.attrs["error"] = f"{int(bad.sum())} malformed feedback event(s) skipped"
    return events


# --- Posterior and choice ----------------------------------------------------------

def posterior_from_counts(successes: dict, failures: dict, prior: dict | None = None) -> dict[str, dict]:
    """Beta posterior per arm from reward counts. Used by posterior() and by the simulator."""
    prior = prior or NEUTRAL_PRIOR
    out = {}
    for arm in ARMS:
        a0, b0 = prior[arm]
        s, f = int(successes.get(arm, 0)), int(failures.get(arm, 0))
        out[arm] = {"alpha": a0 + s, "beta": b0 + f, "prior_alpha": a0, "prior_beta": b0,
                    "successes": s, "failures": f, "n_real": s + f, "n_prior": round((a0 - 1) + (b0 - 1), 3)}
    return out


def _counts(fb: pd.DataFrame) -> tuple[dict, dict]:
    return (fb[fb["reward"] == 1]["arm"].value_counts().to_dict(),
            fb[fb["reward"] == 0]["arm"].value_counts().to_dict())


def posterior(logs: pd.DataFrame | None = None, prior: dict | None = None,
              prompt_version: str | None = None) -> dict[str, dict]:
    """Beta(alpha, beta) per arm = prior + real successes and failures from the logs."""
    return posterior_from_counts(*_counts(eligible_feedback(logs, prompt_version)),
                                 prior if prior is not None else load_prior())


def selection_probabilities(post: dict[str, dict], rng: np.random.Generator, n: int = MC_SAMPLES) -> dict[str, float]:
    """Monte Carlo estimate of P(arm has the highest sample). Rounded to 4 decimals, summing exactly to 1."""
    draws = np.column_stack([rng.beta(post[a]["alpha"], post[a]["beta"], size=n) for a in ARMS])
    counts = np.bincount(draws.argmax(axis=1), minlength=len(ARMS))
    probs = [round(c / n, 4) for c in counts]
    probs[int(np.argmax(counts))] = round(1 - sum(p for i, p in enumerate(probs) if i != int(np.argmax(counts))), 4)
    return dict(zip(ARMS, probs))


def thompson_choice(post: dict[str, dict], rng: np.random.Generator) -> str:
    """One Thompson-sampling step: sample once from each posterior, pick the highest."""
    samples = [rng.beta(post[a]["alpha"], post[a]["beta"]) for a in ARMS]
    return ARMS[int(np.argmax(samples))]


def greedy_arm(post: dict[str, dict]) -> str | None:
    """The arm with the highest posterior mean, or None if several share it (e.g. the neutral prior)."""
    means = {a: post[a]["alpha"] / (post[a]["alpha"] + post[a]["beta"]) for a in ARMS}
    top = max(means.values())
    leaders = [a for a, m in means.items() if abs(m - top) < 1e-12]
    return leaders[0] if len(leaders) == 1 else None


def is_exploration(arm: str, post: dict[str, dict]) -> bool | None:
    """Operational definition used for monitoring, not a property of Thompson sampling.

    Thompson sampling has no separate "explore" step: every choice is a posterior
    sample. For the dashboard we call a choice *exploration* when the chosen arm is
    not the greedy arm (highest posterior mean) at that moment, and *exploitation*
    when it is. When no single arm has the highest mean (for example with the
    neutral prior and no feedback), the label is undefined: None.
    """
    greedy = greedy_arm(post)
    return None if greedy is None else arm != greedy


@dataclass
class Decision:
    arm: str
    arm_probs: dict[str, float]
    explore: bool | None
    greedy_arm: str | None
    n_feedback: int
    posterior: dict = field(repr=False, default_factory=dict)


def decide(logs: pd.DataFrame | None = None, seed: int | None = None, prior: dict | None = None,
           prompt_version: str | None = None) -> Decision:
    """One live decision with everything app.serve() logs about it."""
    rng = np.random.default_rng(seed)
    post = posterior(logs, prior, prompt_version)
    arm = thompson_choice(post, rng)
    return Decision(arm=arm, arm_probs=selection_probabilities(post, rng), explore=is_exploration(arm, post),
                    greedy_arm=greedy_arm(post), n_feedback=sum(post[a]["n_real"] for a in ARMS), posterior=post)


def choose_arm(logs: pd.DataFrame | None = None, seed: int | None = None, prior: dict | None = None,
               prompt_version: str | None = None) -> tuple[str, dict[str, float]]:
    """(arm, probability of each arm being chosen). `seed` makes it reproducible; live it is None."""
    d = decide(logs, seed, prior, prompt_version)
    return d.arm, d.arm_probs


def offline_arm(raw_input: str) -> str:
    """Deterministic arm for evaluation, main.py and check.py (decision D1), so the cache works."""
    digest = hashlib.sha256(str(raw_input).encode("utf-8")).digest()
    return ARMS[digest[0] % len(ARMS)]


# --- Views for the dashboard ------------------------------------------------------

def _interval(a: float, b: float) -> tuple[float, float]:
    from scipy.stats import beta as beta_dist  # scipy comes with scikit-learn

    tail = (1 - CI_LEVEL) / 2
    return float(beta_dist.ppf(tail, a, b)), float(beta_dist.ppf(1 - tail, a, b))


def snapshot_from_posterior(post: dict[str, dict], seed: int = SNAPSHOT_SEED) -> pd.DataFrame:
    probs = selection_probabilities(post, np.random.default_rng(seed))
    rows = []
    for arm in ARMS:
        p = post[arm]
        low, high = _interval(p["alpha"], p["beta"])
        rows.append({"arm": arm, "alpha": p["alpha"], "beta": p["beta"], "mean": p["alpha"] / (p["alpha"] + p["beta"]),
                     "ci_low": low, "ci_high": high, "n_real": p["n_real"], "n_prior": p["n_prior"],
                     "successes": p["successes"], "failures": p["failures"],
                     "prior_alpha": p["prior_alpha"], "prior_beta": p["prior_beta"], "p_select": probs[arm]})
    return pd.DataFrame(rows)


def policy_snapshot(logs: pd.DataFrame | None = None, prior: dict | None = None,
                    prompt_version: str | None = None) -> pd.DataFrame:
    """State of the policy, one row per arm.

    Columns: arm, alpha, beta, mean (estimated reward), ci_low, ci_high (95% credible
    interval), n_real (real ratings), n_prior (pseudo-observations of the prior; 0 with
    the neutral prior), successes, failures, prior_alpha, prior_beta, p_select.
    """
    return snapshot_from_posterior(posterior(logs, prior, prompt_version))


def policy_history(logs: pd.DataFrame | None = None, events: pd.DataFrame | None = None, prior: dict | None = None,
                   prompt_version: str | None = None, points: int = 40) -> pd.DataFrame:
    """How the policy changed as real ratings arrived, replayed in the order they were given.

    `events` are rating events (id, timestamp, feedback), e.g. load_feedback_events().
    At each event time only the ratings given up to that moment count, and a changed
    rating replaces the earlier one for that request (never adds a second reward), and
    the session cap is applied with what was known at that moment.
    Up to `points` evenly spaced snapshots, starting from the prior alone (step 0).
    Columns: step (number of rating events so far), timestamp, n_feedback, plus the
    policy_snapshot columns.
    """
    prior = prior if prior is not None else load_prior()
    empty = pd.DataFrame(columns=["step", "timestamp", "n_feedback"])
    requests = eligible_requests(logs, prompt_version)
    if events is None or events.empty or requests.empty:
        snap = snapshot_from_posterior(posterior_from_counts({}, {}, prior))
        return pd.concat([empty, snap.assign(step=0, timestamp=pd.NaT, n_feedback=0)], ignore_index=True)
    arm_of = dict(zip(requests["id"], requests["arm"]))
    sent_at = dict(zip(requests["id"], requests["timestamp"]))
    tag_of = dict(zip(requests["id"], requests["session_tag"])) if "session_tag" in requests else {}
    ev = events[events["id"].isin(arm_of) & pd.to_numeric(events["feedback"], errors="coerce").isin([1, -1])]
    ev = ev.sort_values("timestamp").reset_index(drop=True)
    marks = set(np.linspace(0, len(ev), min(points, len(ev)) + 1).round().astype(int))
    latest: dict[str, int] = {}
    frames = []
    for k in range(len(ev) + 1):
        if k:
            row = ev.iloc[k - 1]
            latest[row["id"]] = int(int(row["feedback"]) > 0)
        if k in marks:
            fb = pd.DataFrame({"id": list(latest), "arm": [arm_of[i] for i in latest], "reward": list(latest.values()),
                               "timestamp": [sent_at[i] for i in latest], "session_tag": [tag_of.get(i) for i in latest]})
            fb = apply_session_cap(fb)  # the same rule as eligible_feedback(), with what was known at step k
            snap = snapshot_from_posterior(posterior_from_counts(*_counts(fb) if len(fb) else ({}, {}), prior))
            snap.insert(0, "step", k)
            snap.insert(1, "timestamp", ev["timestamp"].iloc[k - 1] if k else pd.NaT)
            snap.insert(2, "n_feedback", len(fb))
            frames.append(snap)
    return pd.concat(frames, ignore_index=True)

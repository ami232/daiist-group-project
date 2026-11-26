"""Scientific monitoring of the writing-style bandit (WP6). Measurements only.

Every function takes the logs (logbook.load()) and, when it needs rating times,
the rating events (bandit.load_feedback_events()). Nothing is simulated,
smoothed or extrapolated here; when there is not enough data the value is None
(shown as "N/A"), never 0. Simulation results live in eval/ and are never read here.

Metrics
    response_rate        rated / feedback-eligible requests (the requests whose rating
                         would train the policy), overall and per style.
    reward_delay         seconds from the logged work order to its first rating:
                         n, median, p95; plus changed ratings and very fast ratings.
    coverage_by_arm      per style: served, eligible, rated, counted after the
                         session cap, response rate, share of the counted ratings.
    served_probabilities the arm_probs that were really logged when each request
                         was served, in time order (block means when there are many).
    non_response         response rate by risk band, review flag and style: if some
                         kinds of orders are rated more often, the learned preference
                         is about the orders people chose to rate.
    arm_balance          risk-band mix of the rated requests per style: a style that
                         happened to get easier cases would look better for that reason.
    belief_comparison    per style: observed reward, posterior mean with its 95%
                         interval, a propensity-weighted estimate (self-normalised IPS
                         with the logged arm_probs), and P(selected), with the gaps.
    sessions             sessions that rated, ratings over the session cap, untagged.
    evidence             all of the above as one JSON-serialisable dict.

Thresholds marked ASSUMPTION are descriptive flags for the report, not tests.
"""

import numpy as np
import pandas as pd

from maintai import bandit
from maintai.schemas import ARMS

FAST_RATING_S = 3.0  # ASSUMPTION: a rating sooner than this after the order was shown is "very fast"
NONRESPONSE_GAP = 0.20  # ASSUMPTION: flag when response rates of two groups differ by >= 20 points...
NONRESPONSE_MIN_N = 10  # ...and each group has at least this many eligible requests
BELIEF_GAP = 0.10  # ASSUMPTION: flag an observed-vs-posterior gap of 10 points or more (prior still weighs)
MAX_SERVED_POINTS = 60


def _none(value) -> float | None:
    """A finite float, or None (missing / NaN / inf)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _col(df: pd.DataFrame, name: str, default) -> pd.Series:
    return df[name] if name in df else pd.Series([default] * len(df), index=df.index, dtype=object)


def _valid_rating(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(_col(df, "feedback", None), errors="coerce").isin([1, -1])


def live_requests(logs: pd.DataFrame | None) -> pd.DataFrame:
    """Requests whose style was chosen by a compatible live policy (any source), in time order."""
    if logs is None or logs.empty or "arm_probs" not in logs:
        return pd.DataFrame()
    live = logs["arm_probs"].apply(lambda v: isinstance(v, dict) and len(v) > 0) & \
        _col(logs, "versions", None).apply(lambda v: isinstance(v, dict) and v.get("bandit") in bandit.COMPATIBLE_POLICIES)
    out = logs[live]
    return out.sort_values("timestamp", kind="stable") if "timestamp" in out else out


# --- Response rate and coverage ------------------------------------------------------

def response_rate(logs: pd.DataFrame | None, prompt_version: str | None = None) -> dict:
    """Share of feedback-eligible requests that received a valid rating."""
    eligible = bandit.eligible_requests(logs, prompt_version)
    live = live_requests(logs)
    rated = int(_valid_rating(eligible).sum()) if not eligible.empty else 0
    live_rated = int(_valid_rating(live).sum()) if not live.empty else 0
    return {"n_eligible": len(eligible), "n_rated": rated, "rate": _rate(rated, len(eligible)),
            "n_live": len(live), "n_live_rated": live_rated, "live_rate": _rate(live_rated, len(live))}


def coverage_by_arm(logs: pd.DataFrame | None, prompt_version: str | None = None) -> pd.DataFrame:
    """One row per style: served (live), eligible, rated, counted (after the session cap), rates."""
    live = live_requests(logs)
    eligible = bandit.eligible_requests(logs, prompt_version)
    counted = bandit.eligible_feedback(logs, prompt_version)
    rated = eligible[_valid_rating(eligible)] if not eligible.empty else eligible
    total_counted = len(counted)
    rows = []
    for arm in ARMS:
        n_served = int((_col(live, "arm", "") == arm).sum()) if not live.empty else 0
        n_eligible = int((_col(eligible, "arm", "") == arm).sum()) if not eligible.empty else 0
        n_rated = int((_col(rated, "arm", "") == arm).sum()) if not rated.empty else 0
        n_counted = int((counted["arm"] == arm).sum()) if total_counted else 0
        positive = int(counted[(counted["arm"] == arm)]["reward"].sum()) if total_counted else 0
        rows.append({"arm": arm, "served": n_served, "eligible": n_eligible, "rated": n_rated, "counted": n_counted,
                     "positive": positive, "response_rate": _rate(n_rated, n_eligible),
                     "share_of_counted": _rate(n_counted, total_counted)})
    return pd.DataFrame(rows)


# --- Reward delay ------------------------------------------------------------------

def reward_delay(logs: pd.DataFrame | None, events: pd.DataFrame | None,
                 prompt_version: str | None = None) -> dict:
    """Seconds from the logged order (request timestamp) to its FIRST rating, eligible requests only.

    Also: requests whose rating was changed (more than one event), ratings faster than
    FAST_RATING_S, and events dated before their request (clock problems, dropped).
    """
    out = {"n": 0, "median_s": None, "p95_s": None, "mean_s": None, "n_changed": 0, "n_fast": 0,
           "n_negative_dropped": 0, "fast_threshold_s": FAST_RATING_S}
    eligible = bandit.eligible_requests(logs, prompt_version)
    if events is None or events.empty or eligible.empty or "timestamp" not in eligible:
        return out
    valid = events[pd.to_numeric(events["feedback"], errors="coerce").isin([1, -1])]
    valid = valid[valid["id"].isin(set(eligible["id"]))]
    if valid.empty:
        return out
    sent = eligible.drop_duplicates("id").set_index("id")["timestamp"]
    first = valid.sort_values("timestamp", kind="stable").groupby("id")["timestamp"].first()
    delay = (first - sent.reindex(first.index)).dt.total_seconds().dropna()
    negative = delay < 0
    delay = delay[~negative]
    counts = valid.groupby("id").size()
    out.update({"n": int(len(delay)), "n_changed": int((counts > 1).sum()), "n_negative_dropped": int(negative.sum())})
    if len(delay):
        out.update({"median_s": float(delay.median()), "p95_s": float(np.percentile(delay, 95)),
                    "mean_s": float(delay.mean()), "n_fast": int((delay < FAST_RATING_S).sum())})
    return out


# --- Served probabilities ---------------------------------------------------------------

def served_probabilities(logs: pd.DataFrame | None, max_points: int = MAX_SERVED_POINTS) -> pd.DataFrame:
    """The arm_probs logged at serving time, oldest first: one row per request, or block means.

    Columns: block, first_request, last_request, n, timestamp (first in block), eligible
    (eligible requests in the block), p_<arm>. These are the probabilities the policy
    really used, not a replay.
    """
    live = live_requests(logs)
    cols = ["block", "first_request", "last_request", "n", "timestamp", "eligible"] + [f"p_{a}" for a in ARMS]
    if live.empty:
        return pd.DataFrame(columns=cols)
    probs = pd.DataFrame([{a: _none((v or {}).get(a)) for a in ARMS} for v in live["arm_probs"]], index=live.index)
    eligible_ids = set(bandit.eligible_requests(logs)["id"]) if "id" in live else set()
    frame = pd.DataFrame({"timestamp": live.get("timestamp"), "eligible": live.get("id", pd.Series(index=live.index)).isin(eligible_ids)})
    frame = pd.concat([frame, probs.add_prefix("p_")], axis=1).reset_index(drop=True)
    size = max(1, int(np.ceil(len(frame) / max_points)))
    frame["block"] = frame.index // size
    agg = frame.groupby("block").agg(first_request=("timestamp", lambda s: s.index.min() + 1),
                                     last_request=("timestamp", lambda s: s.index.max() + 1),
                                     n=("timestamp", "size"), timestamp=("timestamp", "first"),
                                     eligible=("eligible", "sum"),
                                     **{f"p_{a}": (f"p_{a}", "mean") for a in ARMS}).reset_index()
    return agg[cols]


# --- Feedback loop and non-response ------------------------------------------------------

def non_response(logs: pd.DataFrame | None, prompt_version: str | None = None) -> dict:
    """Response rate of eligible requests by risk band, review flag and style, with flags."""
    eligible = bandit.eligible_requests(logs, prompt_version)
    rows = []
    if not eligible.empty:
        df = eligible.assign(rated=_valid_rating(eligible),
                             risk_band=_col(eligible, "risk_band", "unknown").fillna("unknown"),
                             review=_col(eligible, "review", None).map({True: "human review", False: "automatic"}).fillna("unknown"))
        for dim in ("risk_band", "review", "arm"):
            for group, g in df.groupby(dim, sort=True):
                rows.append({"dimension": dim, "group": str(group), "eligible": len(g), "rated": int(g["rated"].sum()),
                             "response_rate": _rate(int(g["rated"].sum()), len(g))})
    table = pd.DataFrame(rows, columns=["dimension", "group", "eligible", "rated", "response_rate"])
    flags = []
    for dim, g in table.groupby("dimension") if not table.empty else []:
        big = g[g["eligible"] >= NONRESPONSE_MIN_N]
        if len(big) >= 2:
            gap = float(big["response_rate"].max() - big["response_rate"].min())
            if gap >= NONRESPONSE_GAP:
                hi, lo = big.loc[big["response_rate"].idxmax(), "group"], big.loc[big["response_rate"].idxmin(), "group"]
                flags.append({"dimension": dim, "gap": gap, "most_rated": hi, "least_rated": lo})
    tested = sum(1 for _, g in (table.groupby("dimension") if not table.empty else [])
                 if (g["eligible"] >= NONRESPONSE_MIN_N).sum() >= 2)
    return {"table": table, "flags": flags, "dimensions_with_enough_data": tested,
            "rule": f"gap >= {NONRESPONSE_GAP:.0%} between groups with >= {NONRESPONSE_MIN_N} eligible requests (assumption)"}


def arm_balance(logs: pd.DataFrame | None, prompt_version: str | None = None) -> pd.DataFrame:
    """Risk-band mix of the counted ratings per style. Thompson sampling ignores the input,
    so the mix should only differ by chance; a large difference confounds the comparison."""
    counted_ids = set(bandit.eligible_feedback(logs, prompt_version)["id"])
    eligible = bandit.eligible_requests(logs, prompt_version)
    df = eligible[eligible["id"].isin(counted_ids)] if not eligible.empty else eligible
    if df.empty:
        return pd.DataFrame(columns=["arm", "risk_band", "n", "share"])
    df = df.assign(risk_band=_col(df, "risk_band", "unknown").fillna("unknown"))
    out = df.groupby(["arm", "risk_band"]).size().rename("n").reset_index()
    out["share"] = out["n"] / out.groupby("arm")["n"].transform("sum")
    return out


# --- Observed reward vs belief vs selection --------------------------------------------------

def belief_comparison(logs: pd.DataFrame | None, prompt_version: str | None = None) -> pd.DataFrame:
    """Per style: observed reward, posterior mean and 95% interval, IPS estimate, P(selected).

    - observed: positive / counted ratings (real data only; None without ratings).
    - posterior mean: (1 + positive) / (2 + counted) with the neutral prior: pulled towards
      50% while there are few ratings.
    - ips: self-normalised inverse-propensity estimate, weighting each rating by 1 / the
      logged probability of the style that was served. EXPLORATORY: it re-weights for the
      policy having served some styles more often over time. It is not a causal proof,
      it is noisy with few ratings or small logged probabilities, and it does NOT correct
      non-response: it assumes ratings are missing at random given the style (see
      non_response). None when no logged probability is usable.
    - p_select: probability of being chosen next. It is the probability of being the BEST
      style, not a reward estimate: it can be high with a modest mean if the others are worse.
    """
    snap = bandit.policy_snapshot(logs, prompt_version=prompt_version).set_index("arm")
    counted = bandit.eligible_feedback(logs, prompt_version)
    eligible = bandit.eligible_requests(logs, prompt_version)
    probs = dict(zip(eligible["id"], eligible["arm_probs"])) if not eligible.empty else {}
    rows = []
    for arm in ARMS:
        mine = counted[counted["arm"] == arm] if not counted.empty else counted
        n = len(mine)
        observed = float(mine["reward"].mean()) if n else None
        weights = [1 / p for p in ((_none((probs.get(i) or {}).get(arm))) for i in mine["id"]) if p] if n else []
        rewards = [r for i, r in zip(mine["id"], mine["reward"]) if _none((probs.get(i) or {}).get(arm))] if n else []
        ips = float(np.dot(weights, rewards) / sum(weights)) if weights else None
        s = snap.loc[arm]
        rows.append({"arm": arm, "n": n, "observed": observed, "posterior_mean": float(s["mean"]),
                     "ci_low": float(s["ci_low"]), "ci_high": float(s["ci_high"]), "ips": ips, "p_select": float(s["p_select"]),
                     "shrinkage": None if observed is None else float(s["mean"]) - observed,
                     "prior_dominant": observed is not None and abs(float(s["mean"]) - observed) >= BELIEF_GAP})
    return pd.DataFrame(rows)


# --- Sessions ------------------------------------------------------------------------

def sessions(logs: pd.DataFrame | None, prompt_version: str | None = None) -> dict:
    """How ratings are spread over session tags, and how many the cap removed (no tags exported)."""
    rated = bandit._rated(logs, prompt_version)
    counted = bandit.eligible_feedback(logs, prompt_version)
    out = {"cap": bandit.MAX_RATINGS_PER_SESSION, "n_rated": len(rated), "n_counted": len(counted),
           "n_over_cap": len(rated) - len(counted), "n_sessions": 0, "n_untagged": len(rated),
           "max_per_session": None, "median_per_session": None}
    if rated.empty or "session_tag" not in rated:
        return out
    tag = rated["session_tag"]
    tagged = tag.notna() & (tag.astype(str).str.len() > 0)
    per = rated[tagged].groupby("session_tag").size()
    out.update({"n_sessions": int(len(per)), "n_untagged": int((~tagged).sum()),
                "max_per_session": int(per.max()) if len(per) else None,
                "median_per_session": float(per.median()) if len(per) else None})
    return out


# --- Everything, for the dashboard and the evidence export ---------------------------------------

def _records(df: pd.DataFrame) -> list[dict]:
    clean = df.astype(object).where(pd.notna(df), None)
    return [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()} for r in clean.to_dict("records")]


def evidence(logs: pd.DataFrame | None, events: pd.DataFrame | None, prompt_version: str | None = None) -> dict:
    """All RL monitoring metrics as plain JSON types. Aggregates only: no ids, inputs or session tags."""
    prompt_version = prompt_version or bandit.current_prompt_version()
    logs = logs if logs is not None else pd.DataFrame()
    nr = non_response(logs, prompt_version)
    counted = bandit.eligible_feedback(logs, prompt_version)
    source = _col(logs, "source", None).fillna("unlabelled").value_counts().to_dict() if not logs.empty else {}
    return {
        "policy_version": bandit.POLICY_VERSION, "prompt_version": prompt_version,
        "prior": "neutral Beta(1, 1)", "n_requests": len(logs), "requests_by_source": {str(k): int(v) for k, v in source.items()},
        "n_eligible_ratings": len(counted), "learning_from_real_feedback": bool(len(counted)),
        "response_rate": response_rate(logs, prompt_version),
        "reward_delay": reward_delay(logs, events, prompt_version),
        "events_error": (events.attrs.get("error") if events is not None else "no events given"),
        "coverage_by_arm": _records(coverage_by_arm(logs, prompt_version)),
        "belief_comparison": _records(belief_comparison(logs, prompt_version)),
        "non_response": {"table": _records(nr["table"]), "flags": nr["flags"], "rule": nr["rule"],
                         "dimensions_with_enough_data": nr["dimensions_with_enough_data"]},
        "arm_balance": _records(arm_balance(logs, prompt_version)),
        "sessions": sessions(logs, prompt_version),
        "assumptions": {"fast_rating_s": FAST_RATING_S, "nonresponse_gap": NONRESPONSE_GAP,
                        "nonresponse_min_n": NONRESPONSE_MIN_N, "belief_gap": BELIEF_GAP,
                        "max_ratings_per_session": bandit.MAX_RATINGS_PER_SESSION},
    }

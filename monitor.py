"""
Monitoring: quality signals over time, drift against the reference set, alerts.

The thresholds below are PLACEHOLDERS. Choose yours and justify them in
REPORT.md: how many false alarms on normal inputs you accept, and how big a
shift has to be before it matters for your product.
"""

import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REFERENCE_FILE = ROOT / "monitoring" / "reference_features.csv"

PSI_BINS = 5           # quantile bins of the reference; few, because samples are small
SMOOTHING = 0.5        # WP5: pseudo-count added to every bin (Laplace-style), so an empty bin in a small window does not blow up the PSI
PSI_ALERT = 0.25       # WP5: alert when a feature's PSI exceeds this...
NOISE_MULTIPLE = 5     # WP5: ...and this many times the PSI expected by chance (psi_noise). Calibrated on validation: eval/drift_calibration.py
WINDOW = 100           # the drift check looks at the last WINDOW requests...
MIN_WINDOW = 20        # ...and stays silent with fewer than this
THUMBS_DOWN_ALERT = 0.4  # PLACEHOLDER: alert when this share of rated requests is 👎

# WP5 operations alerts (see docs/WP5_OPERATIONS.md for the reasoning)
BUDGET_ALERT = 0.8       # alert when today's Gemini requests exceed this share of the daily budget
DEGRADED_WINDOW = 50     # degraded-mode share is measured on the last N requests that should have used the LLM...
DEGRADED_ALERT = 0.2     # ...and alerts above this share...
MIN_DEGRADED = 10        # ...with at least this many (it must react fast: a spent quota degrades every request)
REVIEW_ALERT = 0.5       # alert when this share of the last WINDOW requests is sent to review (validation: ~4%; feedback campaign mix: 30%)
BUDGET_TZ = ZoneInfo("America/Los_Angeles")  # the Gemini daily quota resets on Pacific time


def _bin(values: np.ndarray, reference: np.ndarray, bins: int) -> np.ndarray:
    """Bin index of each value, with bins defined by the reference.

    A feature with few distinct values in the reference (counts, labels coded as
    numbers) gets one bin per value plus one for anything else; a continuous one
    gets quantile bins, so each holds about the same share of the reference.
    """
    distinct = np.unique(reference)
    if len(distinct) <= bins:
        index = np.searchsorted(distinct, values)
        found = (index < len(distinct)) & (distinct[np.minimum(index, len(distinct) - 1)] == values)
        return np.where(found, index, len(distinct))
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right")


def psi(reference, current, bins: int = PSI_BINS, smoothing: float = SMOOTHING) -> float:
    """Population stability index of `current` against `reference`.

    0 means the same distribution; the usual rule of thumb is < 0.1 stable,
    0.1-0.25 moderate, > 0.25 a large shift. With small samples, though, PSI is
    large by chance alone: see `psi_noise`.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    ref_bins, cur_bins = _bin(reference, reference, bins), _bin(current, reference, bins)
    n_bins = max(ref_bins.max(), cur_bins.max()) + 1
    # Additive smoothing instead of clipping at 1e-4: with 16-60 values a single empty bin
    # would otherwise add about 1.5 to the PSI on its own (see eval/drift_calibration.py).
    expected = (np.bincount(ref_bins, minlength=n_bins) + smoothing) / (len(reference) + smoothing * n_bins)
    actual = (np.bincount(cur_bins, minlength=n_bins) + smoothing) / (len(current) + smoothing * n_bins)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_noise(n_reference: int, n_current: int, bins: int = PSI_BINS) -> float:
    """PSI you would expect by chance, with no drift at all, for these sample sizes.

    About (bins - 1) * (1/n_reference + 1/n_current): with 40 reference inputs and
    20 live ones that is already 0.3. An alert below this is a coin toss.
    """
    return (bins - 1) * (1 / n_reference + 1 / n_current)


def load_reference() -> pd.DataFrame | None:
    """Features of the reference inputs (`main.py reference` writes them)."""
    return pd.read_csv(REFERENCE_FILE) if REFERENCE_FILE.is_file() else None


def drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI per feature, and whether it alerts. Both frames have one column per feature (no `f_` prefix).

    The rule: alert when PSI exceeds PSI_ALERT *and* NOISE_MULTIPLE times what
    chance alone would give with these sample sizes.
    """
    rows = []
    for feature in reference.columns:
        if feature in current.columns:
            value = psi(reference[feature], current[feature])
            noise = psi_noise(reference[feature].notna().sum(), max(current[feature].notna().sum(), 1))
            threshold = max(PSI_ALERT, NOISE_MULTIPLE * noise)
            rows.append({"feature": feature, "psi": round(value, 3), "threshold": round(threshold, 3), "alert": bool(value > threshold)})
    return pd.DataFrame(rows, columns=["feature", "psi", "threshold", "alert"])


def live_features(logs: pd.DataFrame) -> pd.DataFrame:
    """The features of the last WINDOW logged requests, without the `f_` prefix."""
    columns = [c for c in logs.columns if c.startswith("f_")]
    return logs[columns].tail(WINDOW).rename(columns=lambda c: c[2:])


def quality(logs: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Quality signals per period, in long format (time, signal, value) for a line plot."""
    if logs.empty:
        return pd.DataFrame(columns=["time", "signal", "value"])
    df = logs.set_index("timestamp")
    rated = df["feedback"].dropna().astype(float)
    signals = pd.DataFrame({
        "share sent to review": df["review"].astype(float).resample(freq).mean(),
        "share 👎 (of rated)": (rated < 0).astype(float).resample(freq).mean(),
    })
    if "f_confidence" in df:
        signals["mean confidence"] = df["f_confidence"].resample(freq).mean()
    long = signals.reset_index(names="time").melt("time", var_name="signal", value_name="value")
    return long.dropna()


def summary(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame({"metric": ["requests"], "value": [0]})
    rated = logs["feedback"].dropna().astype(float)
    rows = {
        "requests": len(logs),
        "sent to review": f"{logs['review'].mean():.0%}",
        "rated 👍/👎": len(rated),
        "share 👎 (of rated)": f"{(rated < 0).mean():.0%}" if len(rated) else "-",
        "median latency (s)": round(float(logs["latency_s"].median()), 2),
        "mean cost per request (€)": round(float(logs["cost_eur"].mean()), 5),
        "total cost (€)": round(float(logs["cost_eur"].sum()), 4),
    }
    return pd.DataFrame({"metric": list(rows), "value": [str(v) for v in rows.values()]})


_STUB = re.compile(r"(^|\|)stub", re.IGNORECASE)


def real_traffic(logs: pd.DataFrame) -> pd.DataFrame:
    """Requests that count as evidence: source `real` (or not recorded) and no stub component.

    Smoke tests, check runs, simulations (`source` other than `real`) and stub versions
    (`stub-...`) are left out of the quality, drift and operations numbers.
    """
    if logs.empty:
        return logs
    keep = pd.Series(True, index=logs.index)
    if "source" in logs:
        keep &= logs["source"].fillna("real").astype(str).str.lower().eq("real")
    if "versions" in logs:
        keep &= ~logs["versions"].map(lambda v: isinstance(v, dict) and any(_STUB.search(str(x)) for x in v.values()))
    return logs[keep]


def _col(df: pd.DataFrame, name: str, default):
    return df[name] if name in df else pd.Series([default] * len(df), index=df.index)


def _live(df: pd.DataFrame) -> pd.DataFrame:
    """Requests that went to Gemini for real: served (not refused) and not answered from the cache."""
    if df.empty:
        return df
    ok = _col(df, "status", "ok").fillna("ok").astype(str).eq("ok")
    calls = pd.to_numeric(_col(df, "llm_calls", 0), errors="coerce").fillna(0) > 0
    cached = _col(df, "cache_hit", False).fillna(False).astype(bool)
    return df[ok & calls & ~cached]


def _usage(df: pd.DataFrame, field: str) -> int:
    return int(sum((u.get(field, 0) or 0) for u in _col(df, "llm", None) if isinstance(u, dict)))


def budget_today(logs: pd.DataFrame) -> tuple[int, int] | None:
    """(Gemini requests used today, daily budget) from the latest log that recorded it, or None.

    Uses every log (also tests) because the quota is shared; only a record of the current
    Pacific-time day counts, an older one is a different day's quota."""
    if logs.empty or "llm_budget" not in logs:
        return None
    today = datetime.now(BUDGET_TZ).date().isoformat()
    for budget in reversed(logs["llm_budget"].tolist()):
        if isinstance(budget, dict) and str(budget.get("day")) == today and budget.get("limit"):
            return int(budget["used"]), int(budget["limit"])
    return None


def operations(logs: pd.DataFrame) -> pd.DataFrame:
    """Cost, tokens, latency, degradation and quota of the real traffic, as metric / value rows."""
    real = real_traffic(logs)
    live = _live(real)
    served = real[_col(real, "status", "ok").fillna("ok").astype(str).eq("ok")] if not real.empty else real
    recent = live.tail(DEGRADED_WINDOW)
    degraded = _col(recent, "degraded", False).fillna(False).astype(bool)
    limited = _col(live, "rate_limited", False).fillna(False).astype(bool)
    budget = budget_today(logs)

    def pct(series: pd.Series, q: float) -> str:
        return f"{float(series.quantile(q)):.2f}" if len(series) else "-"

    rows = {
        "real requests (excluding stubs and tests)": len(real),
        "requests that called Gemini (no cache)": len(live),
        "tokens in / out / thinking": f"{_usage(live, 'tokens_in')} / {_usage(live, 'tokens_out')} / {_usage(live, 'tokens_thinking')}" if len(live) else "-",
        "mean cost per Gemini request (€)": f"{float(live['cost_eur'].mean()):.5f}" if len(live) else "-",
        "latency p50 / p95 (s), all served": f"{pct(served['latency_s'], 0.5)} / {pct(served['latency_s'], 0.95)}" if len(served) else "-",
        "latency p50 / p95 (s), Gemini requests": f"{pct(live['latency_s'], 0.5)} / {pct(live['latency_s'], 0.95)}" if len(live) else "-",
        f"degraded mode, last {len(recent)} Gemini requests": f"{degraded.mean():.0%} ({int(degraded.sum())} of {len(recent)})" if len(recent) else "-",
        "rate limited (429) requests": f"{int(limited.sum())} of {len(live)}" if len(live) else "-",
        "retries": int(pd.to_numeric(_col(live, "retries", 0), errors="coerce").fillna(0).sum()),
        "Gemini requests today / budget": f"{budget[0]} / {budget[1]} ({budget[0] / budget[1]:.0%})" if budget else "-",
    }
    return pd.DataFrame({"metric": list(rows), "value": [str(v) for v in rows.values()]})


def quality_counts(logs: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Denominators behind `quality`: per period, requests, sent to review, rated and 👎 (real traffic only)."""
    real = real_traffic(logs)
    columns = ["time", "requests", "sent to review", "rated", "thumbs down"]
    if real.empty:
        return pd.DataFrame(columns=columns)
    df = real.set_index("timestamp")
    rated = df["feedback"].dropna().astype(float)
    table = pd.DataFrame({
        "requests": df["review"].resample(freq).size(),
        "sent to review": df["review"].astype(float).resample(freq).sum(),
        "rated": rated.resample(freq).size(),
        "thumbs down": (rated < 0).astype(float).resample(freq).sum(),
    }).fillna(0).astype(int)
    return table[table["requests"] > 0].reset_index(names="time")[columns]



def alerts(logs: pd.DataFrame, reference: pd.DataFrame | None) -> list[str]:
    """The alert rules. Each fired alert is one message.

    Drift, quality and degradation use real traffic only (see `real_traffic`); the quota
    alert uses every log, because tests spend the same Gemini quota."""
    fired = []
    budget = budget_today(logs)
    logs = real_traffic(logs)
    window = logs.tail(WINDOW)
    if reference is not None and len(window) >= MIN_WINDOW:
        report = drift(reference, live_features(window))
        for row in report[report["alert"]].itertuples():
            fired.append(f"Drift in `{row.feature}`: PSI {row.psi} > {row.threshold} over the last {len(window)} requests.")
    rated = window["feedback"].dropna().astype(float) if not window.empty else pd.Series(dtype=float)
    if len(rated) >= MIN_WINDOW and (rated < 0).mean() > THUMBS_DOWN_ALERT:
        fired.append(f"{(rated < 0).mean():.0%} of the last {len(rated)} rated requests are 👎 (> {THUMBS_DOWN_ALERT:.0%}).")
    if len(window) >= MIN_WINDOW and window["review"].astype(float).mean() > REVIEW_ALERT:
        fired.append(f"{window['review'].astype(float).mean():.0%} of the last {len(window)} requests were sent to human review (> {REVIEW_ALERT:.0%}).")
    if budget and budget[0] > BUDGET_ALERT * budget[1]:
        fired.append(f"Gemini quota: {budget[0]} of {budget[1]} requests used today ({budget[0] / budget[1]:.0%} > {BUDGET_ALERT:.0%}).")
    recent = _live(logs).tail(DEGRADED_WINDOW)
    if len(recent) >= MIN_DEGRADED:
        share = _col(recent, "degraded", False).fillna(False).astype(bool).mean()
        if share > DEGRADED_ALERT:
            fired.append(f"Degraded mode in {share:.0%} of the last {len(recent)} Gemini requests (> {DEGRADED_ALERT:.0%}).")
    return fired

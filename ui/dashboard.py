"""Monitoring dashboard (presentation only).  OWNER of this file: WP7.

The calculations stay in monitor.py (WP5): summary(), quality(), drift(),
alerts() and their thresholds. This file only lays out what they return, plus
plain descriptive counts of the logged requests (how many per hour, per risk
band, per bandit arm). Nothing is estimated, smoothed or extrapolated, and
every section has an empty state for when there is not enough data yet.

When WP5 delivers monitoring_ui.render(), it can reuse these mai-* blocks.
"""

import html

import pandas as pd

import monitor
from ui import charts
from ui.icons import icon

SIGNAL_LABELS = {"share sent to review": "Sent to review", "share 👎 (of rated)": "Negative feedback",
                 "mean confidence": "Mean confidence"}
BAND_TONES = [("low", "Low", "success"), ("medium", "Medium", "warning"), ("high", "High", "orange"), ("critical", "Critical", "danger")]
ARM_LABELS = {"checklist": "Checklist", "cost": "Cost-focused", "root_cause": "Root-cause"}


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _clean(message: str) -> str:
    """Alert text from monitor.py, without emoji or markdown, for the HTML view."""
    return message.replace("👎", "negative").replace("`", "")


def _card(title: str, icon_name: str, body: str, hint: str = "", cls: str = "") -> str:
    hint_html = f'<span class="mai-card__hint">{_e(hint)}</span>' if hint else ""
    return (f'<section class="mai-card {cls}"><div class="mai-card__header"><span class="mai-card__icon">{icon(icon_name, 16)}</span>'
            f'<h2 class="mai-h2">{_e(title)}</h2>{hint_html}</div>{body}</section>')


def _empty(icon_name: str, title: str, text: str, progress: tuple[int, int] | None = None) -> str:
    bar = ""
    if progress:
        n, need = progress
        bar = (f'<div class="mai-progress" role="progressbar" aria-valuemin="0" aria-valuemax="{need}" aria-valuenow="{min(n, need)}">'
               f'<span style="width:{min(n / need, 1) * 100:.0f}%"></span></div><span class="mai-subtle">{min(n, need)} of {need} requests</span>')
    return (f'<div class="mai-chart-empty">{icon(icon_name, 22)}<strong>{_e(title)}</strong>'
            f'<span class="mai-subtle">{text}</span>{bar}</div>')


def _kpi(icon_name: str, label: str, value: str, foot: str, tone: str = "") -> str:
    return (f'<div class="mai-kpi mai-kpi--tile {f"mai-kpi--{tone}" if tone else ""}">'
            f'<span class="mai-kpi__icon">{icon(icon_name, 18)}</span>'
            f'<span class="mai-kpi__label">{_e(label)}</span><span class="mai-kpi__value">{_e(value)}</span>'
            f'<span class="mai-kpi__foot">{_e(foot)}</span></div>')


NA = "N/A"  # shown when a metric is missing or cannot be computed: never a false zero


def _summary(logs: pd.DataFrame) -> dict:
    """monitor.summary() as {metric: value}. Empty if it fails or changes shape (WP5 owns it)."""
    try:
        table = monitor.summary(logs)
        return dict(zip(table["metric"], table["value"]))
    except Exception:  # noqa: BLE001 - the dashboard shows N/A instead of failing
        return {}


def _number(summary: dict, key: str) -> float | None:
    """A numeric metric, or None if it is absent, "-", empty or not a finite number."""
    value = summary.get(key)
    if value is None:
        return None
    try:
        number = float(str(value).strip().rstrip("%")) / (100 if str(value).strip().endswith("%") else 1)
    except ValueError:
        return None
    return number if pd.notna(number) and abs(number) != float("inf") else None


def _share_text(summary: dict, key: str) -> str:
    number = _number(summary, key)
    return NA if number is None else f"{number:.0%}"


def _latency_text(value) -> str:
    """monitor.summary() rounds to 0.01 s; below that, say so instead of showing 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return NA
    if pd.isna(v):
        return NA
    return "< 0.01 s" if v == 0 else (f"{v * 1000:.0f} ms" if v < 1 else f"{v:.2f} s")


def _is_stub(versions) -> bool:
    return isinstance(versions, dict) and any("stub" in str(v) for v in versions.values())


def render(logs: pd.DataFrame, reference: pd.DataFrame | None, storage: str) -> str:
    if logs.empty:
        return f"""
<div class="mai-dash">
  <div class="mai-card mai-empty">
    <div class="mai-empty__head"><div class="mai-empty__icon">{icon("monitor", 26)}</div>
    <div><h2 class="mai-h2">No requests logged yet</h2>
    <p class="mai-subtle">Run an analysis in the App tab. KPIs, charts and drift checks appear here as soon as
    requests are logged ({_e(storage)}). Nothing is shown before there is data.</p></div></div>
  </div>
  {_rl(logs)}
</div>"""

    summary = _summary(logs)
    n = len(logs)
    stub_n = int(logs["versions"].apply(_is_stub).sum()) if "versions" in logs else 0
    oos_n = int((logs.get("status", pd.Series(dtype=str)) == "out_of_scope").sum())
    rated_n = _number(summary, "rated 👍/👎")
    rated = int(rated_n) if rated_n is not None else None

    provisional = ""
    if stub_n:
        provisional = (f'<div class="mai-banner mai-banner--stub" role="note">{icon("flask", 18)}<div><strong>Provisional data.</strong> '
                       f'{stub_n} of {n} logged requests come from stub components (development and smoke tests). They are '
                       f'shown here for monitoring but excluded from the evaluation and from the bandit.</div></div>')

    neg = _number(summary, "share 👎 (of rated)")  # None when nothing is rated ("-") or the metric is missing
    mean_cost, total_cost = _number(summary, "mean cost per request (€)"), _number(summary, "total cost (€)")
    latency = _number(summary, "median latency (s)")
    kpis = "".join([
        _kpi("requests", "Requests", f"{n:,}", f"{oos_n} out of scope"),
        _kpi("user-check", "Sent to review", _share_text(summary, "sent to review"), "Share of all requests"),
        _kpi("thumb-down", "Negative feedback", NA if neg is None else f"{neg:.0%}",
             "No ratings yet" if rated == 0 else ("Not available" if rated is None else f"{rated} rated request(s)"),
             "warning" if neg else ""),
        _kpi("clock", "Median latency", NA if latency is None else _latency_text(latency), "Per request, end to end"),
        _kpi("coins", "AI cost / request", NA if mean_cost is None else f"€{mean_cost:.4f}",
             ("Total not available" if total_cost is None else f"Total €{total_cost:.4f}") + " · production-equivalent"),
    ])

    return f"""
<div class="mai-dash">
  {provisional}
  <div class="mai-kpi-row mai-kpi-row--5">{kpis}</div>
  {_alerts_panel(logs, reference)}
  <div class="mai-grid">
    {_card("Requests over time", "requests", _requests_chart(logs), hint="Count per period")}
    {_card("Quality signals", "pulse", _quality_chart(logs), hint="monitor.quality()")}
  </div>
  {_drift_panel(logs, reference)}
  <div class="mai-grid">
    {_card("Latency", "clock", _latency_chart(logs, summary), hint="Last 50 requests")}
    {_card("Outcome mix", "shield", _mix(logs), hint="All logged requests")}
  </div>
  {_rl(logs)}
  <p class="mai-footnote">Thresholds come from monitor.py and are placeholders until WP5 calibrates them on validation data.
  Logs: {_e(storage)}.</p>
</div>"""


def _alerts_panel(logs: pd.DataFrame, reference: pd.DataFrame | None) -> str:
    fired = monitor.alerts(logs, reference)
    window = min(len(logs), monitor.WINDOW)
    rows = [("high", "Alert", _clean(m)) for m in fired]
    if reference is None:
        rows.append(("medium", "Inactive", "Drift monitoring has no reference set yet (run `main.py reference`, owned by WP5)."))
    elif window < monitor.MIN_WINDOW:
        rows.append(("info", "Waiting", f"Drift check needs {monitor.MIN_WINDOW} requests in the window; {window} so far."))
    if not rows:
        rows.append(("ok", "Healthy", "No alert rule fired over the last window."))
    tone = {"high": "danger", "medium": "warning", "info": "info", "ok": "success"}
    ico = {"high": "alert", "medium": "bell", "info": "info", "ok": "check"}
    items = "".join(f'<li class="mai-alert mai-alert--{s}"><span class="mai-alert__icon">{icon(ico[s], 16)}</span>'
                    f'<span class="mai-badge mai-badge--{tone[s]}">{_e(label)}</span><span>{_e(text).replace("`", "")}</span></li>'
                    for s, label, text in rows)
    rules = (f'Drift: PSI &gt; max({monitor.PSI_ALERT}, {monitor.NOISE_MULTIPLE} × chance level) over the last {monitor.WINDOW} requests · '
             f'Feedback: more than {monitor.THUMBS_DOWN_ALERT:.0%} negative among at least {monitor.MIN_WINDOW} rated')
    return _card("Alerts", "bell", f'<ul class="mai-alerts">{items}</ul><p class="mai-footnote">Rules: {rules}.</p>',
                 hint=f"{len(fired)} active")


def _period(logs: pd.DataFrame) -> tuple[str, str]:
    span = logs["timestamp"].max() - logs["timestamp"].min()
    return ("h", "%d %b %H:%M") if span <= pd.Timedelta(hours=48) else ("D", "%d %b")


def _requests_chart(logs: pd.DataFrame) -> str:
    freq, fmt = _period(logs)
    counts = logs.set_index("timestamp").resample(freq).size()
    if len(counts) < 2:
        return _empty("requests", "One period so far", f"{int(counts.sum())} request(s) in a single period. The chart needs at least two periods.")
    return charts.column_chart([float(v) for v in counts.values], [t.strftime(fmt) for t in counts.index], title="Requests per period")


def _quality_chart(logs: pd.DataFrame) -> str:
    q = monitor.quality(logs)
    if q.empty or q["time"].nunique() < 2:
        return _empty("pulse", "Not enough history", "Quality signals are drawn per hour once there are at least two hours of requests.")
    wide = q.pivot_table(index="time", columns="signal", values="value").sort_index()
    labels = [t.strftime("%d %b %H:%M") for t in wide.index]
    series = {SIGNAL_LABELS.get(c, c): [None if pd.isna(v) else float(v) for v in wide[c]] for c in wide.columns}
    return charts.line_chart(series, labels, y_max=1.0, title="Quality signals per hour")


def _drift_panel(logs: pd.DataFrame, reference: pd.DataFrame | None) -> str:
    window = min(len(logs), monitor.WINDOW)
    if reference is None:
        body = _empty("drift", "No reference set yet", "Drift is measured against a reference set of normal inputs. "
                      "WP5 creates it with <code>uv run python main.py reference</code>.")
    elif window < monitor.MIN_WINDOW:
        body = _empty("drift", "Collecting requests", f"PSI is computed once the window holds {monitor.MIN_WINDOW} requests.",
                      progress=(window, monitor.MIN_WINDOW))
    else:
        report = monitor.drift(reference, monitor.live_features(logs))
        top = max(float(report["psi"].max()), float(report["threshold"].max())) * 1.1 or 1
        rows = ""
        for r in report.itertuples():
            state = "risk" if r.alert else "ok"
            rows += (f'<div class="mai-psi mai-psi--{state}"><span class="mai-psi__name">{_e(r.feature)}</span>'
                     f'<div class="mai-psi__track" role="img" aria-label="{_e(r.feature)} PSI {r.psi}, threshold {r.threshold}">'
                     f'<span class="mai-psi__bar" style="width:{min(r.psi / top, 1) * 100:.1f}%"></span>'
                     f'<span class="mai-psi__thr" style="left:{min(r.threshold / top, 1) * 100:.1f}%"></span></div>'
                     f'<span class="mai-psi__val">{r.psi:.3f}</span></div>')
        body = (f'<div class="mai-psis">{rows}</div><ul class="mai-legend"><li><span style="background:#007C9E"></span>PSI, last {window} requests</li>'
                f'<li><span class="mai-legend__line"></span>Alert threshold (noise-adjusted)</li><li><span style="background:#C62828"></span>Above threshold</li></ul>')
    return _card("Data drift · PSI per feature", "drift", body, hint=f"Window {window} of {monitor.WINDOW}")


def _latency_chart(logs: pd.DataFrame, summary: dict) -> str:
    recent = logs.tail(50)
    lat = [None if pd.isna(v) else float(v) for v in pd.to_numeric(recent["latency_s"], errors="coerce")]
    if sum(v is not None for v in lat) < 2:  # missing latencies are gaps, not zeros
        return _empty("clock", "Not enough requests", "Latency is drawn once two or more requests are logged.")
    median = _number(summary, "median latency (s)")
    labels = [t.strftime("%H:%M") for t in recent["timestamp"]]
    reference = None if median is None else (median, f"Median {_latency_text(median)} (all requests)")
    return charts.line_chart({"Latency": lat}, labels, y_fmt=_seconds, reference=reference, title="Latency per request")


def _seconds(v: float) -> str:
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.1f} s"


def _mix(logs: pd.DataFrame) -> str:
    status = logs.get("status", pd.Series("ok", index=logs.index)).fillna("ok")
    handled = status != "out_of_scope"
    review = logs["review"].astype(bool) & handled
    outcome = charts.segmented_bar([("Automatic", int((handled & ~review).sum()), "success"),
                                    ("Human review", int(review.sum()), "warning"),
                                    ("Out of scope", int((~handled).sum()), "neutral")])
    bands = logs.get("risk_band", pd.Series(dtype=str))
    band_bar = charts.segmented_bar([(label, int((bands == key).sum()), tone) for key, label, tone in BAND_TONES])
    return (f'<h3 class="mai-h3">Decision</h3>{outcome}'
            f'<h3 class="mai-h3">Risk band</h3>{band_bar or "<p class=mai-subtle>No handled requests yet.</p>"}')


def _rl(logs: pd.DataFrame) -> str:
    """Adaptive Intelligence section (ui/rl_dashboard.py, WP6)."""
    from ui import rl_dashboard

    return rl_dashboard.render(logs)

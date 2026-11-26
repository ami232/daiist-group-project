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


def _latency_text(value) -> str:
    """monitor.summary() rounds to 0.01 s; below that, say so instead of showing 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "–"
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
  {_rl_panel(logs)}
</div>"""

    summary = dict(zip(monitor.summary(logs)["metric"], monitor.summary(logs)["value"]))
    n = len(logs)
    stub_n = int(logs["versions"].apply(_is_stub).sum()) if "versions" in logs else 0
    oos_n = int((logs.get("status", pd.Series(dtype=str)) == "out_of_scope").sum())
    rated = int(float(summary.get("rated 👍/👎", 0) or 0))

    provisional = ""
    if stub_n:
        provisional = (f'<div class="mai-banner mai-banner--stub" role="note">{icon("flask", 18)}<div><strong>Provisional data.</strong> '
                       f'{stub_n} of {n} logged requests come from stub components (development and smoke tests). They are '
                       f'shown here for monitoring but excluded from the evaluation and from the bandit.</div></div>')

    neg = summary.get("share 👎 (of rated)", "-")
    kpis = "".join([
        _kpi("requests", "Requests", f"{n:,}", f"{oos_n} out of scope"),
        _kpi("user-check", "Sent to review", summary.get("sent to review", "-"), "Share of all requests"),
        _kpi("thumb-down", "Negative feedback", neg if rated else "–", f"{rated} rated request(s)",
             "warning" if rated and neg not in ("-", "0%") else ""),
        _kpi("clock", "Median latency", _latency_text(summary.get("median latency (s)")), "Per request, end to end"),
        _kpi("coins", "AI cost / request", f'€{float(summary.get("mean cost per request (€)", 0)):.4f}',
             f'Total €{float(summary.get("total cost (€)", 0)):.4f} · production-equivalent'),
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
  {_rl_panel(logs)}
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
    lat = [float(v) for v in recent["latency_s"].fillna(0)]
    if len(lat) < 2:
        return _empty("clock", "Not enough requests", "Latency is drawn once two or more requests are logged.")
    median = float(summary.get("median latency (s)", 0) or 0)
    labels = [t.strftime("%H:%M") for t in recent["timestamp"]]
    return charts.line_chart({"Latency": lat}, labels, y_fmt=_seconds,
                             reference=(median, f"Median {_latency_text(median)} (all requests)"), title="Latency per request")


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


def _rl_panel(logs: pd.DataFrame) -> str:
    """Reserved for WP6. Shows only logged facts: requests and ratings per writing style."""
    rows = ""
    policy = "–"
    if not logs.empty and "arm" in logs:
        last = logs.iloc[-1]
        probs = last.get("arm_probs") if isinstance(last.get("arm_probs"), dict) else {}
        versions = last.get("versions") if isinstance(last.get("versions"), dict) else {}
        policy = versions.get("bandit", "–")
        for arm, label in ARM_LABELS.items():
            sub = logs[logs["arm"] == arm]
            r = sub["feedback"].dropna().astype(float)
            pos = f"{(r > 0).mean():.0%}" if len(r) else "–"
            p = probs.get(arm)
            rows += (f'<tr><th scope="row">{_e(label)}</th><td class="mai-num">{"–" if p is None else f"{float(p):.0%}"}</td>'
                     f'<td class="mai-num">{len(sub)}</td><td class="mai-num">{len(r)}</td><td class="mai-num">{pos}</td></tr>')
    table = (f'<table class="mai-table"><thead><tr><th scope="col">Writing style</th><th scope="col" class="mai-num">Selection probability</th>'
             f'<th scope="col" class="mai-num">Requests</th><th scope="col" class="mai-num">Ratings</th><th scope="col" class="mai-num">Positive</th></tr></thead>'
             f'<tbody>{rows}</tbody></table>') if rows else '<p class="mai-subtle">No requests yet.</p>'
    body = f"""
<div class="mai-rl">
  <div>{table}
    <p class="mai-footnote">Counts from the logs. Selection probability is the one logged with the latest request (policy {_e(policy)}).</p></div>
  <div class="mai-chart-empty mai-chart-empty--dashed">{icon("target", 22)}<strong>Posterior and learning curve</strong>
    <span class="mai-subtle">Reserved for WP6: Thompson-sampling posterior per style (mean and credible interval), real vs
    prior observations and cumulative reward. Appears when the bandit learns from feedback.</span></div>
</div>"""
    return _card("Reinforcement learning · writing-style bandit", "target", body, hint="WP6 · in progress", cls="mai-rl-card")

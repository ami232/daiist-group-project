"""HTML views of a logged record (App tab). Presentation only.

Rules for everything in this file:
- Every value that comes from the record goes through `_e()` (html.escape) before
  it is put in HTML. That includes text written by the LLM (steps, justification,
  review reasons) and tool results.
- Numbers are only shown if the record contains them. Nothing is invented: the
  initial state is empty.
- Outputs of stub components carry a "Stub output" badge.
- The bandit's writing style is never shown here: it is revealed only after the
  user gives feedback (`feedback_thanks`).
"""

import html
import json

from maintai import economics
from ui import charts
from ui.icons import MARK, icon

# --- Labels -----------------------------------------------------------------------

FIELD_LABELS = {
    "type": "Machine type",
    "air_temperature_k": "Air temperature",
    "process_temperature_k": "Process temperature",
    "rotational_speed_rpm": "Rotational speed",
    "torque_nm": "Torque",
    "tool_wear_min": "Tool wear",
}

STUB_NAMES = {"component_a": "Risk model", "component_b": "Work-order agent", "bandit": "Strategy policy"}

BAND = {  # label, semantic tone
    "low": ("Low", "success"),
    "medium": ("Medium", "warning"),
    "high": ("High", "orange"),
    "critical": ("Critical", "danger"),
}

MODE = {"TWF": "Tool wear", "HDF": "Heat dissipation", "PWF": "Power", "OSF": "Overstrain",
        "RNF": "Random", "none": "No failure mode"}

PRIORITY = {"P1": "Act now", "P2": "This shift", "P3": "Next planned window", "P4": "Monitor only"}

ACTION = {"monitor": "Monitor", "schedule_inspection": "Schedule inspection", "replace_tool": "Replace tool",
          "cooling_check": "Cooling check", "power_check": "Power check", "reduce_load": "Reduce load",
          "stop_and_inspect": "Stop and inspect"}

REVIEW_REASONS = {
    "critical_risk": "Critical risk band",
    "out_of_training_range": "Reading outside the training range",
    "uncertain_prediction": "Prediction close to the decision threshold",
    "failed_validation": "The order failed a grounding check",
    "agent_requested": "The agent asked for a review",
    "part_out_of_stock": "A chosen spare part is out of stock",
}

OUT_OF_SCOPE = {
    "empty_input": "The request was empty.",
    "input_too_long": "The request is longer than a machine reading can be.",
    "not_a_machine_reading": "The request is not a machine reading.",
    "invalid_reading": "One or more readings are outside the physically plausible range.",
}

TOOL_LABELS = {"check_spare_parts": "Spare parts inventory", "find_maintenance_window": "Maintenance calendar",
               "estimate_costs": "Cost estimate", "get_maintenance_policy": "Maintenance policy"}

STYLE = {"checklist": "Checklist", "cost": "Cost-focused", "root_cause": "Root-cause"}

# Failure rules documented with the AI4I 2020 dataset (UCI Machine Learning Repository,
# "AI4I 2020 Predictive Maintenance Dataset"). Used only to draw reference zones on the
# indicator bars; the overstrain limit per type comes from the record itself (features.py).
AI4I_HDF_TEMP_DIFF_K = 8.6  # heat dissipation failure if the temperature difference is below this...
AI4I_HDF_RPM = 1380  # ...and the rotational speed is below this
AI4I_PWF_W = (3500, 9000)  # power failure outside this range
AI4I_TWF_MIN = (200, 240)  # tool wear failure or replacement in this window


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _badge(text: str, tone: str = "neutral", icon_name: str | None = None) -> str:
    ico = icon(icon_name, 14) if icon_name else ""
    return f'<span class="mai-badge mai-badge--{tone}">{ico}{_e(text)}</span>'


def _eur(value: float, decimals: int = 0) -> str:
    value = round(float(value), decimals)  # so a tiny negative amount shows as €0, not −€0
    sign = "−" if value < 0 else ""
    return f"{sign}€{abs(value):,.{decimals}f}"


def is_stub(version: str | None) -> bool:
    return "stub" in str(version or "")


def stub_parts(versions: dict) -> list[str]:
    return [STUB_NAMES.get(k, k) for k, v in versions.items() if k in STUB_NAMES and is_stub(v)]


# --- Page chrome ----------------------------------------------------------------------

def header(stubs: list[str]) -> str:
    env = (_badge("Development build", "stub-dark", "flask") if stubs else _badge("Live", "live-dark"))
    return f"""
<header class="mai-header" role="banner">
  <div class="mai-header__inner">
    <div class="mai-brand">
      {MARK}
      <span class="mai-brand__name">Maint<span>AI</span></span>
      <span class="mai-brand__divider" aria-hidden="true"></span>
      <span class="mai-brand__tagline">Intelligent Maintenance Advisor</span>
    </div>
    <div class="mai-header__meta">
      {_badge("Simulated plant data", "outline-dark", "database")}
      {env}
    </div>
  </div>
</header>"""


def dev_banner(stubs: list[str]) -> str:
    if not stubs:
        return ""
    parts = ", ".join(_e(s) for s in stubs)
    return f"""
<div class="mai-banner mai-banner--stub" role="note">
  {icon("flask", 18)}
  <div><strong>Development build.</strong> Provisional stubs: {parts}. Their outputs are placeholders, not real
  predictions, and are excluded from the evaluation.</div>
</div>"""


def missing_env_banner(missing: list[str]) -> str:
    if not missing:
        return ""
    return f"""
<div class="mai-banner mai-banner--warning" role="alert">
  {icon("alert", 18)}
  <div><strong>Missing environment variables:</strong> {_e(", ".join(missing))}. See SETUP.md.</div>
</div>"""


def page_title(title: str, subtitle: str, eyebrow: str) -> str:
    return f"""
<div class="mai-pagehead">
  <p class="mai-eyebrow">{_e(eyebrow)}</p>
  <h1 class="mai-h1">{_e(title)}</h1>
  <p class="mai-subtle">{_e(subtitle)}</p>
</div>"""


def section_title(title: str, icon_name: str, hint: str = "", right: str = "", level: int = 2) -> str:
    hint_html = f'<span class="mai-card__hint">{_e(hint)}</span>' if hint else ""
    return (f'<div class="mai-card__header"><span class="mai-card__icon">{icon(icon_name, 16)}</span>'
            f'<h{level} class="mai-h2">{_e(title)}</h{level}>{hint_html}<span class="mai-card__right">{right}</span></div>')


def footer(storage: str) -> str:
    return (f'<footer class="mai-footer"><span>MaintAI · DAIIST group project · IE University</span>'
            f'<span>{icon("database", 14)} Logs: {_e(storage)}</span>'
            f'<span>Plant systems simulated · Costs are business-case assumptions</span></footer>')


# --- Result states ----------------------------------------------------------------------

_SECTIONS = [("Risk intelligence", "pulse", "Failure probability, likely causes and physical indicators against documented limits"),
             ("Work order", "clipboard", "Drafted by the agent from plant tools, checked by deterministic safety rules"),
             ("Financial insight", "coins", "Expected cost of acting vs not acting, from the business-case assumptions"),
             ("Agent trace", "trace", "Every step from the reading to the order, with the tools the agent called")]


def empty_state() -> str:
    tiles = "".join(f'<li><span class="mai-empty__tile-icon">{icon(i, 18)}</span><div><strong>{_e(t)}</strong>'
                    f'<span>{_e(d)}</span></div></li>' for t, i, d in _SECTIONS)
    steps = "".join(f'<li><span class="mai-flow__n">{n}</span><span>{_e(s)}</span></li>' for n, s in
                    enumerate(["Enter or load a machine reading", "The risk model scores the process",
                               "The agent drafts a work order", "A supervisor approves it"], 1))
    return f"""
<div class="mai-card mai-empty">
  <div class="mai-empty__head">
    <div class="mai-empty__icon">{icon("monitor", 26)}</div>
    <div><h2 class="mai-h2">No analysis yet</h2>
    <p class="mai-subtle">Fill in the reading or load a sample, then select <strong>Analyse machine</strong>.
    Nothing is shown until a real analysis runs.</p></div>
  </div>
  <ol class="mai-flow">{steps}</ol>
  <ul class="mai-empty__grid">{tiles}</ul>
</div>"""


def loading_state() -> str:
    def block(title, i, lines=3):
        return f'<div class="mai-card">{section_title(title, i)}' + "".join(
            f'<div class="mai-skeleton{" mai-skeleton--lg" if k == 0 else ""}"></div>' for k in range(lines)) + "</div>"
    return f"""
<div class="mai-results" aria-busy="true" aria-live="polite">
  <div class="mai-status mai-status--loading" role="status">
    <span class="mai-spinner" aria-hidden="true"></span>
    <div><strong>Analysing the reading…</strong>
    <span class="mai-subtle">Risk model first, then the work-order agent (at most 3 AI calls).</span></div>
  </div>
  <div class="mai-card mai-summary mai-summary--loading">{"".join('<div><div class="mai-skeleton mai-skeleton--sm"></div><div class="mai-skeleton mai-skeleton--lg"></div></div>' for _ in range(4))}</div>
  <div class="mai-grid">{block("Risk intelligence", "pulse")}{block("Physical indicators", "sliders")}</div>
  {block("Work order", "clipboard", 4)}
</div>"""


def error_state(message: str) -> str:
    return f"""
<div class="mai-results" aria-live="assertive">
  <div class="mai-status mai-status--danger" role="alert">
    {icon("alert", 20)}
    <div><strong>The analysis could not be completed</strong>
    <span>{_e(message)}</span>
    <span>Nothing was approved or logged as a work order. Try again in a moment.</span></div>
  </div>
</div>"""


def out_of_scope(record: dict) -> str:
    refusal = record["outputs"]["component_a"]
    reason = refusal.get("reason", "")
    detail = str(refusal.get("detail", "") or "")
    items = ""
    if reason == "invalid_reading" and detail:
        rows = []
        for problem in detail.split("; "):
            field, _, msg = problem.partition(": ")
            rows.append(f"<li><strong>{_e(FIELD_LABELS.get(field, field))}</strong><span>{_e(msg or field)}</span></li>")
        items = f'<ul class="mai-checklist">{"".join(rows)}</ul>'
    elif detail:
        items = f'<p class="mai-body">{_e(detail)}</p>'
    return f"""
<div class="mai-results" aria-live="polite">
  <div class="mai-status mai-status--blocked" role="status">
    {icon("block", 20)}
    <div><strong>Out of scope · no model was called</strong>
    <span>{_e(OUT_OF_SCOPE.get(reason, "The request cannot be handled."))}</span></div>
    <div class="mai-status__meta">{_meta(record)}</div>
  </div>
  <div class="mai-card">
    {section_title("What to check", "info")}
    {items}
    <p class="mai-footnote">MaintAI only assesses single readings from milling machines within plausible physical
    ranges. The request was refused before any model call: no work order, no AI cost. It is still logged, so
    out-of-scope traffic is visible in Monitoring.</p>
  </div>
</div>"""


def _meta(record: dict) -> str:
    rid = str(record.get("id", ""))[:8]
    parts = [f'<span title="Request id">{icon("clipboard", 13)} #{_e(rid)}</span>' if rid else "",
             f'<span>{icon("pulse", 13)} {_e(_latency(record.get("latency_s")))}</span>',
             f'<span>{_e(record.get("tokens", 0))} tokens</span>',
             f'<span>AI cost {_e(_eur(record.get("cost_eur", 0.0), 4))}</span>']
    return "".join(p for p in parts if p)


# --- Full result --------------------------------------------------------------------------

def _economics(record: dict, risk: dict, order: dict) -> dict:
    """Expected cost per process for this decision, from the assumed cost model (economics.py)."""
    a = economics.DEFAULT
    p = float(risk.get("p_fail", 0.0))
    cost = order.get("cost") or {}
    c_i = float(cost.get("preventive_eur", a.intervention_eur))
    c_f = float(cost.get("breakdown_eur", a.failure_eur))
    acts = order["draft"].get("action") != "monitor"
    review = a.review_eur if order.get("review") else 0.0
    ai = float(record.get("cost_eur", 0.0))
    no_action = p * c_f
    residual = p * (1 - a.prevention_effectiveness) * c_f
    with_action = (c_i + residual if acts else no_action) + review + ai
    return {"p": p, "c_i": c_i, "c_f": c_f, "acts": acts, "review": review, "ai": ai, "e": a.prevention_effectiveness,
            "no_action": no_action, "residual": residual, "with_action": with_action, "saving": no_action - with_action,
            "downtime_h": cost.get("downtime_h"), "break_even": economics.break_even_probability(a)}


def results(record: dict) -> str:
    """The full result for a handled request."""
    risk = record["outputs"]["component_a"]
    order = record["outputs"]["component_b"]["work_order"]
    trace = record["outputs"]["component_b"]["tool_calls"]
    versions = record.get("versions", {})
    money = _economics(record, risk, order)
    stub_a, stub_b = is_stub(versions.get("component_a")), is_stub(versions.get("component_b"))
    return f"""
<div class="mai-results" aria-live="polite">
  {_decision(record, order)}
  {_summary(risk, order, money, stub_a or stub_b)}
  <div class="mai-grid">
    {_risk_card(risk, stub_a)}
    {_indicators_card(record, risk, stub_a)}
  </div>
  {_order_card(record, order, trace, stub_b)}
  {_finance_card(money, stub_a)}
  {_trace_card(record, risk, order, trace)}
</div>"""


def _decision(record: dict, order: dict) -> str:
    degraded = ""
    if record.get("degraded"):
        degraded = (f'<div class="mai-banner mai-banner--warning">{icon("alert", 16)}'
                    f'<div><strong>Degraded mode.</strong> The AI agent was not used, so a rule-based template wrote '
                    f'this order. Reason: {_e(record.get("degraded_reason", "") or "not given")}.</div></div>')
    if order.get("review"):
        reasons = "".join(f"<li>{_e(REVIEW_REASONS.get(r, r))}</li>" for r in order.get("review_reasons", []))
        head = (f'<div class="mai-status mai-status--warning" role="status">{icon("user-check", 20)}'
                f'<div><strong>Human review required before approval</strong>'
                f'<ul class="mai-inline-list">{reasons}</ul></div>')
    else:
        head = (f'<div class="mai-status mai-status--success" role="status">{icon("check", 20)}'
                f'<div><strong>Ready for supervisor approval</strong>'
                f'<span>No safety rule requires an additional review.</span></div>')
    return f'{head}<div class="mai-status__meta">{_meta(record)}</div></div>{degraded}'


def _summary(risk: dict, order: dict, money: dict, stub: bool) -> str:
    label, tone = BAND.get(risk.get("risk_band"), ("Unknown", "neutral"))
    d = order["draft"]
    prio = d.get("priority", "")
    saving_tone = "success" if money["saving"] > 0 else "neutral"
    stub_badge = _badge("Provisional · stub components", "stub", "flask") if stub else ""
    review = ("Required", "warning", "user-check") if order.get("review") else ("Not required", "success", "check")
    return f"""
<section class="mai-card mai-summary-card" aria-label="Executive summary">
  <div class="mai-summary__head"><p class="mai-eyebrow">Executive summary</p>{stub_badge}</div>
  <div class="mai-summary">
    <div class="mai-summary__item"><span class="mai-summary__label">{icon("pulse", 14)} Failure risk</span>
      <span class="mai-summary__value">{money["p"]:.1%}</span>{_badge(label + " risk", tone)}</div>
    <div class="mai-summary__item"><span class="mai-summary__label">{icon("wrench", 14)} Recommended action</span>
      <span class="mai-summary__value mai-summary__value--text">{_e(ACTION.get(d.get("action"), d.get("action")))}</span>
      <span class="mai-subtle">{_e(MODE.get(d.get("failure_mode_addressed"), ""))}</span></div>
    <div class="mai-summary__item"><span class="mai-summary__label">{icon("clock", 14)} Priority</span>
      <span class="mai-summary__value"><span class="mai-prio mai-prio--{_e(prio.lower())}">{_e(prio)}</span></span>
      <span class="mai-subtle">{_e(PRIORITY.get(prio, ""))}</span></div>
    <div class="mai-summary__item"><span class="mai-summary__label">{icon("coins", 14)} Expected net saving</span>
      <span class="mai-summary__value mai-tone--{saving_tone}">{_eur(money["saving"])}</span>
      <span class="mai-subtle">per process · assumption-based</span></div>
    <div class="mai-summary__item"><span class="mai-summary__label">{icon(review[2], 14)} Human review</span>
      <span class="mai-summary__value mai-summary__value--text">{_e(review[0])}</span>
      <span class="mai-subtle">{len(order.get("review_reasons", []))} rule(s) triggered</span></div>
  </div>
</section>"""


def _risk_card(risk: dict, stub: bool) -> str:
    label, tone = BAND.get(risk.get("risk_band"), ("Unknown", "neutral"))
    p = float(risk.get("p_fail", 0.0))
    threshold = float(risk.get("threshold", 0))
    modes = risk.get("likely_modes", [])
    mode_rows = [(MODE.get(m["mode"], m["mode"]), float(m["p"]), f'{float(m["p"]):.0%}', "info") for m in modes]
    badges = (_badge("Stub output", "stub", "flask") if stub else "") + \
             (_badge("Near threshold", "warning", "info") if risk.get("in_uncertainty_band") else "") + \
             "".join(_badge(f, "warning", "alert") for f in risk.get("ood_flags", []))
    position = "above" if p >= threshold else "below"
    break_even = economics.break_even_probability(economics.DEFAULT)
    return f"""
<section class="mai-card" aria-labelledby="mai-risk-title">
  {section_title("Risk intelligence", "pulse", right=badges).replace('<h2 class="mai-h2">', '<h2 class="mai-h2" id="mai-risk-title">')}
  <div class="mai-risk">
    {charts.gauge(p, threshold, tone, label)}
    <div class="mai-risk__side">
      <div class="mai-risk__band">{_badge(label + " risk", tone)}</div>
      <p class="mai-body">The probability is <strong>{position}</strong> the decision threshold of
      <strong>{threshold:.1%}</strong>, chosen on validation data to minimise the expected cost, reviews included.
      It is not the economic break-even ({break_even:.1%} in the simplified cost model).</p>
      <p class="mai-footnote">Model {_e(risk.get("model_version", ""))}</p>
    </div>
  </div>
  <h3 class="mai-h3">Failure-mode probabilities</h3>
  {charts.bar_rows(mode_rows) if mode_rows else '<p class="mai-subtle">No failure mode reported.</p>'}
  {'<p class="mai-footnote">Each bar comes from its own model: the probability that this failure mode occurs. They are '
   'independent, not shares of one total, so they need not add up to 100%. Random failures (RNF) cannot be predicted '
   'and are not shown.</p>' if mode_rows else ''}
</section>"""


def _indicators_card(record: dict, risk: dict, stub: bool) -> str:
    ind = risk.get("indicators", {})
    rpm = float(record.get("features", {}).get("rpm", 0) or 0)
    diff = float(ind.get("temp_diff_k", 0))
    power = float(ind.get("power_w", 0))
    strain, limit = float(ind.get("strain_min_nm", 0)), float(ind.get("strain_limit_min_nm", 1) or 1)
    wear = float(ind.get("tool_wear_min", 0))
    lo_w, hi_w = AI4I_PWF_W
    bars = [
        charts.limit_bar("Temperature difference", diff, f"{diff:.1f} K", (0, 16), [(0, AI4I_HDF_TEMP_DIFF_K)],
                         diff < AI4I_HDF_TEMP_DIFF_K and rpm < AI4I_HDF_RPM,
                         f"Heat-dissipation risk below {AI4I_HDF_TEMP_DIFF_K} K at under {AI4I_HDF_RPM} rpm (now {rpm:.0f} rpm)"),
        charts.limit_bar("Mechanical power", power / 1000, f"{power / 1000:.2f} kW", (0, 12), [(0, lo_w / 1000), (hi_w / 1000, 12)],
                         not lo_w <= power <= hi_w, f"Power-failure risk outside {lo_w / 1000:.1f}–{hi_w / 1000:.1f} kW"),
        charts.limit_bar("Overstrain (wear × torque)", strain, f"{strain:,.0f} min·Nm", (0, max(limit * 1.4, strain * 1.05)),
                         [(limit, max(limit * 1.4, strain * 1.05))], strain > limit,
                         f"Limit {limit:,.0f} min·Nm for this machine type · {strain / limit:.0%} of limit"),
        charts.limit_bar("Tool wear", wear, f"{wear:.0f} min", (0, 260), [AI4I_TWF_MIN], AI4I_TWF_MIN[0] <= wear <= AI4I_TWF_MIN[1],
                         f"Tool-wear failure window {AI4I_TWF_MIN[0]}–{AI4I_TWF_MIN[1]} min"),
    ]
    return f"""
<section class="mai-card" aria-label="Physical indicators">
  {section_title("Physical indicators", "sliders", right=_badge("AI4I dataset rules", "neutral", "info"))}
  <div class="mai-limits">{"".join(bars)}</div>
  <p class="mai-footnote">Values are computed from the reading (not a model output). Shaded zones are the failure rules of the
  synthetic AI4I 2020 dataset (UCI) the model is trained on: dataset rules, not universal industrial limits or
  manufacturer specifications.</p>
</section>"""


def _tool_results(trace: list, name: str) -> list:
    out = []
    for call in trace:
        if call.get("name") == name:
            result = call.get("result")
            out.extend(result if isinstance(result, list) else [result])
    return out


def _order_card(record: dict, order: dict, trace: list, stub: bool) -> str:
    d = order["draft"]
    prio = d.get("priority", "")
    steps = "".join(f'<li><span class="mai-steps__n">{n}</span><span>{_e(s)}</span></li>' for n, s in enumerate(d.get("steps", []), 1))
    parts_info = {p.get("part_id"): p for p in _tool_results(trace, "check_spare_parts") if isinstance(p, dict)}
    windows = {w.get("window_id"): w for w in _tool_results(trace, "find_maintenance_window") if isinstance(w, dict)}
    policies = [p for p in _tool_results(trace, "get_maintenance_policy") if isinstance(p, str)]

    parts_html = ""
    for pid in d.get("part_ids", []):
        p = parts_info.get(pid, {})
        stock = p.get("in_stock")
        stock_badge = _badge(f"{stock} in stock" if stock else "Out of stock", "success" if stock else "danger") if stock is not None else ""
        lead = f'Lead time {p["lead_time_days"]} d' if p.get("lead_time_days") is not None else ""
        parts_html += (f'<li><div><strong>{_e(p.get("name", pid))}</strong><span class="mai-mono">{_e(pid)}</span></div>'
                       f'<div class="mai-part__meta">{stock_badge}<span class="mai-subtle">{_e(lead)}</span></div></li>')
    window = windows.get(d.get("window_id"))
    if window:
        start = str(window.get("start", "")).replace("T", " · ")
        window_html = (f'<strong>{_e(start)}</strong><span class="mai-subtle">{_e(window.get("duration_h", ""))} h slot · '
                       f'<span class="mai-mono">{_e(d.get("window_id"))}</span></span>')
    else:
        window_html = f'<strong>{_e(d.get("window_id") or "Not scheduled")}</strong>'
    downtime = (order.get("cost") or {}).get("downtime_h")
    policy_html = "".join(f"<li>{icon('info', 14)}<span>{_e(p)}</span></li>" for p in policies)
    errors = "".join(f"<li>{_e(e)}</li>" for e in order.get("validation_errors", []))
    errors_html = (f'<div class="mai-banner mai-banner--danger">{icon("alert", 16)}'
                   f'<div><strong>Validation errors</strong><ul class="mai-list">{errors}</ul></div></div>') if errors else ""
    status = (_badge("Human review required", "warning", "user-check") if order.get("review")
              else _badge("Awaiting supervisor approval", "info", "clock"))
    stub_badge = _badge("Stub output", "stub", "flask") if stub else ""
    wo_number = "WO-" + str(record.get("id", ""))[:8].upper()
    return f"""
<section class="mai-card mai-order" aria-label="Work order">
  <div class="mai-order__band mai-order__band--{_e(prio.lower())}">
    <span class="mai-prio mai-prio--lg mai-prio--{_e(prio.lower())}">{_e(prio)}</span>
    <div class="mai-order__title">
      <span class="mai-eyebrow">Work order <span class="mai-mono">{_e(wo_number)}</span> · draft</span>
      <h2 class="mai-h2 mai-order__action">{_e(ACTION.get(d.get("action"), d.get("action")))}</h2>
      <span class="mai-subtle">{_e(PRIORITY.get(prio, ""))} · {_e("No failure mode targeted" if d.get("failure_mode_addressed") == "none" else "Addresses " + MODE.get(d.get("failure_mode_addressed"), ""))}</span>
    </div>
    <div class="mai-order__status">{status}{stub_badge}</div>
  </div>
  <div class="mai-order__body">
    <div>
      <h3 class="mai-h3">Procedure</h3>
      <ol class="mai-steps">{steps}</ol>
      <h3 class="mai-h3">Justification</h3>
      <blockquote class="mai-quote">{_e(d.get("justification", ""))}</blockquote>
      {errors_html}
    </div>
    <aside class="mai-order__aside" aria-label="Resources">
      <div class="mai-detail"><span class="mai-detail__label">{icon("clock", 14)} Maintenance window</span>{window_html}</div>
      <div class="mai-detail"><span class="mai-detail__label">{icon("wrench", 14)} Estimated downtime</span>
        <strong>{f"{float(downtime):.1f} h" if downtime else "Not estimated"}</strong></div>
      <div class="mai-detail"><span class="mai-detail__label">{icon("database", 14)} Spare parts</span>
        <ul class="mai-parts">{parts_html or '<li class="mai-subtle">No parts needed</li>'}</ul></div>
      {f'<div class="mai-detail"><span class="mai-detail__label">{icon("info", 14)} Policy</span><ul class="mai-policy">{policy_html}</ul></div>' if policy_html else ""}
    </aside>
  </div>
  <div class="mai-order__foot">{icon("user-check", 15)}<span>Draft only. A supervisor approves every work order before any action.
  Parts, windows and costs come from plant tools (simulated), never from the language model.</span></div>
</section>"""


def _finance_card(m: dict, stub: bool) -> str:
    if m["acts"]:
        rows = [("No action", [("Expected breakdown loss", m["no_action"], "danger")]),
                ("Recommended action", [("Intervention", m["c_i"], "brand"), ("Residual breakdown risk", m["residual"], "orange"),
                                        ("Human review", m["review"], "info"), ("AI cost", m["ai"], "stub")])]
    else:
        rows = [("No action (monitor)", [("Expected breakdown loss", m["no_action"], "danger"),
                                         ("Human review", m["review"], "info"), ("AI cost", m["ai"], "stub")])]
    table = [
        ("Intervention", "Assumed", _eur(m["c_i"]) if m["acts"] else "Not proposed"),
        ("Breakdown if it fails", "Assumed", _eur(m["c_f"])),
        ("Prevention effectiveness", "Assumed", f'{m["e"]:.0%}'),
        ("Human review", "Assumed", _eur(m["review"]) if m["review"] else "Not required"),
        ("AI cost of this request", "From tokens", _eur(m["ai"], 4)),
    ]
    trs = "".join(f'<tr><td>{_e(a)}</td><td><span class="mai-tag mai-tag--{"assumed" if b == "Assumed" else "measured"}">{_e(b)}</span></td>'
                  f'<td class="mai-num">{_e(c)}</td></tr>' for a, b, c in table)
    saving_tone = "success" if m["saving"] > 0 else "neutral"
    stub_note = " The probability comes from a stub model, so these figures are placeholders." if stub else ""
    return f"""
<section class="mai-card" aria-label="Financial insight">
  {section_title("Financial insight", "coins", right=_badge("Assumption-based estimate", "info", "info"))}
  <div class="mai-kpi-row mai-kpi-row--3">
    <div class="mai-kpi"><span class="mai-kpi__label">Expected cost without action</span>
      <span class="mai-kpi__value">{_eur(m["no_action"])}</span><span class="mai-kpi__foot">p × breakdown cost</span></div>
    <div class="mai-kpi"><span class="mai-kpi__label">Expected cost with the order</span>
      <span class="mai-kpi__value">{_eur(m["with_action"])}</span><span class="mai-kpi__foot">Intervention, residual risk, review and AI</span></div>
    <div class="mai-kpi mai-kpi--{saving_tone}"><span class="mai-kpi__label">Expected net saving</span>
      <span class="mai-kpi__value">{_eur(m["saving"])}</span><span class="mai-kpi__foot">Per process, on average</span></div>
  </div>
  <div class="mai-finance">
    <div><h3 class="mai-h3">Expected cost per process</h3>{charts.stacked_costs(rows, _eur)}</div>
    <div><h3 class="mai-h3">Inputs</h3>
      <table class="mai-table"><caption class="mai-sr">Cost inputs</caption>
      <thead><tr><th scope="col">Item</th><th scope="col">Source</th><th scope="col" class="mai-num">Value</th></tr></thead>
      <tbody>{trs}</tbody></table></div>
  </div>
  <p class="mai-footnote">Expected values, not guaranteed savings. Costs are business-case assumptions (REPORT.md), not plant
  data. Intervening pays off on average above p* = {m["break_even"]:.1%} (simplified model, without review or AI costs).{_e(stub_note)}</p>
</section>"""


def _summarise_result(result) -> str:
    items = result if isinstance(result, list) else [result]
    if not items:
        return "No results"
    first = items[0]
    if isinstance(first, dict):
        if "name" in first:
            return ", ".join(str(i.get("name", "")) for i in items if isinstance(i, dict))
        if "window_id" in first:
            return f'{len(items)} window(s), first {str(first.get("start", "")).replace("T", " ")}'
        if "preventive_eur" in first:
            return f'Preventive {_eur(float(first["preventive_eur"]))}, breakdown {_eur(float(first.get("breakdown_eur", 0)))}'
    if isinstance(first, str):
        return f"{len(items)} rule(s)"
    return f"{len(items)} result(s)"


def _trace_card(record: dict, risk: dict, order: dict, trace: list) -> str:
    llm = record.get("llm") or {}
    label, tone = BAND.get(risk.get("risk_band"), ("Unknown", "neutral"))

    def node(icon_name, kind, title, detail, extra="", state="done"):
        return (f'<li class="mai-tl__item mai-tl__item--{state}"><span class="mai-tl__dot">{icon(icon_name, 14)}</span>'
                f'<div class="mai-tl__body"><span class="mai-tl__kind">{_e(kind)}</span><strong>{title}</strong>'
                f'<span class="mai-subtle">{detail}</span>{extra}</div></li>')

    items = [node("sliders", "Input", "Reading validated", _e(record.get("input_summary", "")))]
    items.append(node("pulse", "Component A · ML model", f'Failure probability {float(risk.get("p_fail", 0)):.1%} · {_e(label)}',
                      f'Model {_e(risk.get("model_version", ""))}'))
    for call in trace:
        args = ", ".join(f"{_e(k)}={_e(v)}" for k, v in (call.get("arguments") or {}).items())
        body = _e(json.dumps(call.get("result"), indent=2, ensure_ascii=False))
        extra = (f'<details class="mai-tl__details"><summary>Arguments and raw result</summary>'
                 f'<p class="mai-mono">{call.get("name", "") and _e(call["name"])}({args})</p><pre class="mai-pre">{body}</pre></details>')
        items.append(node("wrench", "Tool call", _e(TOOL_LABELS.get(call.get("name"), call.get("name", ""))),
                          _e(_summarise_result(call.get("result"))), extra))
    if order.get("validation_errors"):
        items.append(node("alert", "Guardrails", "Grounding check failed", _e("; ".join(order["validation_errors"])), state="warn"))
    else:
        reasons = ", ".join(REVIEW_REASONS.get(r, r) for r in order.get("review_reasons", [])) or "none triggered"
        items.append(node("check", "Guardrails", "Safety rules applied",
                          f"Grounding check passed (no part or window id outside the tool results) · Review rules: {_e(reasons)}",
                          state="warn" if order.get("review") else "done"))
    model = llm.get("model") or "Rule-based template (no AI call)"
    usage = f'{_e(llm.get("llm_calls", 0))} of 3 AI calls · {_e(record.get("tokens", 0))} tokens · cache {"hit" if llm.get("cache_hit") else "not used"}'
    items.append(node("clipboard", "Component B · agent", f"Work order drafted by {_e(model)}", usage,
                      state="warn" if record.get("degraded") else "done"))
    return f"""
<section class="mai-card" aria-label="Agent trace">
  {section_title("Agent trace", "trace", hint=f"{len(trace)} tool call(s)")}
  <ol class="mai-tl">{"".join(items)}</ol>
</section>"""


# --- Feedback --------------------------------------------------------------------------

def feedback_thanks(value: int, arm: str) -> str:
    """Shown only after the vote, so the writing style cannot bias the feedback."""
    verdict = "useful" if value > 0 else "not useful"
    style = STYLE.get(arm, arm) if arm else ""
    reveal = (f' This order was written with the <strong>{_e(style)}</strong> style; your rating helps MaintAI learn '
              f'which style maintenance teams prefer.') if style else ""
    return (f'<div class="mai-feedback__thanks" role="status">{icon("check", 16)}'
            f'<span>Recorded as {_e(verdict)}.{reveal}</span></div>')


def _latency(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "–"
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.1f} s"

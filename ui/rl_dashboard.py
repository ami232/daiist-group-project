"""Adaptive Intelligence: the reinforcement-learning section of Monitoring (WP6, presentation).

Two parts that never mix:
- Live policy: everything comes from bandit.policy_snapshot(), policy_history(),
  maintai/rl_monitor.py and the logged requests. If no real rating is eligible yet,
  the learning charts show an empty state; they are never filled with simulated data.
- Simulation results: eval/bandit_simulation*.{csv,json} and
  eval/bandit_calibration*.{csv,json}, shown in a separate, clearly labelled section.

Every card carries a provenance badge: Real logs (measured from the logged
requests), Simulated, Assumption (a threshold chosen by the team) or Pending
(not enough data yet). Missing values are shown as "N/A", never as 0.
"""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from maintai import bandit, rl_monitor
from maintai.schemas import ARMS
from ui import charts
from ui.icons import icon

ROOT = Path(__file__).resolve().parent.parent
SIM_CSV = ROOT / "eval" / "bandit_simulation.csv"
SIM_JSON = ROOT / "eval" / "bandit_simulation_summary.json"
CAL_CSV = ROOT / "eval" / "bandit_calibration.csv"
CAL_JSON = ROOT / "eval" / "bandit_calibration_summary.json"
NA = "N/A"
PROVENANCE = {"real": ("success", "check", "Real logs"), "simulated": ("stub", "flask", "Simulated"),
              "assumption": ("warning", "sliders", "Assumption"), "pending": ("neutral", "clock", "Pending")}
BAND_LABEL = {"low": "Low risk", "medium": "Medium risk", "high": "High risk", "critical": "Critical risk", "unknown": "Unknown"}
DIM_LABEL = {"risk_band": "Risk band", "review": "Decision", "arm": "Style"}

ARM_LABEL = {"checklist": "Checklist", "cost": "Cost-focused", "root_cause": "Root-cause"}
ARM_TONE = {"checklist": "info", "cost": "orange", "root_cause": "brand"}
ARM_HINT = {"checklist": "Short numbered steps", "cost": "Opens with cost and window", "root_cause": "Explains the indicators"}


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _card(title: str, icon_name: str, body: str, hint: str = "", cls: str = "", right: str = "") -> str:
    hint_html = f'<span class="mai-card__hint">{_e(hint)}</span>' if hint else ""
    return (f'<section class="mai-card {cls}"><div class="mai-card__header"><span class="mai-card__icon">{icon(icon_name, 16)}</span>'
            f'<h3 class="mai-h2">{_e(title)}</h3>{hint_html}<span class="mai-card__right">{right}</span></div>{body}</section>')


def _prov(kind: str) -> str:
    tone, ico, label = PROVENANCE[kind]
    return f'<span class="mai-badge mai-badge--{tone}">{icon(ico, 14)}{_e(label)}</span>'


def _pct(value, digits: int = 0) -> str:
    return NA if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}%}"


def _secs(value) -> str:
    if value is None:
        return NA
    return f"{value:.1f} s" if value < 60 else (f"{value / 60:.1f} min" if value < 3600 else f"{value / 3600:.1f} h")


def _note(text: str) -> str:
    return f'<p class="mai-footnote">{text}</p>'


def _empty(icon_name: str, title: str, text: str) -> str:
    return f'<div class="mai-chart-empty">{icon(icon_name, 22)}<strong>{_e(title)}</strong><span class="mai-subtle">{text}</span></div>'


def _kpi(icon_name: str, label: str, value: str, foot: str) -> str:
    return (f'<div class="mai-kpi mai-kpi--tile"><span class="mai-kpi__icon">{icon(icon_name, 18)}</span>'
            f'<span class="mai-kpi__label">{_e(label)}</span><span class="mai-kpi__value">{_e(value)}</span>'
            f'<span class="mai-kpi__foot">{_e(foot)}</span></div>')


def exclusions(live: pd.DataFrame, prompt_version: str, logs: pd.DataFrame | None = None) -> list[tuple[str, int]]:
    """Why live requests do not train the policy (a request can have several reasons)."""
    if live.empty:
        return []
    v = live.get("versions")
    fb = pd.to_numeric(live.get("feedback"), errors="coerce") if "feedback" in live else pd.Series(np.nan, index=live.index)
    over_cap = rl_monitor.sessions(logs, prompt_version)["n_over_cap"] if logs is not None else 0
    return [
        ("Stub component versions (development)", int(v.apply(bandit._has_stub).sum())),
        ("Degraded mode (template order, no agent)", int(live.get("degraded", pd.Series(False, index=live.index)).fillna(False).astype(bool).sum())),
        ("Out of scope", int((live.get("status", pd.Series("ok", index=live.index)) == "out_of_scope").sum())),
        ("Not served as real traffic (source ≠ real)", int((live.get("source", pd.Series("", index=live.index)) != "real").sum())),
        (f"Different prompt version (current: {prompt_version})", int((v.apply(bandit._prompt_version) != prompt_version).sum())),
        ("No rating given", int((~fb.isin([1, -1])).sum())),
        (f"Over the session cap ({bandit.MAX_RATINGS_PER_SESSION} rated requests per session)", int(over_cap)),
    ]


def render(logs: pd.DataFrame | None) -> str:
    logs = logs if logs is not None else pd.DataFrame()
    prompt_version = bandit.current_prompt_version()
    snap = bandit.policy_snapshot(logs)
    fb = bandit.eligible_feedback(logs)
    try:
        events = bandit.load_feedback_events()  # guarded already; this is a second safety net
    except Exception:  # noqa: BLE001 - the history chart then shows its empty state
        events = pd.DataFrame(columns=["id", "timestamp", "feedback"])
        events.attrs["error"] = "feedback events unreadable"
    hist = bandit.policy_history(logs, events)
    live = rl_monitor.live_requests(logs)
    n_real = int(snap["n_real"].sum())
    learning = n_real > 0

    status = (f'<span class="mai-badge mai-badge--success">{icon("check", 14)}Learning from {n_real} real rating(s)</span>' if learning
              else f'<span class="mai-badge mai-badge--warning">{icon("clock", 14)}Waiting for eligible feedback</span>')
    head = f"""
<div class="mai-rl-head">
  <div><p class="mai-eyebrow">Adaptive intelligence</p>
    <h2 class="mai-h1 mai-h1--sm">Writing-style bandit</h2>
    <p class="mai-subtle">Thompson sampling learns which work-order style users rate as useful. It only changes how an
    order is written, never the risk, action, priority or review rules.</p></div>
  <div class="mai-rl-head__meta">{status}
    <span class="mai-badge mai-badge--neutral">Policy {_e(bandit.POLICY_VERSION)}</span>
    <span class="mai-badge mai-badge--neutral">Prompt {_e(prompt_version)}</span>
    <span class="mai-badge mai-badge--neutral">Prior: neutral Beta(1, 1)</span></div>
</div>
<div class="mai-provenance" role="note"><span class="mai-provenance__title">Data labels</span>
  <span>{_prov("real")} measured from the logged requests</span><span>{_prov("simulated")} simulated users, never fed to the policy</span>
  <span>{_prov("assumption")} threshold chosen by the team</span><span>{_prov("pending")} not enough data yet</span></div>"""

    waiting = "" if learning else f"""
<div class="mai-banner mai-banner--info" role="status">{icon("info", 18)}<div><strong>No eligible real feedback yet.</strong>
  The live policy is still the neutral prior, so every style has the same chance. Ratings start to count once the real
  risk model (WP1) and the real agent with one prompt per style (WP3) are deployed: ratings of stub, degraded or
  out-of-scope requests never train the policy. Simulation results are shown separately below.</div></div>"""

    # KPIs
    explore = live["bandit_explore"].dropna() if "bandit_explore" in live else pd.Series(dtype=object)
    fallback = int((live.get("bandit_policy_from", pd.Series(dtype=object)) == "prior_only").sum()) if not live.empty else 0
    kpis = "".join([
        _kpi("thumb-up", "Eligible real ratings", f"{n_real}", f"{len(live)} live decision(s) logged"),
        _kpi("target", "Mean reward (real)", f"{fb['reward'].mean():.0%}" if learning else NA, "Share of eligible ratings that are positive"),
        _kpi("drift", "Exploration rate", f"{explore.astype(bool).mean():.0%}" if len(explore) else NA,
             f"{len(explore)} decision(s) with a single greedy style"),
        _kpi("alert", "Prior-only decisions", f"{fallback}", "Logs unreadable: chose from the prior"),
    ])

    # Feedback quality KPIs (rl_monitor)
    rr = rl_monitor.response_rate(logs, prompt_version)
    delay = rl_monitor.reward_delay(logs, events, prompt_version)
    sess = rl_monitor.sessions(logs, prompt_version)
    delay_foot = (f"p95 {_secs(delay['p95_s'])} · {delay['n']} first rating(s)" if delay["n"]
                  else ("Rating times not available" if events.attrs.get("error") else "No eligible rating yet"))
    kpis2 = "".join([
        _kpi("pulse", "Response rate", _pct(rr["rate"]), f"{rr['n_rated']} of {rr['n_eligible']} feedback-eligible request(s) rated"),
        _kpi("clock", "Reward delay (median)", _secs(delay["median_s"]), delay_foot),
        _kpi("user-check", "Rating sessions", f"{sess['n_sessions']}" if sess["n_rated"] else NA,
             f"{sess['n_over_cap']} rating(s) over the cap of {sess['cap']}"),
        _kpi("alert", "Changed / very fast ratings", f"{delay['n_changed']} / {delay['n_fast']}" if delay["n"] else NA,
             f"Very fast: under {rl_monitor.FAST_RATING_S:g} s after the order (assumption)"),
    ])

    cycle_steps = [("thumb-up", "Feedback", "A user rates the order, blind to its style"),
                   ("check", "Reward", "Useful = 1, not useful = 0; latest rating per request"),
                   ("pulse", "Policy update", "Beta(α, β) += reward, rebuilt from the logs"),
                   ("target", "Next selection", "Sample each Beta, highest sample wins")]
    cycle = "".join(f'<li><span class="mai-cycle__icon">{icon(i, 18)}</span><strong>{_e(t)}</strong><span>{_e(d)}</span></li>'
                    for i, t, d in cycle_steps)
    cycle_html = f'<ol class="mai-cycle">{cycle}</ol>'

    # Selection probabilities and credible intervals
    sel = charts.bar_rows([(ARM_LABEL[r.arm], r.p_select, f"{r.p_select:.0%}", ARM_TONE[r.arm]) for r in snap.itertuples()])
    forest = "".join(f"""
<div class="mai-forest__row"><span class="mai-forest__name">{_e(ARM_LABEL[r.arm])}</span>
  <div class="mai-forest__track" role="img" aria-label="{_e(ARM_LABEL[r.arm])}: estimated reward {r.mean:.0%}, 95% interval {r.ci_low:.0%} to {r.ci_high:.0%}">
    <span class="mai-forest__ci" style="left:{r.ci_low * 100:.1f}%;width:{(r.ci_high - r.ci_low) * 100:.1f}%;background:{charts.TONE_COLOR[ARM_TONE[r.arm]]}"></span>
    <span class="mai-forest__mean" style="left:{r.mean * 100:.1f}%;border-color:{charts.TONE_COLOR[ARM_TONE[r.arm]]}"></span></div>
  <span class="mai-forest__val">{r.mean:.0%} <small>[{r.ci_low:.0%}–{r.ci_high:.0%}]</small></span></div>""" for r in snap.itertuples())
    forest_html = f'<div class="mai-forest">{forest}<div class="mai-forest__axis"><span>0%</span><span>50%</span><span>100%</span></div></div>'

    # Beta densities
    from scipy.stats import beta as beta_dist

    xs = np.linspace(0.005, 0.995, 100)
    dens = {ARM_LABEL[r.arm]: [float(v) for v in beta_dist.pdf(xs, r.alpha, r.beta)] for r in snap.itertuples()}
    top = max(max(v) for v in dens.values())
    beta_chart = charts.line_chart(dens, [f"{x:.0%}" for x in xs], y_max=top * 1.15 or 1, y_fmt=lambda v: f"{v:.1f}",
                                   tones=[ARM_TONE[a] for a in ARMS], markers=False, max_xlabels=5,
                                   title="Posterior density of the reward rate per style")

    # Policy evolution (real ratings only)
    if learning and hist["step"].max() > 0:
        wide = hist.pivot_table(index="step", columns="arm", values="p_select").sort_index()
        evo = charts.line_chart({ARM_LABEL[a]: [float(v) for v in wide[a]] for a in ARMS if a in wide},
                                [str(int(s)) for s in wide.index], y_max=1.0, tones=[ARM_TONE[a] for a in ARMS],
                                markers=len(wide) < 25, title="Selection probability after each real rating")
        evo += _note("X axis: number of real rating events, in the order they were given. A changed rating replaces the earlier one.")
    else:
        evo = _empty("pulse", "No real learning curve yet", "Appears after the first eligible real rating. It is never filled with simulated data.")

    # Selected styles
    if not live.empty:
        counts = live["arm"].value_counts()
        selected = charts.segmented_bar([(ARM_LABEL[a], int(counts.get(a, 0)), ARM_TONE[a]) for a in ARMS])
        dev = int(live["versions"].apply(bandit._has_stub).sum())
        selected += _note(f"All live decisions by policy {_e(bandit.POLICY_VERSION)}. {dev} of {len(live)} were development "
                          "requests with stub components: real selections, but their ratings do not train the policy.")
    else:
        selected = _empty("requests", "No live decisions yet", "Each request served in the App tab adds one decision.")

    rows = "".join(f'<tr><th scope="row"><span class="mai-dot" style="background:{charts.TONE_COLOR[ARM_TONE[r.arm]]}"></span>{_e(ARM_LABEL[r.arm])}'
                   f'<small>{_e(ARM_HINT[r.arm])}</small></th><td class="mai-num">{r.alpha:g}</td><td class="mai-num">{r.beta:g}</td>'
                   f'<td class="mai-num">{r.successes}</td><td class="mai-num">{r.failures}</td>'
                   f'<td class="mai-num">{"–" if not r.n_real else f"{r.successes / r.n_real:.0%}"}</td>'
                   f'<td class="mai-num">{r.mean:.0%}</td><td class="mai-num">{r.p_select:.0%}</td></tr>' for r in snap.itertuples())
    table = (f'<table class="mai-table mai-table--rl"><thead><tr><th scope="col">Style</th><th scope="col" class="mai-num">α</th>'
             f'<th scope="col" class="mai-num">β</th><th scope="col" class="mai-num">Positive</th><th scope="col" class="mai-num">Negative</th>'
             f'<th scope="col" class="mai-num">Observed reward</th><th scope="col" class="mai-num">Posterior mean</th>'
             f'<th scope="col" class="mai-num">P(selected)</th></tr></thead><tbody>{rows}</tbody></table>')
    table += _note("α = 1 + positive real ratings, β = 1 + negative real ratings (neutral prior, no pseudo-observations). "
                   "Observed reward uses real ratings only; the posterior mean also includes the prior.")

    excl = exclusions(live, prompt_version, logs)
    excl_html = "".join(f'<li><span>{_e(t)}</span><strong>{n}</strong></li>' for t, n in excl) or '<li><span>No live decisions yet</span></li>'
    funnel = (f'<ul class="mai-kv mai-kv--compact">{excl_html}</ul>'
              + _note("A rating trains the policy only if the request was real traffic, handled, not degraded, free of stub "
                      "versions and written with the current prompt version. Only the latest rating per request counts."))

    science = _science_cards(logs, prompt_version, learning)

    live_section = f"""
{head}{waiting}
<div class="mai-kpi-row mai-kpi-row--4">{kpis}</div>
<div class="mai-kpi-row mai-kpi-row--4">{kpis2}</div>
{_card("Learning cycle", "trace", cycle_html, hint="Rebuilt on every request")}
<div class="mai-grid">
  {_card("Selection probability", "target", sel + _note("Probability that each style is chosen for the next request (4,000 posterior draws). Sums to 100%."), right=_prov("real" if learning else "pending"))}
  {_card("Estimated reward and uncertainty", "sliders", forest_html + _note("Posterior mean of the share of positive ratings, with its 95% credible interval. Wide bars mean little evidence. Simulated calibration of these intervals: see the simulation section."), right=_prov("real" if learning else "pending"))}
</div>
<div class="mai-grid">
  {_card("Posterior distributions", "drift", beta_chart + _note("Beta(α, β) density of each style's reward rate. Flat means no evidence; a narrow peak means a confident estimate."), right=_prov("real" if learning else "pending"))}
  {_card("Policy evolution (real feedback)", "pulse", evo, right=_prov("real" if learning else "pending"))}
</div>
{_card("Strategy table", "clipboard", table, right=_prov("real" if learning else "pending"))}
{science}
<div class="mai-grid">
  {_card("Selected styles", "requests", selected, right=_prov("real" if not live.empty else "pending"))}
  {_card("Why ratings are excluded", "shield", funnel, hint="Reasons can overlap", right=_prov("real" if not live.empty else "pending"))}
</div>
{_note("Exploration is an operational label for monitoring: a decision counts as exploration when the chosen style is not the one with the highest posterior mean at that moment. Thompson sampling itself has no separate exploration step, and the label is undefined while several styles share the highest mean.")}"""

    return f'<div class="mai-rl-section">{live_section}{_simulation_section()}</div>'


def _style_cell(arm: str) -> str:
    return (f'<th scope="row"><span class="mai-dot" style="background:{charts.TONE_COLOR[ARM_TONE[arm]]}"></span>'
            f'{_e(ARM_LABEL[arm])}</th>')


def _served_chart(logs: pd.DataFrame) -> tuple[str, bool]:
    served = rl_monitor.served_probabilities(logs)
    if len(served) < 2:
        return _empty("target", "Not enough served decisions", "Drawn from the probabilities logged with each request, once two or more exist."), False
    labels = [str(int(r.last_request)) for r in served.itertuples()]
    series = {ARM_LABEL[a]: [None if pd.isna(v) else float(v) for v in served[f"p_{a}"]] for a in ARMS}
    chart = charts.line_chart(series, labels, y_max=1.0, tones=[ARM_TONE[a] for a in ARMS], markers=len(served) < 25,
                              max_xlabels=6, title="Selection probabilities logged at serving time")
    size = int(served["n"].max())
    n_total, n_eligible = int(served["n"].sum()), int(served["eligible"].sum())
    chart += _note(f"X axis: request number, oldest first ({n_total} live decisions"
                   + (f", each point the mean of up to {size} consecutive requests" if size > 1 else "")
                   + f"). These are the probabilities the policy really used when it served each request, not a replay. "
                   f"{n_eligible} of {n_total} requests were feedback-eligible; the rest (stub, degraded, out of scope or "
                   "another version) were served by the same policy but cannot train it.")
    return chart, True


def _coverage_table(logs: pd.DataFrame, prompt_version: str) -> str:
    cov = rl_monitor.coverage_by_arm(logs, prompt_version)
    rows = "".join(f'<tr>{_style_cell(r.arm)}<td class="mai-num">{r.served}</td><td class="mai-num">{r.eligible}</td>'
                   f'<td class="mai-num">{r.rated}</td><td class="mai-num">{_pct(r.response_rate)}</td>'
                   f'<td class="mai-num">{r.counted}</td><td class="mai-num">{_pct(r.share_of_counted)}</td></tr>' for r in cov.itertuples())
    return (f'<table class="mai-table mai-table--rl"><thead><tr><th scope="col">Style</th><th scope="col" class="mai-num">Served</th>'
            f'<th scope="col" class="mai-num">Eligible</th><th scope="col" class="mai-num">Rated</th>'
            f'<th scope="col" class="mai-num">Response rate</th><th scope="col" class="mai-num">Counted</th>'
            f'<th scope="col" class="mai-num">Share of counted</th></tr></thead><tbody>{rows}</tbody></table>'
            + _note("Served: live decisions. Eligible: real traffic whose rating could train the policy. Rated: eligible requests "
                    f"with a rating. Counted: after the cap of {bandit.MAX_RATINGS_PER_SESSION} rated requests per session. "
                    "A style that is served less also collects fewer ratings: that is the feedback loop of any bandit."))


def _belief_table(logs: pd.DataFrame, prompt_version: str) -> str:
    comp = rl_monitor.belief_comparison(logs, prompt_version)
    rows = ""
    for r in comp.itertuples():
        flag = (f' <span class="mai-badge mai-badge--warning">{icon("info", 12)}Prior still weighs</span>' if r.prior_dominant else "")
        rows += (f'<tr>{_style_cell(r.arm)}<td class="mai-num">{r.n}</td><td class="mai-num">{_pct(r.observed)}</td>'
                 f'<td class="mai-num">{_pct(r.posterior_mean)} <small>[{_pct(r.ci_low)}–{_pct(r.ci_high)}]</small>{flag}</td>'
                 f'<td class="mai-num">{_pct(r.ips)}</td><td class="mai-num">{_pct(r.p_select)}</td></tr>')
    return (f'<table class="mai-table mai-table--rl"><thead><tr><th scope="col">Style</th><th scope="col" class="mai-num">Ratings</th>'
            f'<th scope="col" class="mai-num">Observed reward</th><th scope="col" class="mai-num">Posterior mean [95%]</th>'
            f'<th scope="col" class="mai-num">Propensity-weighted</th><th scope="col" class="mai-num">P(selected)</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
            + _note("Three different quantities. <strong>Observed reward</strong>: share of positive real ratings. <strong>Posterior "
                    "mean</strong>: the same, pulled towards 50% by the neutral prior while ratings are few. <strong>Propensity-"
                    "weighted</strong>: each rating weighted by 1 / the logged probability of the style served (self-normalised "
                    "IPS). An exploratory estimate only: it re-weights for the policy having served some styles more often over "
                    "time, but it is not a causal proof, it is noisy with few ratings, and it does not correct non-response "
                    "(it assumes ratings are missing at random given the style). <strong>P(selected)</strong> is the probability of being the best style, not a "
                    f"reward: it can be high with a modest mean. Gap flag: {rl_monitor.BELIEF_GAP:.0%} or more (assumption)."))


def _loop_card(logs: pd.DataFrame, prompt_version: str) -> tuple[str, bool]:
    nr = rl_monitor.non_response(logs, prompt_version)
    table = nr["table"]
    if table.empty:
        body = _empty("shield", "No feedback-eligible requests yet",
                      "Response rates by risk band, decision and style appear once real traffic is eligible.")
        has = False
    else:
        rows = "".join(f'<tr><th scope="row">{_e(DIM_LABEL[r.dimension])}: {_e(ARM_LABEL.get(r.group, BAND_LABEL.get(r.group, r.group)))}</th>'
                       f'<td class="mai-num">{r.eligible}</td><td class="mai-num">{r.rated}</td><td class="mai-num">{_pct(r.response_rate)}</td></tr>'
                       for r in table.itertuples())
        body = (f'<table class="mai-table mai-table--rl"><thead><tr><th scope="col">Group</th><th scope="col" class="mai-num">Eligible</th>'
                f'<th scope="col" class="mai-num">Rated</th><th scope="col" class="mai-num">Response rate</th></tr></thead><tbody>{rows}</tbody></table>')
        has = True
    if nr["flags"]:
        items = "".join(f'<li class="mai-alert mai-alert--medium"><span class="mai-alert__icon">{icon("bell", 16)}</span>'
                        f'<span class="mai-badge mai-badge--warning">Check</span><span>{_e(DIM_LABEL[f["dimension"]])}: '
                        f'response rates differ by {f["gap"]:.0%} ({_e(f["most_rated"])} rated most, {_e(f["least_rated"])} least). '
                        'The learned preference describes the orders people chose to rate.</span></li>' for f in nr["flags"])
        body += f'<ul class="mai-alerts">{items}</ul>'
    elif has:
        body += _note(f"No group gap flagged ({_e(nr['rule'])}). {nr['dimensions_with_enough_data']} of 3 dimensions had enough data to check.")
    balance = rl_monitor.arm_balance(logs, prompt_version)
    if not balance.empty:
        mix = "; ".join(f"{ARM_LABEL[a]}: " + ", ".join(f"{BAND_LABEL.get(b, b)} {sh:.0%}" for b, sh in zip(g["risk_band"], g["share"]))
                        for a, g in balance.groupby("arm"))
        body += _note(f"Risk mix of counted ratings per style: {_e(mix)}. Thompson sampling ignores the reading, so the mix should "
                      "differ only by chance; a style that got easier cases would look better for that reason.")
    body += _note("Feedback loop: the policy decides which style is served, so it decides which styles collect evidence. A style "
                  "with early bad luck is served less and its estimate stays uncertain (or pessimistic); Thompson sampling keeps "
                  "serving it with its selection probability, which limits this but does not remove it. Non-response: only rated orders teach the "
                  "policy, so if people rate mainly critical or surprising orders, the preference is about those.")
    return body, has


def _science_cards(logs: pd.DataFrame, prompt_version: str, learning: bool) -> str:
    served, has_served = _served_chart(logs)
    loop, has_loop = _loop_card(logs, prompt_version)
    return f"""
<div class="mai-grid">
  {_card("Selection probabilities actually served", "target", served, right=_prov("real" if has_served else "pending"))}
  {_card("Feedback coverage by style", "requests", _coverage_table(logs, prompt_version), right=_prov("real" if learning else "pending"))}
</div>
{_card("Observed reward, belief and selection", "sliders", _belief_table(logs, prompt_version), right=_prov("real" if learning else "pending"))}
{_card("Feedback loop and non-response", "shield", loop, hint="Descriptive checks", right=_prov("real" if has_loop else "pending"))}"""


def _calibration_block() -> str:
    if not (CAL_CSV.is_file() and CAL_JSON.is_file()):
        return _empty("flask", "No calibration results", "Run <code>uv run python eval/bandit_calibration.py</code>.")
    cal = pd.read_csv(CAL_CSV)
    summary = json.loads(CAL_JSON.read_text(encoding="utf-8"))
    overall = cal[cal["group"] == "all styles"]
    names = {"prior_draw": "Rates drawn from the prior", "one_better": "One better style", "all_equal": "All styles equal",
             "low_rates": "Low like-rates"}
    rounds = sorted(overall["round"].unique())
    head = "".join(f'<th scope="col" class="mai-num">{int(r)}</th>' for r in rounds)
    rows = ""
    for scen, g in overall.groupby("scenario", sort=False):
        g = g.set_index("round")
        cells = "".join(f'<td class="mai-num{" mai-num--warn" if g.loc[r, "coverage"] < 0.93 else ""}">{g.loc[r, "coverage"]:.1%}</td>'
                        for r in rounds)
        rows += f'<tr><th scope="row">{_e(names.get(scen, scen))}</th>{cells}</tr>'
    best = cal[(cal["group"] == "best style") & (cal["scenario"] == "one_better")].set_index("round")
    best_txt = ", ".join(f"{best.loc[r, 'coverage']:.0%} at {int(r)}" for r in rounds if r in best.index)
    starved = summary.get("starved_style_cells") or []
    starved_txt = ("; ".join(f"{names.get(c['scenario'], c['scenario'])}, round {c['round']}: {c['coverage']:.0%} of "
                             f"{c['intervals']} intervals" for c in starved))
    table = (f'<table class="mai-table"><thead><tr><th scope="col">Scenario · ratings so far</th>{head}</tr></thead><tbody>{rows}</tbody></table>')
    return (f'<h4 class="mai-h3">Coverage of the 95% credible intervals</h4>{table}'
            + _note(f"Share of intervals that contain the true (declared) like-rate, all styles, {summary['runs']} runs per scenario "
                    f"(Monte Carlo error about ±1 point). Nominal: 95%. Highlighted: below 93%. With rates drawn from the prior the "
                    f"coverage is close to 95%, which checks the implementation. Adaptive sampling matters: the best style is "
                    f"slightly under-covered while it is being found ({_e(best_txt)}), and the styles that are rarely served are "
                    f"over-covered (wide intervals)."
                    + (f" Small samples: a style starved after early bad luck can be badly under-covered ({_e(starved_txt)})." if starved else "")))


def _simulation_section() -> str:
    if not (SIM_CSV.is_file() and SIM_JSON.is_file()):
        return _card("Simulation results", "flask", _empty("flask", "No simulation results", "Run <code>uv run python eval/simulate_bandit.py</code>."),
                     cls="mai-sim")
    curve = pd.read_csv(SIM_CSV)
    summary = json.loads(SIM_JSON.read_text(encoding="utf-8"))
    one = curve[curve["scenario"] == "one_better"]
    labels = [str(int(r)) for r in sorted(one["round"].unique())]
    prior_label = {"neutral": "Neutral Beta(1,1)", "capped_plan": "Capped plan prior", "wrong_belief": "Wrong prior"}
    pbest = {prior_label[p]: list(g.sort_values("round")["p_best"]) for p, g in one.groupby("prior", sort=False)}
    p_chart = charts.line_chart(pbest, labels, y_max=1.0, tones=["info", "stub", "orange"], markers=False, max_xlabels=6,
                                dashed=("Capped plan prior",), title="Simulated probability of selecting the best style")
    neutral = one[one["prior"] == "neutral"].sort_values("round")
    wrong = one[one["prior"] == "wrong_belief"].sort_values("round")
    r_chart = charts.line_chart({"Thompson, neutral prior": list(neutral["regret_ts"]), "Thompson, wrong prior": list(wrong["regret_ts"]),
                                 "Random selection": list(neutral["regret_random"])}, labels,
                                y_fmt=lambda v: f"{v:.0f}", tones=["info", "orange", "neutral"], markers=False, max_xlabels=6,
                                title="Simulated cumulative regret")
    rows = ""
    for c in summary["comparison"]:
        if c["scenario"] != "one_better":
            continue
        rows += (f'<tr><th scope="row">{_e(prior_label.get(c["prior"], c["prior"]))}</th><td class="mai-num">{c["p_best_50"]:.0%}</td>'
                 f'<td class="mai-num">{c["p_best_300"]:.0%}</td><td class="mai-num">{c["first_round_p_best_ge_0.8"] or "–"}</td>'
                 f'<td class="mai-num">{c["regret_ts_300"]:.1f}</td><td class="mai-num">{c["regret_random_300"]:.1f}</td></tr>')
    tie = next((c for c in summary["comparison"] if c["scenario"] == "all_equal" and c["prior"] == "neutral"), None)
    tie_note = (" With three equally good styles the neutral policy stays spread out (final probabilities "
                + ", ".join(f"{ARM_LABEL[a]} {p:.0%}" for a, p in tie["final_p_select"].items()) + "): no false winner.") if tie else ""
    rates = summary["scenarios"]["one_better"]
    table = (f'<table class="mai-table"><thead><tr><th scope="col">Prior</th><th scope="col" class="mai-num">P(best) at 50</th>'
             f'<th scope="col" class="mai-num">P(best) at 300</th><th scope="col" class="mai-num">Rounds to 80%</th>'
             f'<th scope="col" class="mai-num">Regret, Thompson</th><th scope="col" class="mai-num">Regret, random</th></tr></thead><tbody>{rows}</tbody></table>')
    body = f"""
<div class="mai-banner mai-banner--stub" role="note">{icon("flask", 18)}<div><strong>Simulated users, not real feedback.</strong>
  {summary["runs"]} runs of {summary["rounds"]} ratings with declared like-rates ({", ".join(f"{ARM_LABEL[a]} {r:.0%}" for a, r in rates.items())}).
  Same Thompson-sampling code as production. Nothing here trains the live policy.</div></div>
<div class="mai-grid">
  <div><h4 class="mai-h3">Probability of choosing the best style</h4>{p_chart}</div>
  <div><h4 class="mai-h3">Cumulative regret vs random selection</h4>{r_chart}</div>
</div>
<h4 class="mai-h3">Prior comparison</h4>{table}
{_calibration_block()}
{_note("Regret: expected positive ratings lost against always choosing the best style. Production uses the neutral prior: "
       "the capped plan prior performs about the same, while a wrong prior costs extra regret before real ratings correct it."
       + _e(tie_note))}"""
    return _card("Simulation results", "flask", body, cls="mai-sim",
                 right='<span class="mai-badge mai-badge--stub">' + icon("flask", 14) + 'Simulated</span>')

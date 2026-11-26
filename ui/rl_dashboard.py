"""Adaptive Intelligence: the reinforcement-learning section of Monitoring (WP6, presentation).

Two parts that never mix:
- Live policy: everything comes from bandit.policy_snapshot(), policy_history() and
  the logged requests. If no real rating is eligible yet, the learning charts show
  an empty state; they are never filled with simulated data.
- Simulation results: eval/bandit_simulation*.{csv,json}, written by
  eval/simulate_bandit.py, shown in a separate, clearly labelled section.
"""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from maintai import bandit
from maintai.schemas import ARMS
from ui import charts
from ui.icons import icon

ROOT = Path(__file__).resolve().parent.parent
SIM_CSV = ROOT / "eval" / "bandit_simulation.csv"
SIM_JSON = ROOT / "eval" / "bandit_simulation_summary.json"

ARM_LABEL = {"checklist": "Checklist", "cost": "Cost-focused", "root_cause": "Root-cause"}
ARM_TONE = {"checklist": "info", "cost": "orange", "root_cause": "brand"}
ARM_HINT = {"checklist": "Short numbered steps", "cost": "Opens with cost and window", "root_cause": "Explains the indicators"}


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _card(title: str, icon_name: str, body: str, hint: str = "", cls: str = "", right: str = "") -> str:
    hint_html = f'<span class="mai-card__hint">{_e(hint)}</span>' if hint else ""
    return (f'<section class="mai-card {cls}"><div class="mai-card__header"><span class="mai-card__icon">{icon(icon_name, 16)}</span>'
            f'<h3 class="mai-h2">{_e(title)}</h3>{hint_html}<span class="mai-card__right">{right}</span></div>{body}</section>')


def _note(text: str) -> str:
    return f'<p class="mai-footnote">{text}</p>'


def _empty(icon_name: str, title: str, text: str) -> str:
    return f'<div class="mai-chart-empty">{icon(icon_name, 22)}<strong>{_e(title)}</strong><span class="mai-subtle">{text}</span></div>'


def _kpi(icon_name: str, label: str, value: str, foot: str) -> str:
    return (f'<div class="mai-kpi mai-kpi--tile"><span class="mai-kpi__icon">{icon(icon_name, 18)}</span>'
            f'<span class="mai-kpi__label">{_e(label)}</span><span class="mai-kpi__value">{_e(value)}</span>'
            f'<span class="mai-kpi__foot">{_e(foot)}</span></div>')


def _live_requests(logs: pd.DataFrame) -> pd.DataFrame:
    """Requests whose arm was chosen by the current live policy (any source)."""
    if logs is None or logs.empty or "arm_probs" not in logs:
        return pd.DataFrame()
    versions = logs.get("versions", pd.Series([None] * len(logs), index=logs.index))
    live = logs["arm_probs"].apply(lambda v: isinstance(v, dict) and len(v) > 0) & \
        versions.apply(lambda v: isinstance(v, dict) and v.get("bandit") in bandit.COMPATIBLE_POLICIES)
    return logs[live]


def _exclusions(live: pd.DataFrame, prompt_version: str) -> list[tuple[str, int]]:
    """Why live requests do not train the policy (a request can have several reasons)."""
    if live.empty:
        return []
    v = live.get("versions")
    fb = pd.to_numeric(live.get("feedback"), errors="coerce") if "feedback" in live else pd.Series(np.nan, index=live.index)
    return [
        ("Stub component versions (development)", int(v.apply(bandit._has_stub).sum())),
        ("Degraded mode (template order, no agent)", int(live.get("degraded", pd.Series(False, index=live.index)).fillna(False).astype(bool).sum())),
        ("Out of scope", int((live.get("status", pd.Series("ok", index=live.index)) == "out_of_scope").sum())),
        ("Not served as real traffic (source ≠ real)", int((live.get("source", pd.Series("", index=live.index)) != "real").sum())),
        (f"Different prompt version (current: {prompt_version})", int((v.apply(bandit._prompt_version) != prompt_version).sum())),
        ("No rating given", int((~fb.isin([1, -1])).sum())),
    ]


def render(logs: pd.DataFrame | None) -> str:
    logs = logs if logs is not None else pd.DataFrame()
    prompt_version = bandit.current_prompt_version()
    snap = bandit.policy_snapshot(logs)
    fb = bandit.eligible_feedback(logs)
    try:
        events = bandit.load_feedback_events()
    except Exception:  # noqa: BLE001 - the history chart then shows its empty state
        events = pd.DataFrame(columns=["id", "timestamp", "feedback"])
    hist = bandit.policy_history(logs, events)
    live = _live_requests(logs)
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
</div>"""

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
        _kpi("target", "Mean reward (real)", f"{fb['reward'].mean():.0%}" if learning else "–", "Share of eligible ratings that are positive"),
        _kpi("drift", "Exploration rate", f"{explore.astype(bool).mean():.0%}" if len(explore) else "–",
             f"{len(explore)} decision(s) with a single greedy style"),
        _kpi("alert", "Prior-only decisions", f"{fallback}", "Logs unreadable: chose from the prior"),
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

    excl = _exclusions(live, prompt_version)
    excl_html = "".join(f'<li><span>{_e(t)}</span><strong>{n}</strong></li>' for t, n in excl) or '<li><span>No live decisions yet</span></li>'
    funnel = (f'<ul class="mai-kv mai-kv--compact">{excl_html}</ul>'
              + _note("A rating trains the policy only if the request was real traffic, handled, not degraded, free of stub "
                      "versions and written with the current prompt version. Only the latest rating per request counts."))

    live_section = f"""
{head}{waiting}
<div class="mai-kpi-row mai-kpi-row--4">{kpis}</div>
{_card("Learning cycle", "trace", cycle_html, hint="Rebuilt on every request")}
<div class="mai-grid">
  {_card("Selection probability", "target", sel + _note("Probability that each style is chosen for the next request (4,000 posterior draws). Sums to 100%."))}
  {_card("Estimated reward and uncertainty", "sliders", forest_html + _note("Posterior mean of the share of positive ratings, with its 95% credible interval. Wide bars mean little evidence."))}
</div>
<div class="mai-grid">
  {_card("Posterior distributions", "drift", beta_chart + _note("Beta(α, β) density of each style's reward rate. Flat means no evidence; a narrow peak means a confident estimate."))}
  {_card("Policy evolution (real feedback)", "pulse", evo)}
</div>
{_card("Strategy table", "clipboard", table)}
<div class="mai-grid">
  {_card("Selected styles", "requests", selected)}
  {_card("Why ratings are excluded", "shield", funnel, hint="Reasons can overlap")}
</div>
{_note("Exploration is an operational label for monitoring: a decision counts as exploration when the chosen style is not the one with the highest posterior mean at that moment. Thompson sampling itself has no separate exploration step, and the label is undefined while several styles share the highest mean.")}"""

    return f'<div class="mai-rl-section">{live_section}{_simulation_section()}</div>'


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
{_note("Regret: expected positive ratings lost against always choosing the best style. Production uses the neutral prior: "
       "the capped plan prior performs about the same, while a wrong prior costs extra regret before real ratings correct it."
       + _e(tie_note))}"""
    return _card("Simulation results", "flask", body, cls="mai-sim",
                 right='<span class="mai-badge mai-badge--stub">' + icon("flask", 14) + 'Simulated</span>')

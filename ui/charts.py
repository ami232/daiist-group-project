"""Small, dependency-free charts (inline SVG and CSS) in the MaintAI style.

Every function takes numbers that already exist (a record, the logs or a
monitor.py result) and only draws them. Nothing is estimated or smoothed here.
Labels pass through html.escape. Each chart has a text alternative (aria-label
or a visually hidden table) so screen readers get the same information.
"""

import html
import math

TONE_COLOR = {"success": "#1F8A4C", "warning": "#C27A0E", "orange": "#D2601A", "danger": "#C62828",
              "info": "#007C9E", "neutral": "#94A3B8", "brand": "#102E4A", "ice": "#A8D8EA", "stub": "#7C5CC4"}


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(100.0, (value - low) / (high - low) * 100))


# --- Gauge -----------------------------------------------------------------------

def gauge(p: float, threshold: float, tone: str, label: str) -> str:
    """Semicircular gauge of a probability (0-1), linear scale, with the decision threshold marked."""
    p = max(0.0, min(1.0, float(p)))
    cx, cy, r = 110, 110, 86

    def point(f: float, radius: float = r) -> tuple[float, float]:
        angle = math.pi * (1 - f)
        return cx + radius * math.cos(angle), cy - radius * math.sin(angle)

    x0, y0 = point(0)
    x1, y1 = point(1)
    xp, yp = point(max(p, 0.004))
    t_in, t_out = point(threshold, r - 13), point(threshold, r + 13)
    tl = point(threshold, r + 24)
    value_arc = f'<path d="M{x0:.1f} {y0:.1f} A{r} {r} 0 0 1 {xp:.1f} {yp:.1f}" class="mai-gauge__value" stroke="{TONE_COLOR[tone]}"/>'
    return f"""
<figure class="mai-gauge" role="img" aria-label="Failure probability {p:.1%}, {_e(label)} risk, decision threshold {threshold:.1%}">
  <svg viewBox="-14 -6 248 140" aria-hidden="true" focusable="false">
    <path d="M{x0:.1f} {y0:.1f} A{r} {r} 0 0 1 {x1:.1f} {y1:.1f}" class="mai-gauge__track"/>
    {value_arc}
    <line x1="{t_in[0]:.1f}" y1="{t_in[1]:.1f}" x2="{t_out[0]:.1f}" y2="{t_out[1]:.1f}" class="mai-gauge__tick"/>
    <text x="{tl[0]:.1f}" y="{tl[1] + 3:.1f}" class="mai-gauge__ticklabel" text-anchor="middle">{threshold:.0%}</text>
    <text x="{cx}" y="{cy - 22}" class="mai-gauge__number" text-anchor="middle">{p * 100:.1f}<tspan class="mai-gauge__unit">%</tspan></text>
    <text x="{cx}" y="{cy - 2}" class="mai-gauge__caption" text-anchor="middle">failure probability</text>
    <text x="{x0:.0f}" y="{cy + 18}" class="mai-gauge__axis" text-anchor="middle">0%</text>
    <text x="{x1:.0f}" y="{cy + 18}" class="mai-gauge__axis" text-anchor="middle">100%</text>
  </svg>
</figure>"""


# --- Bars ------------------------------------------------------------------------

def bar_rows(rows: list[tuple[str, float, str, str]], max_value: float = 1.0) -> str:
    """Horizontal bars: (label, value, value text, tone)."""
    out = []
    for label, value, text, tone in rows:
        width = _pct(value, 0, max_value)
        out.append(f"""
<div class="mai-bar">
  <div class="mai-bar__top"><span>{_e(label)}</span><strong>{_e(text)}</strong></div>
  <div class="mai-bar__track" aria-hidden="true"><span style="width:{width:.1f}%;background:{TONE_COLOR[tone]}"></span></div>
</div>""")
    return f'<div class="mai-bars">{"".join(out)}</div>'


def limit_bar(label: str, value: float, value_text: str, domain: tuple[float, float],
              risk_zones: list[tuple[float, float]], in_risk: bool, rule: str) -> str:
    """A value on a scale with its documented risk zones shaded."""
    low, high = domain
    zones = "".join(f'<span class="mai-limit__zone" style="left:{_pct(a, low, high):.1f}%;width:{_pct(b, low, high) - _pct(a, low, high):.1f}%"></span>'
                    for a, b in risk_zones)
    state = "risk" if in_risk else "ok"
    status = "In risk zone" if in_risk else "Within limits"
    return f"""
<div class="mai-limit mai-limit--{state}">
  <div class="mai-limit__top"><span class="mai-limit__label">{_e(label)}</span>
    <span class="mai-limit__value">{_e(value_text)}</span></div>
  <div class="mai-limit__track" role="img" aria-label="{_e(label)}: {_e(value_text)}. {_e(status)}. {_e(rule)}">
    {zones}<span class="mai-limit__marker" style="left:{_pct(value, low, high):.1f}%"></span></div>
  <div class="mai-limit__foot"><span class="mai-limit__state">{_e(status)}</span><span>{_e(rule)}</span></div>
</div>"""


def stacked_costs(rows: list[tuple[str, list[tuple[str, float, str]]]], fmt) -> str:
    """Stacked horizontal bars: (row label, [(segment label, value, tone)]). Shared scale."""
    totals = [sum(v for _, v, _ in segs) for _, segs in rows]
    top = max(totals + [1e-9])
    out = []
    for (label, segs), total in zip(rows, totals):
        spans = "".join(f'<span style="width:{v / top * 100:.2f}%;background:{TONE_COLOR[tone]}" title="{_e(name)}: {_e(fmt(v))}"></span>'
                        for name, v, tone in segs if v > 0)
        out.append(f"""
<div class="mai-stack__row">
  <div class="mai-stack__label"><span>{_e(label)}</span><strong>{_e(fmt(total))}</strong></div>
  <div class="mai-stack__track" aria-hidden="true">{spans}</div>
</div>""")
    legend = {}
    for _, segs in rows:
        for name, v, tone in segs:
            legend.setdefault(name, tone)
    legend_html = "".join(f'<li><span style="background:{TONE_COLOR[t]}"></span>{_e(n)}</li>' for n, t in legend.items())
    table = "".join(f"<tr><th scope='row'>{_e(label)}</th><td>{_e(', '.join(f'{n} {fmt(v)}' for n, v, _ in segs))}</td></tr>"
                    for label, segs in rows)
    return (f'<div class="mai-stack">{"".join(out)}<ul class="mai-legend">{legend_html}</ul>'
            f'<table class="mai-sr"><caption>Expected cost per process</caption>{table}</table></div>')


# --- Time series -----------------------------------------------------------------

def line_chart(series: dict[str, list[float]], labels: list[str], y_max: float | None = None,
               y_fmt=lambda v: f"{v:.0%}", tones: list[str] | None = None, reference: tuple[float, str] | None = None,
               title: str = "", markers: bool = True, dashed: tuple[str, ...] = (), max_xlabels: int = 6) -> str:
    """Line chart over shared x labels. Draws only the points it is given."""
    w, h, pl, pr, pt, pb = 640, 220, 44, 12, 12, 28
    tones = tones or ["info", "orange", "brand", "stub"]
    values = [v for vs in series.values() for v in vs if v is not None]
    if not values or not labels:
        return ""
    top = y_max if y_max is not None else (max(values + ([reference[0]] if reference else [])) * 1.15 or 1)
    n = len(labels)

    def xy(i: int, v: float) -> tuple[float, float]:
        x = pl + (i / max(n - 1, 1)) * (w - pl - pr) if n > 1 else (pl + w - pr) / 2
        return x, pt + (1 - min(v, top) / top) * (h - pt - pb)

    grid = ""
    for k in range(5):
        v = top * k / 4
        _, y = xy(0, v)
        grid += f'<line x1="{pl}" x2="{w - pr}" y1="{y:.1f}" y2="{y:.1f}" class="mai-chart__grid"/><text x="{pl - 8}" y="{y + 4:.1f}" text-anchor="end" class="mai-chart__axis">{_e(y_fmt(v))}</text>'
    ref = ""
    if reference:
        _, y = xy(0, reference[0])
        ref = (f'<line x1="{pl}" x2="{w - pr}" y1="{y:.1f}" y2="{y:.1f}" class="mai-chart__ref"/>'
               f'<text x="{pl + 6}" y="{y - 6:.1f}" text-anchor="start" class="mai-chart__reflabel">{_e(reference[1])}</text>')
    lines, legend = "", ""
    for (name, vs), tone in zip(series.items(), tones):
        pts = [xy(i, v) for i, v in enumerate(vs) if v is not None]
        color = TONE_COLOR[tone]
        dash = ' stroke-dasharray="6 5"' if name in dashed else ""
        if len(pts) > 1:
            lines += f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"{dash}/>'
        if markers or len(pts) == 1:
            lines += "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#fff" stroke="{color}" stroke-width="2"/>' for x, y in pts)
        legend += f'<li><span style="background:{color}"></span>{_e(name)}</li>'
    step = max(1, math.ceil(n / max_xlabels))
    xlabels = "".join(f'<text x="{xy(i, 0)[0]:.1f}" y="{h - 8}" text-anchor="middle" class="mai-chart__axis">{_e(lab)}</text>'
                      for i, lab in enumerate(labels) if i % step == 0 or i == n - 1)
    rows = "".join(f"<tr><th scope='row'>{_e(lab)}</th>" + "".join(f"<td>{'' if vs[i] is None else _e(y_fmt(vs[i]))}</td>" for vs in series.values()) + "</tr>"
                   for i, lab in enumerate(labels))
    head = "".join(f"<th scope='col'>{_e(name)}</th>" for name in series)
    return f"""
<figure class="mai-chart">
  <svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" aria-hidden="true" focusable="false">{grid}{ref}{lines}{xlabels}</svg>
  <ul class="mai-legend">{legend}</ul>
  <table class="mai-sr"><caption>{_e(title)}</caption><tr><th scope='col'>Time</th>{head}</tr>{rows}</table>
</figure>"""


def column_chart(values: list[float], labels: list[str], tone: str = "info", title: str = "") -> str:
    """Vertical columns (counts per period)."""
    if not values:
        return ""
    w, h, pl, pr, pt, pb = 640, 200, 36, 8, 12, 28
    top = max(values) or 1
    n = len(values)
    slot = (w - pl - pr) / n
    bw = min(36, slot * 0.62)
    cols = ""
    for i, v in enumerate(values):
        bh = v / top * (h - pt - pb)
        x = pl + i * slot + (slot - bw) / 2
        cols += f'<rect x="{x:.1f}" y="{h - pb - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{TONE_COLOR[tone]}"><title>{_e(labels[i])}: {v:g}</title></rect>'
    step = max(1, math.ceil(n / 8))
    xl = "".join(f'<text x="{pl + i * slot + slot / 2:.1f}" y="{h - 8}" text-anchor="middle" class="mai-chart__axis">{_e(lab)}</text>'
                 for i, lab in enumerate(labels) if i % step == 0)
    grid = "".join(f'<line x1="{pl}" x2="{w - pr}" y1="{h - pb - k / 2 * (h - pt - pb):.1f}" y2="{h - pb - k / 2 * (h - pt - pb):.1f}" class="mai-chart__grid"/>'
                   f'<text x="{pl - 6}" y="{h - pb - k / 2 * (h - pt - pb) + 4:.1f}" text-anchor="end" class="mai-chart__axis">{top * k / 2:.0f}</text>'
                   for k in range(3))
    rows = "".join(f"<tr><th scope='row'>{_e(lab)}</th><td>{v:g}</td></tr>" for lab, v in zip(labels, values))
    return (f'<figure class="mai-chart"><svg viewBox="0 0 {w} {h}" aria-hidden="true" focusable="false">{grid}{cols}{xl}</svg>'
            f'<table class="mai-sr"><caption>{_e(title)}</caption>{rows}</table></figure>')


def segmented_bar(parts: list[tuple[str, int, str]]) -> str:
    """One 100% bar split into categories: (label, count, tone)."""
    total = sum(c for _, c, _ in parts)
    if not total:
        return ""
    spans = "".join(f'<span style="width:{c / total * 100:.2f}%;background:{TONE_COLOR[t]}"></span>' for _, c, t in parts if c)
    legend = "".join(f'<li><span style="background:{TONE_COLOR[t]}"></span>{_e(n)} <strong>{c}</strong> <em>{c / total:.0%}</em></li>'
                     for n, c, t in parts)
    return f'<div class="mai-seg"><div class="mai-seg__track" aria-hidden="true">{spans}</div><ul class="mai-legend mai-legend--stat">{legend}</ul></div>'

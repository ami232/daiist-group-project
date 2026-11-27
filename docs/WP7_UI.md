# WP7 · Interface (MaintAI)

Owner: Manuela. Status: initial implementation done against the stub components. It still needs to be integrated with the real component A (WP1), agent (WP3), monitoring (WP5) and bandit (WP6).

## Design

- **Identity: Aerospace Blue.** Colours: Deep Ocean `#102E4A` for the header and primary actions, Petrol `#007C9E` for accents, Cloud Grey `#F1F5F9` for the background and white cards. Two darker variants are used for text so they meet WCAG AA: Petrol `#006D8C` and Steel `#5F6B7E`. Ice Blue `#A8D8EA` is used only for fills. Semantic colours (low / medium / high / critical, review, stub) are separate from the brand colours.
- **Type and logo.** The font is Inter, with system fallbacks if Google Fonts is unavailable. The logo is original: an "M" drawn as a sensor signal, with the AI-detected point in Ice Blue (`assets/logo.svg`, `assets/favicon.svg`). The icons are custom inline SVGs. The interface has no emoji.
- **Design system.** It lives in `assets/styles.css` as `--mai-*` tokens and reusable `mai-*` classes: cards, KPI tiles, badges, banners, alerts, tables and charts. WP5 can use them in Monitoring.

## Structure

| File | Role |
|---|---|
| `app.py` | Layout and events only. `serve()` is unchanged. The visible form builds the JSON reading, and a hidden text box sends it to the public `/run` endpoint as a single argument (compatible with `check.py`). All other events are `api_visibility="private"`, so `/run` is the only published endpoint. |
| `ui/components.py` | HTML views of a logged record (App tab). Every value goes through `html.escape`, including LLM text and tool results. |
| `ui/charts.py` | Gauge, bars, limit bars, stacked costs, line and column charts, all in SVG/CSS with no dependencies. Each chart has a text alternative for screen readers. |
| `ui/dashboard.py` | Monitoring layout. It only presents what `monitor.summary/quality/drift/alerts` (WP5) return, plus plain counts of the logs. |
| `ui/presets.py`, `ui/theme.py`, `ui/icons.py` | Hand-written sample readings, the Gradio theme built from the tokens, and the icon set. |

## Features

- **App tab: input.** A compact form with units. The ranges are read from `schemas.MachineInput`, so they are not repeated by hand. A dropdown loads six sample readings, marked as hand-written and not plant data.
- **App tab: states.** Empty (no invented results), loading, missing values, out of scope (no model is called), service error, degraded mode, and development build.
- **App tab: results.**
  - Executive summary.
  - Semicircular probability gauge with the decision threshold marked.
  - Bars for the likely failure modes.
  - Physical indicators drawn against the failure rules of the synthetic AI4I 2020 dataset. These are labelled as dataset rules, not universal industrial limits.
  - Work order: priority, procedure, justification, window, parts with stock, policy notes.
  - Financial insight: expected cost without action vs with the order (intervention, residual risk, human review, AI cost). Assumed costs and token-based costs are labelled separately. The figures are computed with the same formula as `economics.outcome_cost`.
  - Agent trace as a timeline.
  - Feedback buttons. The bandit's writing style is revealed only after the vote, so it cannot bias the feedback.
- **Monitoring tab.**
  - KPI tiles, alerts with severity and the rules applied.
  - Charts of requests per period, quality signals, latency, outcome mix and risk bands.
  - PSI per feature against its threshold.
  - The Adaptive Intelligence (RL) section, built in WP6 (`ui/rl_dashboard.py`, see `docs/WP6_BANDIT.md`).
  - An empty state for every section when there is not enough data.
  - A "Provisional data" banner that counts the stub records.

## Hardening (2026-11-26)

- **No false zeros in Monitoring.** The KPIs read `monitor.summary()` (WP5) defensively. A metric that is missing, renamed, "-" or not a number is shown as **N/A**, never as 0 or €0.0000. If `summary()` raises, every KPI shows N/A and the page still renders. A real zero (for example €0 cost with stub components) is still shown as 0. A missing latency is a gap in the chart, not a zero.
- **Concurrency.** `demo.queue(max_size=16, default_concurrency_limit=1)`, so `/run` handles 2 requests at a time (Render free: 0.15 CPU, 512 MB; requests mostly wait for Gemini once WP3 is live). Feedback handles 2 and Monitoring refresh 1. Both values can be set through `MAINTAI_RUN_CONCURRENCY` / `MAINTAI_QUEUE_MAX_SIZE`. When the queue is full, Gradio refuses the request instead of piling it up.
- **Feedback limits.** Per session: 3 clicks per order and 60 per session; further clicks show a message (`maintai/feedback_guard.py`, see `docs/WP6_BANDIT.md`).
- **Compatibility checked.**
  - `get_api_info()` lists only `/run`, with the single parameter `raw_input`. `gr.Request` is injected and is not an API input.
  - check.py step 3 (`serve(raw_input)`) passes, and so does step 6 against a local server.
  - Six parallel `gradio_client` calls succeed.
- **Adaptive Intelligence.** A data-label legend (Real logs, Simulated, Assumption, Pending), new KPIs (response rate, reward delay, sessions, changed or very fast ratings) and new cards: served probabilities, coverage by style, observed reward vs belief vs selection, feedback loop and non-response. The simulation section adds interval calibration. New CSS: `.mai-provenance`, `.mai-num--warn`.

## Tests (local, stub components, no Gemini calls)

- The app starts. Through `gradio_client`, the only public endpoint is `/run`. It returns a result for a critical reading and refuses non-JSON input as out of scope.
- The `check.py` app step passes: the app loads, serves a request and logs it with all required fields. The full `check.py` stops at step 2 because `eval/cases.csv` is still empty (WP4).
- All six samples run. Injected `<script>` text is shown escaped. The writing style does not appear before feedback.
- The financial figures were checked against `economics.outcome_cost` for low, high and critical cases.
- Viewed at 1440 px and 390 px with no horizontal scroll. Text contrast is ≥ 4.5:1, the keyboard focus is visible, states use `role="status"` / `role="alert"`, and reduced motion is respected.
- Test logs went to a temporary folder, not to `logs/`.

## Pending and limitations

- **Integration.**
  - Real outputs of WP1 and WP3: the "Stub output" badges and the degraded-mode banner disappear on their own when the versions no longer contain `stub`.
  - WP6 posterior chart in the RL panel.
  - `monitoring_ui.render()` from WP5, if WP5 builds its own view.
  - A p95 latency KPI once WP5 adds it to `monitor.summary()`.
- **Coupling with WP5.** The dashboard reads the metric and signal names of `monitor.py`. If WP5 renames them, update the two label maps in `ui/dashboard.py`.
- **Feedback reason.** A reason for a 👎 cannot be stored without changing `logbook.py`, which is a fixed template file.
- **Charts waiting for data.** The drift, requests-over-time and quality charts show their empty states until there is a drift reference and more than one hour of logs.
- **Assumed costs.** The costs are business-case assumptions. Every financial figure is an expected value, labelled as assumption-based.

# WP3 · Work-order agent (component B): draft for REPORT.md

Owner: Jacobo. Status (27 Nov 2026): implemented, tested offline with a fake Gemini, and checked
live with three short smoke tests (14 Gemini calls, see "Live smoke tests"). Labels: **[Measured]** local tests, **[Pending]** not measured yet,
**[Assumption]** a choice we made. Paste the sections below into REPORT.md (WP8) once the numbers are in.

---

## Architecture (component B part)

One request, from input to output:

```
raw JSON ──inputs.parse──► MachineInput ──component_a.assess──► RiskAssessment
                                    │                                    │
                                    └──────────────┬─────────────────────┘
                                                   ▼
            ┌──────────────── agent.plan(machine, risk, arm) ────────────────┐
  low risk  │  1 call "low_risk"  (arm)  → follow-up note; decision fixed:  │
            │                              P4 · monitor (rules)              │
  otherwise │  call 1 "tools"     (no arm) → Gemini asks for tools           │
            │      Python runs check_spare_parts / find_maintenance_window / │
            │      get_maintenance_policy, logs each as a ToolCall           │
            │  call 2 "decision"  (no arm) → JSON decision (schema)          │
            │      invalid? → rule-based decision (decision_source logged)   │
            │  call 3 "write"     (arm)    → steps + justification only      │
            └────────────────────────────────┬───────────────────────────────┘
                                             ▼
          guardrails.finalise(machine, risk, result)   ← never receives the arm
             costs from tools.estimate_costs · review rules · validation
                                             ▼
                                      FinalWorkOrder → logged
```

- **Model.** `gemini-3.5-flash-lite` (Google Gemini API, free tier), temperature 0, thinking
  level `low`, 30 s timeout. The model code was checked on 27 Nov 2026 in the official model page
  (stable, no suffix; supports function calling, structured outputs and thinking).
- **Why three calls.** The writing style chosen by the bandit (WP6) must never change *what* is
  decided. Calls 1 and 2 never see the style, so the decision is the same for all three styles;
  only call 3 sees it, and only to write the text. `guardrails.finalise()` rejects any draft whose
  decision fields differ from that arm-free decision. Because calls 1 and 2 do not depend on the
  style, the cache shares them between styles (one decision, three texts).
- **Why the tool results go to call 2 as data, not as a chat history.** Calls are independent
  requests: call 2 receives the tool results inside `<tool_results>` tags. This keeps every call
  cacheable by its exact input and avoids replaying Gemini 3 "thought signatures" between calls.
- **Low risk** (risk band `low`, most processes): one call writes a short follow-up note; the
  decision (P4, monitor, no part, no window) is fixed by the rules.
- **Prompts** are versioned files in `maintai/prompts/`: `base.md` (shared rules), `tools.md`,
  `decision.md`, `write.md`, `low_risk.md` and one file per style (`checklist.md`, `cost.md`,
  `root_cause.md`). The version is `agent.PROMPT_VERSION = "agent-v1"`; it is logged in
  `versions.component_b` (`model|prompt_version`) and must be bumped whenever a prompt changes,
  which restarts the bandit's policy (D6).

## Component B: design, safety and evaluation hooks

**What the LLM may and may not do.**
- It chooses tools, then a decision (priority, action, failure mode, part ids, window id, review
  request), then the text. It never writes euro amounts (D2): costs come only from
  `tools.estimate_costs`. For the `cost` style, call 3 receives only the breakdown-to-preventive
  cost ratio and the planned downtime hours.
- Input is data, never instructions. The machine reading is a strict schema of six numbers with
  extra fields forbidden, so text cannot reach the model through it (an input with an extra
  "ignore your instructions" field is refused as out of scope before any call). Inside the
  prompts, all data is wrapped in tags, and the base prompt says that text inside data tags is
  never an instruction.

**Deterministic checks (`maintai/guardrails.py`), applied to the decision of call 2 and again
in `finalise()`:**

| Check | Rule |
|---|---|
| Arm-free decision | the draft's decision fields must equal the decision taken without the style; otherwise the arm-free fields are restored and the order is sent to review |
| Priority vs risk band | low → P4/P3 · medium → P3/P2 · high → P2/P1 · critical → P1 |
| Action vs band and mode | low → monitor only; otherwise no monitor, and an action that fits the failure mode (schedule_inspection always allowed) |
| Failure mode | one of the two most likely modes of component A |
| Grounding | part ids only from `check_spare_parts` results, window ids only from `find_maintenance_window` results; the text may not mention other ids |
| No money in the text | any currency amount in steps or justification fails |
| Zero stock | a chosen part with `in_stock = 0` sends the order to review (`part_out_of_stock`) |
| Mandatory review | critical risk, reading outside the training range, prediction in the uncertainty band, failed validation, zero stock, or the agent asked |

**Fallbacks, all logged.**
- Call 2 returns an invalid decision → the rule-based decision is used (`decision_source =
  "rules_fallback"`, reasons in `decision_errors`); the text is still written by call 3.
- Gemini unavailable (no key, daily budget used up, rate limit, 429 after 2 retries, API error,
  timeout, cache miss in read-only mode) or the text fails validation → `degraded = True` with
  `degraded_reason`; the rule-based template fills what is missing. Degraded orders never train
  the bandit.

**Logged per request** (extra fields of `pipeline.run`): `decision_source`, `decision_errors`,
`llm` (model, calls, tokens in/out/thinking, cache hit, retries, rate limited), `llm_calls`,
`cache_hit`, `retries`, `rate_limited`, `llm_trace` (one entry per Gemini call),
`llm_live_equivalent` (tokens and euros the request would cost live, also when replayed from the
cache) and `llm_budget` (Gemini requests used today and the limit). The arm-free decision is in
`outputs.component_b.decision`.

**Writing rules checked in code** (`agent.writing_warnings`, logged as `writing_warnings`; the text is
still used, so they are evidence for the evaluation, not a block): raw field names instead of
readable names with units, safety or protective-equipment steps whose words are not in the data,
and a root_cause order whose first step is an approval or review instead of a diagnostic check.

**Offline tests** (`uv run python tests/test_agent.py`, 26 tests, fake Gemini, no quota) [Measured]:
normal (low risk, 1 call), high (3 calls, tools, grounded ids), critical (review), out of
training range (review), malformed input and three prompt-injection inputs (refused, 0 calls), a
hijacked model answering "P4 / monitor" at critical risk (rejected → rules), an invented part id
in the decision (rejected → rules) and in the text (degraded), money in the text (degraded), a
zero-stock part (review), invalid JSON (rules), no key (degraded, 0 calls), repeated 429 (2
retries with bounded waits, then degraded), one 429 then success, daily budget used up (also
after a simulated restart, counted from the logs), the rate limiter, read-only cache miss
(explicit), cache entries without the key, token accounting without double-counting thinking, and
the style switch: identical decision fields in the three styles, different text, calls 1–2 shared;
the writing rules (stated in the prompts, detected in code, logged without degrading); and a cache key
that changes when a prompt's text changes even if PROMPT_VERSION does not.
The 41 bandit tests, 14 tool tests and 12 economics tests still pass.

## Cost and latency (component B)

**Prices** (Gemini API pricing page, Standard paid tier, read 27 Nov 2026): `gemini-3.5-flash-lite`
$0.30 per 1M input tokens and $2.50 per 1M output tokens, *thinking tokens included in the output
price*. We use the free tier, so the money really paid is 0 €; the logged `cost_eur` is the
production-equivalent cost at the paid price, converted with the dated ECB rate in
`economics.py` (1.0989 USD/EUR, 17 Nov 2026).

**Accounting.** Tokens come from the response's usage metadata: input (prompt), output
(candidates) and thinking (thoughts) are logged separately and thinking is never counted twice
(a test covers both ways the API can report it). Cache hits cost nothing and log 0 tokens; their
original token counts are kept in `llm_live_equivalent`, so an evaluation run from the cache
still reports the live cost.

**Per request** [Measured, live smoke test of 27 Nov 2026, one run per row, local machine; the
evaluation run on eval/cases.csv will give the averages]:

| Path | Gemini calls | Tokens in / out / thinking | € (paid equivalent) | Latency |
|---|---|---|---|---|
| Low risk (follow-up note) | 1 | 791–1,079 / 42–55 / 0 | 0.00034–0.00040 | 2.4–3.1 s |
| Critical (tools + decision + writing) | 3 | 3,662–4,229 / 259–357 / 178–238 | 0.0022–0.0024 | 2.7–3.5 s |
| Same reading, another style, calls 1–2 from the cache (evaluation only) | 1 | 1,344–1,372 / 107–171 / 0 | 0.0006–0.0008 | 0.9–1.2 s |
| Degraded or out of scope | 0 | 0 | 0 | [To measure] |

Thinking tokens appeared only in call 1 (choosing tools); calls 2 and 3 used none.

Monthly cost at the business-case volume (1,000 processes/day × 30 days), [Assumption] on the
mix: between about 10 € (all low risk) and about 65 € (all 3-call) at the paid price; e.g. 90 % low
risk → about 16 €/month. [Pending] the real share of low-risk processes from component A on
validation data. **Quota:** at that volume the agent would need well over 1,000 Gemini requests a
day, above the free tier's daily limit, so production would need the paid tier: the free tier
only covers the demo and the evaluation.

**Quota protection.**
- Rate limit: at most 12 Gemini requests in any 60 s, in-process. A request that would wait more
  than 20 s degrades instead.
- Daily budget: `DAILY_LLM_BUDGET` (400 on Render), counted in Gemini requests (calls plus
  retries) per Pacific-time day, when the free-tier quota resets. It is rebuilt from the logs
  (Postgres on Render) when a process starts, so a restart does not reset it.
- 429/503: at most 2 retries, waiting the server's `retryDelay` or an exponential backoff, capped
  at 8 s each; then degraded mode.
- [Pending] Free-tier limits: the official rate-limits page does not publish per-model numbers
  ("can be viewed in Google AI Studio"). Our plan assumed 15 RPM and 500 requests a day for
  Flash-Lite; to be confirmed in AI Studio for the team's project before the demo.

**Cache.** `eval/llm_cache/`, one JSON file per call, keyed by a SHA-256 of the model, prompt
version, style (none for calls 1–2), call name, system prompt, input, generation parameters,
tool declarations and response schema. Modes (`LLM_CACHE`): `readwrite` (evaluation runs),
`readonly` (default locally: a miss fails explicitly and is logged as degraded, no call is made),
`off` (Render: live traffic never comes from the cache). Entries hold only the model's output and
token counts, never the API key.

## Live smoke tests (27 Nov 2026) [Measured]

**Run 1.** `LLM_CACHE=off`, budget cap 12, logs in a temporary folder. 9 Gemini calls, 0 retries, 0 rate
limits, 0 degraded, 0 validation errors.

| Case | Style | Priority · action · mode | Parts · window | Review | Calls | € | Latency |
|---|---|---|---|---|---|---|---|
| Low risk | checklist | P4 · monitor · none | – · – | no | 1 | 0.00034 | 2.9 s |
| Critical HDF | cost | P1 · cooling_check · HDF | P-CL-010 · none | yes | 3 | 0.00216 | 3.0 s |
| Critical HDF | checklist | P1 · cooling_check · HDF | P-CL-010 · none | yes | 3 | 0.00216 | 2.7 s |
| Critical HDF | cost (calls 1–2 cached) | P1 · cooling_check · HDF | P-CL-010 · none | yes | 1 | 0.00062 | 0.9 s |
| Critical HDF | root_cause (calls 1–2 cached) | P1 · cooling_check · HDF | P-CL-010 · none | yes | 1 | 0.00076 | 1.2 s |

- **Style invariance.** The decision fields were identical in the three styles, and the texts
  differed as each style asks (short imperative checklist; cost ratio and downtime first;
  indicators and mechanism).
- **Two independent live runs of the same reading** (rows 2 and 3, no cache) gave the same
  priority, action, mode, part, window and review flag, but a differently worded `review_reason`
  and a different set of tool calls (one run also looked up TWF parts). Temperature 0 is not fully
  deterministic. Within one request the style still cannot change the decision (calls 1–2 do not
  see it); across requests, small differences come from the model, not from the style.
- **No window.** For "now" the model asked for a 2-hour window; the only window in the next 8 h
  lasts 1 hour, so none was returned and the order has no window. That is correct with the
  simulated calendar, and the order goes to review anyway (critical).
- **Quality notes.** The low-risk note added a safety step that is not in the plant data ("wear
  safety glasses"). The root_cause text quoted raw field names ("temp diff k of 7.9", "power w of
  5497.8") and its first step was an approval, not a diagnostic step. No truncated, repeated or
  looping answers at temperature 0. The prompts were fixed (still `agent-v1`: never deployed, no
  ratings under it) and the rules are now also checked in code (`writing_warnings`).

**Run 2, after the prompt fix.** `LLM_CACHE=off`, budget cap 4. 4 Gemini calls, 0 retries, 0 degraded.

| Case | Style | Priority · action · mode | Parts · window | Review | Calls | Tokens in / out / thinking | € | Latency | Warnings |
|---|---|---|---|---|---|---|---|---|---|
| Low risk | checklist | P4 · monitor · none | – · – | no | 1 | 923 / 42 / 0 | 0.00035 | 2.4 s | raw field name "tool wear min" |
| Critical HDF | root_cause | P1 · cooling_check · HDF | P-CL-010 · none | yes | 3 | 4,229 / 357 / 190 | 0.0024 | 3.5 s | none |

- root_cause now starts with a diagnostic step ("Measure the temperature difference (process minus
  air) of 7.9 K…"), puts the supervisor review second, and uses readable names and units
  ("failure probability of 91.02 %"). The decision fields equal those of run 1.
- The low-risk note no longer invents a safety step, but it named an indicator by its raw field
  name ("tool wear min"). The code check caught it live and logged it as a warning.

**Run 3, readable names for every style.** The table of readable names and units moved from
`root_cause.md` to `base.md` (rule 8), so it applies to all styles and to the low-risk note (still
`agent-v1`; final prompts). `LLM_CACHE=off`, budget cap 1: low risk, checklist, 1 call,
1,079 / 46 / 0 tokens, 0.00040 €, 3.1 s, not degraded, **no warnings**. Steps: "Keep normal
operation." / "Re-check at the next reading." / "Watch the failure probability of 0.49 %."
Justification: "The risk is low, so monitoring is enough."

## Failures (component B) [Pending: fill in from the evaluation]

How each failure shows up in the logs, so the evaluation can group them by cause:

| Cause | Where it shows | Effect on the user |
|---|---|---|
| Gemini unavailable (key, budget, rate limit, 429, timeout, cache miss) | `degraded_reason` starts with `llm_unavailable` | template order, "Degraded mode" banner |
| Invalid decision (schema, band/mode inconsistency, invented id) | `decision_source = rules_fallback`, `decision_errors` | rule-based decision, LLM text |
| Invalid text (money, invented id, schema) | `degraded_reason` starts with `writing_failed` | LLM decision, template text |
| Hand-off A → B (e.g. a mode the tools do not cover) | `validation_errors`, `failed_validation` review | sent to a person |

## Versions

| Item | Version |
|---|---|
| Agent prompts | `agent-v1` (`maintai/prompts/`) |
| Agent model | `gemini-3.5-flash-lite` (stable) |
| Cache format | `llm-cache-v1` |
| Logged as | `versions.component_b = "gemini-3.5-flash-lite\|agent-v1"`; `none\|agent-v1` when no call was made |

## Notes for other work packages

- **WP2 (economics).** The price used in `economics.PRICE_USD_PER_M` matches the official page
  ($0.30 / $2.50, output includes thinking). The FX rate is already dated (17 Nov 2026); refresh it
  only if the team wants a rate closer to the submission date. The judge model
  (`gemini-3.8-flash`) has no price entry, so its cost would show as 0; its paid price is
  $0.75 / $3.75 until 31 Dec 2026 and $1.50 / $7.50 from 1 Jan 2027.
- **WP4 (evaluation).** Run the evaluation once with `LLM_CACHE=readwrite` to fill
  `eval/llm_cache/`, then commit the cache so that later runs (and the professor, with
  `readonly`) reproduce it without quota. Use `llm_live_equivalent` for the cost per request.
  The arm-free decision is in `outputs.component_b.decision`.
- **WP6 (bandit).** The requirement in `docs/WP6_BANDIT.md` ("Instructions for WP3") is in place.
  `tests/test_bandit.py::test_adversarial_agent_cannot_change_rule_based_safety` still passes, but
  its last assertion documents the old gap; with the real agent the action no longer depends on
  the arm (see `tests/test_agent.py::test_arm_changes_writing_but_never_the_decision`).
- **WP7 (UI).** New review reason `part_out_of_stock`: add a label to `REVIEW_REASONS` in
  `ui/components.py` (today it shows the raw key).

# Real feedback campaign: pre-registered protocol (WP6)

**Status: pre-registration, written on 27 Nov 2026, before any eligible real rating exists.** Owner: Manuela (WP6). This file fixes the target, the rules and the analysis *before* the campaign, so the results cannot shape them. Any change made after the campaign starts is added at the end under "Deviations", with the date and the reason. Nothing above that heading is edited once the campaign starts.

**Not started.** The campaign needs Manuela's explicit authorisation to spend Gemini quota and the prerequisites in section 2. No rating has been or will be generated artificially.

## 1. Question and what we accept in advance

- **Question.** Among the three writing styles of a work order (`checklist`, `cost`, `root_cause`), which do raters find useful most often, and does the live Thompson-sampling policy move towards it?
- **What the campaign can show.** That the loop feedback → reward → update → behaviour works on real logs, and how the preference of *these raters* looks. Raters are team members and classmates, not maintenance technicians. The result is not a claim about real users.
- **No winner is an acceptable result.** With about 150 ratings and three styles, a clear preference may not appear. If it does not, we report "no clear preference" with the intervals. We do not extend the campaign, change the prompts or re-run scenarios to produce a winner.

## 2. Prerequisites (all must hold before the first rating)

| # | Prerequisite | Owner | How we check it |
|---|---|---|---|
| P1 | Real risk model merged and deployed (`component_a` version does not start with `stub`) | WP1 | Monitoring banner no longer lists the risk model as a stub. **Met on 27 Nov 2026** |
| P2 | Real agent merged and deployed, with one prompt per style and a fixed prompt version (no `stub`) | WP3 | `versions.prompt` in a test record |
| P3 | The style cannot change the action, the priority or the review decision (the safety gap recorded in `docs/WP6_BANDIT.md`) | WP3 | The adversarial-agent test passes for the real agent |
| P4 | Final scenario file: `eval/select_rl_scenarios.py` re-run with the real model, `status = final` in `eval/rl_scenarios.csv` | WP6 | The script prints `status = final`. **Met on 27 Nov 2026**; re-check if the model version changes |
| P5 | `session_tag`, `arm`, `arm_probs` and feedback rows persist in PostgreSQL on Render | Manuela | `docs/INTEGRATION_CHECKLIST.md`, section 6 |
| P6 | `check.py` passes on a clean clone; `check.py --url` passes against Render | All | `docs/INTEGRATION_CHECKLIST.md`, section 7 |
| P7 | Gemini budget approved by Manuela (section 8) | Manuela | Written approval in the team chat |
| P8 | Prompts frozen for the duration of the campaign | WP3 | A new prompt version starts a new policy and discards the earlier ratings |

## 3. Participants

- **Who.** Team members and external classmates (other IE students). No real technicians are available.
- **Target.** 10 rating sessions × 15 ratings = **150 eligible ratings**, subject to the quota in section 8. For example: 5 team members and 5 external classmates, one session each.
- **No personal data.** We do not record names, e-mails or devices. The coordinator gives each participant a number (1, 2, 3, …) only to generate their scenario order; the number is not entered in the app and is not stored with the ratings. The app logs only a random, temporary session tag (`docs/WP6_BANDIT.md`, "Abuse protection").
- **Consent.** Participation is voluntary. Each participant is told that their ratings are logged without names or personal data, under a random temporary session tag, and reported only as aggregates in a university project report.
- **Team vs external.** Team members know how the system works and may favour a style. We cannot separate their ratings in the logs (no identity). Where possible, team sessions and external sessions run in different, noted time windows, and the report states the share of sessions from each group. This is a recorded limitation, not a correction.

## 4. Instructions for participants (to be read aloud or sent as is)

> 1. Open https://maintai-238a.onrender.com in a normal browser window. Use one window for the whole session. Do not reload the page or open a private window during the session.
> 2. You will receive 15 machine readings. Enter them in the order given, one at a time, in the form of the App tab (type and the five values), and press **Analyse machine**.
> 3. Read the work order as if you were the maintenance manager who has to approve it. Ask yourself: *would this order help me decide and act?*
> 4. Rate **every** order, even the ones you find average: 👍 **Useful** or 👎 **Not useful**. Do not skip any.
> 5. Rate each order once. If you click by mistake, you may correct it once; only your last rating counts.
> 6. Rate on your own. Do not discuss the orders with other participants until everyone has finished, and do not look at other screens.
> 7. The style of the order is shown only after you rate. Do not try to guess it or to favour a style: rate the order in front of you.
> 8. If something looks wrong (an error, an empty order, an order that contradicts the risk), rate it honestly and tell the coordinator afterwards.

## 5. Scenarios

- **Source: the validation split only** (`data/validation.csv`, WP1). **Never the test split**, which is reserved for the final evaluation. The script reads only the validation file and the check in section 10 confirms it.
- **Selection (reproducible).** `uv run python eval/select_rl_scenarios.py` (fixed seed 20261128) writes `eval/rl_scenarios.csv` with 30 readings:
  - 10 that failed, at least one per failure mode where the split has one (TWF, HDF, PWF, OSF; RNF failures are random by design);
  - 20 that did not fail, covering the machine conditions defined by the AI4I physical rules (tool near end of life, near overstrain, heat-dissipation zone, power out of band, normal) and the three product types.
- **Risk bands.** Read from the real component A (`maintai-gb-sigmoid-v1`, merged on 27 Nov 2026). The file is `status = final`: 18 low, 2 medium, 4 high and 6 critical readings. The validation split has only 2 medium and 4 high readings in total, and all of them are included; "at least 4 per band" therefore holds only where the split allows it. If WP1 changes the model before the campaign, the script is re-run (the file records the model version) and the new file is committed before the first rating. Once the campaign starts, the file is not changed.
- **Random order.** Each participant gets a different random order of 15 of the 30 readings: `uv run python eval/select_rl_scenarios.py --order <number>`. The sheet shows only the six input values, never the label or the failure mode, so raters are not told the right answer.
- **Coverage.** With 10 participants each reading is rated about 5 times, by different people, under different styles.

## 6. Eligibility and exclusion (as coded in `maintai/bandit.py`)

A rating counts towards the policy and the analysis only if its request:

1. was served live by the bandit (`source = real`, with logged `arm_probs`);
2. was handled (not out of scope) and not degraded (no template fallback);
3. has no component version starting with `stub`;
4. used the current prompt version and a compatible policy version (`ts-v1`);
5. is among the first 15 rated requests of its session (session cap).

Only the latest rating per request counts. Excluded records stay in the logs and are counted by reason on the dashboard ("Why ratings are excluded") and in `exclusions.csv`.

**Outside the campaign window.** Requests from `check.py --url`, the professor, demos and stray visitors are also real traffic. Their ratings (if any) count like any other under the rules above; their unrated requests lower the overall response rate. The campaign analysis therefore also reports the response rate inside the campaign window.

## 7. Bias, duplicates and non-response

| Risk | Control | What we report |
|---|---|---|
| Raters guess or favour a style | Rating blind to the style; style revealed after the vote; instructions 6–7 | Limitation: raters are not technicians |
| Raters influence each other | Independent sessions, no discussion until all have finished | — |
| Order and fatigue effects | Random order per participant; 15 readings per session (about 15 minutes) | — |
| One person dominates | Session cap of 15 counted ratings; 3 clicks per order, 60 per session | Ratings over the cap, sessions with ratings |
| Duplicate or changed ratings | Only the latest rating per request counts | Number of changed ratings |
| Careless clicks | — | Ratings under 3 s after the order (assumption), median and p95 reward delay |
| Non-response | Instruction 4 (rate every order); coordinator checks that each session has 15 ratings | Response rate overall, by risk band, decision and style; flag at a gap of 20 points or more |
| Feedback loop | Thompson keeps serving every style with its selection probability; logged probabilities | Served selection probabilities; exploratory IPS estimate (not causal, does not correct non-response) |
| Scenario mix differs between styles | Every reading is rated several times; random order | Risk mix per style |

## 8. Gemini budget (to be approved by Manuela before the campaign)

The free tier allows about **500 requests a day** to Gemini Flash-Lite; Render runs with `DAILY_LLM_BUDGET=400` and `LLM_CACHE=off`, so every campaign request is a live call.

| Quantity | Value | Source |
|---|---|---|
| Target eligible ratings | 150 | This protocol |
| Requests needed (10% margin for errors, out-of-scope typos and unrated orders) | about 165 | Assumption |
| Gemini calls per request | *k*, at most 3 | WP3 design (to confirm with the real agent) |
| Retries (429, timeouts) | up to 10% extra calls | Assumption, to confirm with WP3's client |
| **Campaign calls** | 165 × *k* × 1.1 ≈ **182 (k = 1) to 545 (k = 3)** | Formula |
| Reserve kept every day | 100 calls (check.py, professor, demo, evaluation re-runs) | Decision |
| Campaign calls allowed per day | 400 − 100 = **300** | Formula |

So with *k* = 1 the campaign fits in one day; with *k* = 3 it needs two days (for example Saturday and Sunday morning). The evaluation (WP4) uses its own quota and is planned separately. Before starting, WP3 confirms *k* and the retry policy, and Manuela approves the number of days.

## 9. Stopping rules (fixed now)

The campaign **stops** at the first of:

1. **Target reached:** 150 eligible counted ratings.
2. **Deadline:** Sunday 29 Nov 2026, 18:00 (before the freeze at 22:00), whatever the count.
3. **Daily budget:** the day's 300 campaign calls are used; it resumes the next day only if the deadline allows.

It is **paused** (and resumed only after the cause is fixed and noted under "Deviations") if:

4. any order shows an action, priority or review decision that changes with the style, or an out-of-scope input produces an order (a safety failure);
5. more than 20% of the last 20 requests are degraded (quota or API errors);
6. ratings stop appearing in the logs (persistence failure).

**If the target is not reached.** 150 is a target, not a requirement: quota, time or the number of participants may stop the campaign earlier. In that case we report the number of eligible ratings reached and the reason, and we run the same analysis (section 11) on what we have. With fewer ratings a "clear preference" is less likely, because it also needs at least 30 counted ratings for that style. We do not lower the thresholds, add simulated ratings or pool ratings from stub or degraded requests to make up the difference.

It is **never** stopped or extended because a style looks better or worse. During the campaign the coordinator watches only the operational numbers (eligible ratings, response rate, degraded share, errors), not the per-style results.

## 10. Evidence to export

At the **start** (before the first rating) and at the **end** of the campaign:

```bash
uv run python main.py export-logs                       # LOGS_DATABASE_URL in .env, never committed
uv run python eval/export_rl_evidence.py --logs-dir logs --out eval/rl_evidence/<start|end>
```

- `manifest.json`, `metrics.json` and the CSVs (aggregates only: no ids, inputs, session tags or single timestamps), each labelled `measured` or `simulated`;
- screenshots of the Adaptive Intelligence dashboard (desktop width);
- the campaign sheet: date and time windows, number of sessions (team / external), scenario file hash, deviations. No names.

Only the aggregated folders are committed. `logs/` is never committed.

## 11. Analysis: policy before and after

| Quantity | Before (start export) | After (end export) |
|---|---|---|
| Posterior per style | Beta(1, 1): mean 50%, 95% interval 3%–98% | Beta(1 + positive, 1 + negative) |
| Selection probability | about 33% each | From 4,000 posterior draws |
| Observed reward per style | — | Share of positive eligible ratings, with counts |
| Exploratory IPS estimate | — | Self-normalised, from the logged `arm_probs` |
| Policy history | — | `policy_history.csv`: posterior after each counted rating |

**Pre-registered reading of the result.**

- **Clear preference:** one style has a probability of being the best of at least 0.90 at the end, with at least 30 counted ratings for that style.
- **Tentative preference:** that probability is between 0.70 and 0.90. We report the direction and say the evidence is weak.
- **No clear preference:** below 0.70. This is reported as a valid result.

In every case we report the 95% credible intervals, the response rate, the reward delay, the non-response checks, the exclusions and the limitations. The intervals are a guide, not a guarantee: adaptive sampling bends their coverage (`eval/bandit_calibration.py`, simulated). Learning shows that the loop works on real logs. It does not show what technicians would prefer, and it does not show that the preferred style produces more *correct* orders: correctness is judged separately by the evaluation (WP4).

## Deviations

*None yet.*

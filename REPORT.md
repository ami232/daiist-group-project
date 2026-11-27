# Group Assignment Report

- **Product**: MaintAI – Intelligent Maintenance Advisor
- **Team** (Blackboard group name):
- **Section**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A]
- **App URL** (on Render): https://maintai-238a.onrender.com
- **Final commit** (tag `final`):

*At most about 3,000 words, tables excluded. Members and who did what go in CONTRIBUTIONS.md.*

## Business case

> **Pre-registered on 25 Nov 2026, before any model was trained** (commit `1d0135f`). Every number is a dataset fact, an economic assumption or a metric still to be measured, and is labelled. The assumptions and criteria were approved by Manuela and are pending confirmation by the rest of the team. The full version is in [docs/BUSINESS_CASE.md](docs/BUSINESS_CASE.md). The success criteria below are copied **unchanged** from that commit.

| Label | Meaning |
|---|---|
| **[Fact]** | Checked in the dataset documentation or in the repository |
| **[Assumption]** | A value chosen to make the economics concrete. Not taken from a published source. Tested in the sensitivity analysis |
| **[To measure]** | Will be measured in the evaluation. No value yet |
| **[Simulated]** | Produced by a simulator, not by real use of the app |

**Value proposition.** MaintAI helps the maintenance manager of a machining plant decide, for each machine and process, whether to do nothing, schedule an inspection or stop and intervene. It turns that decision into a work order ready to approve: action, spare parts, maintenance window and cost, checked against the plant's (simulated) stock, calendar and policies. The aim is fewer unplanned breakdowns without paying for many unnecessary inspections. It is decision support: it does not control machines or predict when a failure will happen, and every critical or uncertain case goes to a person.

| | AS-IS (today, without MaintAI) | TO-BE (with MaintAI) |
|---|---|---|
| Trigger | Fixed tool-life rule and single-sensor alarms with fixed limits | Calibrated failure risk for every reading, plus the most likely failure mode |
| Work order | Written by a technician, who checks stock and calendar in separate systems | Drafted by an agent that queries stock, maintenance windows, costs and policies, then validated by code |
| Human role | Decides everything, usually after an alarm | Approves orders. Reviews every critical, uncertain or out-of-range case |
| Learning | None | Technicians rate orders (👍/👎). A bandit learns which work-order style they find most useful. Safety rules never change |

**Baselines.** They get the same data and cost model as MaintAI, with thresholds tuned on the training and validation splits, so the bar is not easy to beat. The work-order baseline is the rule-based template used in degraded mode.

| Policy | Rule | Role |
|---|---|---|
| **B0 · Reactive** | Never intervene. Repair after a failure | Reference only, to show the cost of doing nothing. Not the baseline to beat |
| **B1 · Tool-life rule** | Intervene when tool wear ≥ *T* minutes. *T* is tuned to minimise expected cost | Common practice in machining |
| **B2 · Condition-monitoring rules** | Intervene if any single sensor crosses its limit: tool wear, torque, low rotational speed or a small air–process temperature difference. Limits tuned to minimise expected cost | A realistic alarm system without machine learning |
| **Main baseline** | The better of B1 and B2 on the validation split | The bar MaintAI has to clear |
| **Oracle (not a baseline)** | The rules that generated the synthetic dataset | Upper bound only. No real plant knows these rules, so beating them is not expected |

| Type | KPI | Status |
|---|---|---|
| Technical (component A) | PR-AUC, recall and precision of failures at the chosen threshold, Brier score (calibration) | [To measure] on the test split |
| Technical (component B) | Valid work orders, grounding (no invented part, window or cost), correct action for the case, correct review flag, judge pass rate, judge agreement with our own labels | [To measure] |
| Economic | Expected cost per 1,000 processes compared with the main baseline; missed failures and unnecessary interventions per 1,000 | [To measure] with the cost model below |
| Operational | Cases sent to review per day, p50/p95 latency, tokens and € per request, share of requests in degraded mode, share of 👎 | [To measure] in evaluation and in the live logs |

### Economic model

Policies are compared by their **expected cost per 1,000 processes** (one process is one row of the dataset).

| Symbol | Meaning | Working value |
|---|---|---|
| *C_I* | Cost of one preventive intervention: inspection, planned work and planned downtime | €300 [Assumption] |
| *C_F* | Cost of one unplanned failure: repair plus unplanned downtime | €4,000 [Assumption] |
| *e* | Probability that a preventive intervention really prevents a failure that would otherwise happen | 0.8 [Assumption] |
| *C_R* | Cost of one human review (about 15 minutes of a supervisor) | €15 [Assumption] |
| *V* | Processes (machine cycles) assessed per day | 1,000 [Assumption] |
| *C_LLM* | Cost of the AI per request, at the paid Gemini price | [To measure] from logged tokens |

| Outcome | What happens | Cost |
|---|---|---|
| True positive | Intervention on a process that would fail | *C_I* + (1 − *e*) · *C_F* |
| False positive | Unnecessary intervention | *C_I* |
| False negative | Failure not caught | *C_F* |
| True negative | Nothing | 0 |
| Any case sent to review | A supervisor checks the order | + *C_R* |
| Any MaintAI request | Model and API cost | + *C_LLM* |

An intervention does not always prevent the failure, hence the term (1 − *e*) · *C_F*. Ignoring review and AI costs, intervening pays off above *p\** = *C_I* / (*e* · *C_F*) = 300 / (0.8 × 4,000) ≈ 9.4%, which is why component A must give **calibrated** probabilities. The final threshold is chosen on the validation split with the full cost model (review cost, measured AI cost, review band). Savings are converted to euros per month at an assumed **1,000 processes a day** [Assumption], net of review and AI costs.

| Assumption | Range tested |
|---|---|
| *C_F* / *C_I* ratio | 5× to 50× |
| *e* (prevention effectiveness) | 0.5 to 1.0 |
| *C_R* (review cost) | €0 to €50 |
| Daily volume | 200 to 5,000 processes |

For each setting, thresholds are re-tuned only on the training and validation splits, and the test split is used once. Plant-level costs come only from the test split, which keeps the natural failure rate. `eval/cases.csv` over-represents hard cases on purpose and is never extrapolated to the plant.

### Success criteria and go / no-go (set before seeing results)

These thresholds are fixed now so that the decision cannot be adjusted to fit the results. They are preliminary targets, approved by Manuela and pending confirmation by the rest of the team, not measurements. Criteria 1 and 2 are measured on the test split; criteria 3–7 on `eval/cases.csv` and, for operations, also on the live logs.

The seven criteria:

1. **Cost reduction.** At the working values, MaintAI's expected cost per 1,000 processes is **at least 10% lower** than the main baseline's (the better of B1 and B2), and the 95% bootstrap interval of the saving stays above zero. MaintAI must also stay cheaper for most of the sensitivity grid, at least for *e* ≥ 0.6 and *C_F* / *C_I* ≥ 10.
2. **Failure recall of at least 0.80**, and never lower than the main baseline's. A missed failure costs about 13 times as much as an unnecessary intervention (€4,000 against €300), so the product should catch most failures. We do not ask for more than 0.80 because some failures cannot be predicted from the inputs: random failures (RNF) by design, and tool-wear failures happen at a random point between 200 and 240 minutes [Fact]. We do not buy savings with more missed failures.
3. **Safety and review rules are always applied.** In 100% of the evaluation cases where a deterministic rule requires review (critical risk, out-of-range input, uncertainty band, failed validation), the final output is sent to a person. The bandit never changes the required action or the review decision.
4. **Zero ungrounded work orders** after validation: no invented part, window or cost.
5. **Work-order quality.** LLM judge pass rate of at least 80% and higher than the template baseline. Judge agreement with our own labels of at least 0.6 (Cohen's kappa).
6. **Out of scope.** Every out-of-scope input is refused before any model call.
7. **Operations.** p95 latency under 15 s, AI cost under €0.01 per request, and a drift alert that fires on shifted inputs while staying quiet on normal ones.

**What happens if a criterion is not met.** Each failure has a fixed outcome. The final decision is the **worst outcome** across the seven criteria.

| Criterion | Not met when | Outcome |
|---|---|---|
| 3 · Safety and review | Even one case that a rule sends to review is not sent to review, or the bandit changes a required action | **No-go.** Not acceptable under any condition |
| 4 · Grounding | Even one work order contains an invented part, window or cost | **No-go** |
| 6 · Out of scope | An out-of-scope input produces a work order | **No-go** (this is a safety failure) |
| 6 · Out of scope | The input is refused, but only after a model call | Go with conditions: fix the validation before the model call |
| 2 · Recall | Recall below 0.80 or below the main baseline | **No-go** |
| 1 · Cost | MaintAI is not cheaper than the main baseline, or the 95% interval of the saving includes zero | **No-go** |
| 1 · Cost | Cheaper with an interval above zero, but by less than 10% | Go with conditions: confirm the saving in a pilot before scaling |
| 1 · Cost | At least 10% cheaper at the working values, but not for most of the sensitivity grid | Go with conditions: measure the real costs of the plant first |
| 5 · Quality | Judge pass rate below 80%, or not better than the template | Go with conditions: every work order approved by a person, or use the template |
| 5 · Quality | Judge agreement below 0.6 kappa | Go with conditions: quality judged only on our own labels, and the judge is not used in monitoring |
| 7 · Operations | Latency, cost or drift alert targets missed | Go with conditions: fix before a pilot |

**Go with conditions** also applies when all seven criteria are met, because the evidence comes from synthetic data. The first condition will always be a shadow pilot on real plant data, with human approval of every high-priority order. An unconditional **go** would only be possible after that pilot confirms the results.

### Limitations

- **Synthetic data** [Fact]. AI4I 2020 failures come from a few known rules, so results may not transfer to a real plant.
- **No time to failure** [Fact]. Each row is a snapshot: MaintAI estimates the risk that this process fails, not when a machine will fail.
- **Few failures** [Fact] (3.4% of rows). Test metrics are uncertain, so we report bootstrap intervals.
- **Random failures** [Fact]. RNF cannot be predicted by design.
- **Assumed costs and simulated plant systems.** Savings are model outputs, not observed savings.
- **Feedback** comes mainly from the team and classmates, not technicians: learning shows that the loop works, not what technicians prefer.

## Architecture

*The two components and how one feeds the other: a diagram plus a short walk-through of one request from input to output. Which models or APIs, and why those. What you reused from assignments 1–3, and whose it was.*

## Data

*Where each component's data comes from, with links, licences and sizes. Confirm that there is no personal data in your inputs or logs.*

## Evaluation

### Components

*Each component on its own: the metric, the result, the test data and its size.*

### End to end

*How you wrote `eval/cases.csv` (counts by kind, two examples). Results against the baseline. Where the output is free text: the judge's rubric and your agreement with it on your own labels.*

### Cost and latency

*Per request: tokens, euros and seconds, for each component and in total. What it would cost per month at the volume in your business case.*

## Operations

*What you log. The quality signal and the drift check you monitor, the reference set, and the alert threshold with its justification. The drift demo: what you shifted, and the result with the alert firing (screenshot or numbers). The rate of false alarms on normal inputs.*

**Deployment (25 Nov 2026).** The app runs on Render's free plan at https://maintai-238a.onrender.com, created from `render.yaml` as a Blueprint: one web service and one free PostgreSQL database for the logs, connected through `DATABASE_URL`. `GEMINI_API_KEY` is stored only in the service's environment variables. On Render, `LLM_CACHE=off`, so live requests never come from the LLM cache, and `DAILY_LLM_BUDGET=400` keeps a margin below the free tier's 500 requests a day.

**Records excluded from analysis.** The first deployment still used stub components. The 3 requests and 2 👍 logged on 25 Nov 2026 were smoke tests of the deployment (confirmed in the PostgreSQL export of 27 Nov 2026). They are not evidence for the evaluation or for the bandit's learning. They can be identified by their versions, which start with `stub-` (for example `component_a = stub-a-v0`), and every analysis and the bandit exclude records with stub versions.

## Failures

*Failed end-to-end cases grouped by cause (which component, or the hand-off between them), with counts, and what you would fix first.*

## Decision

*Which cases go to a person: the rule, and how many cases a day that would be. Go / no-go / go with conditions: the conditions, and the evidence behind them.*

## Bonus: reinforcement learning

**Status (27 Nov 2026): implemented, tested and deployed; no real learning evidence yet.** Ratings only count once the real risk model (WP1) and the real agent (WP3) are live, so we do not claim the bonus without evidence from real logs. Details: [docs/WP6_BANDIT.md](docs/WP6_BANDIT.md); pre-registered campaign: [docs/RL_FEEDBACK_PROTOCOL.md](docs/RL_FEEDBACK_PROTOCOL.md).

**What learns.** A Thompson-sampling bandit chooses the *writing style* of each work order (`checklist`, `cost`, `root_cause`). Only the agent receives it; the risk model and the safety rules never do.

**The loop.**
1. **Feedback.** The user rates the order blind to its style.
2. **Reward.** Useful = 1, not useful = 0; the latest rating counts; no rating, no reward.
3. **Update.** Beta(1 + positive, 1 + negative) per style, rebuilt from the logs (PostgreSQL) on every request.
4. **Behaviour.** One sample per posterior; the highest wins. Exploration comes from the same rule.

Each request logs the style, the selection probabilities and the policy version.

**Prior and eligibility.** Neutral Beta(1, 1) (Manuela's decision), so every preference comes from real ratings. A rating counts only for real, handled, non-degraded, non-`stub` traffic with the current prompt version, within the first 15 rated requests of a session.

| Prior | P(best) after 50 | P(best) after 300 | Ratings to 80% | Regret after 100 | Regret after 300 |
|---|---|---|---|---|---|
| Neutral Beta(1, 1) | 82% | 98% | 45 | 7.0 | 9.4 |
| Capped plan prior (10 pseudo-obs.) | 81% | 99% | 50 | 7.5 | 9.3 |
| Wrong prior (favours `checklist` at 90%) | 63% | 99% | 80 | 13.0 | 15.9 |
| Uniform random selection | 33% | 33% | never | 20.0 | 60.0 |

[Simulated] 100 runs × 300 ratings; users like `root_cause` 75% of the time, the others 45%. Regret flattens (about 9 positive ratings lost after 300, against 60 for random). With three equal styles no false winner appears. Simulated users are independent coin flips, unlike real raters. The 95% intervals cover 95–96% when rates come from the prior, about 90% for the best style while it is being found, and much less for a style starved early (31% of 29 intervals, 0% of 16: small samples, kept). They are a guide, not a guarantee.

**Risks and controls.**
- *Reward hacking:* raters may prefer confident text over correct orders. The style never changes risk, action or review, and quality is judged separately.
- *Feedback bias:* raters are the team and classmates, not technicians.
- *Delayed, missing or changed ratings:* response rate and reward delay are monitored.
- *Feedback loop and non-response:* a style with early bad luck is served less. We show logged probabilities, response rates by risk band and style, and an exploratory propensity-weighted (IPS) estimate: not causal, and no correction for non-response.
- *Abuse:* no login (check.py must work). A temporary, pseudonymous session tag without personal data caps counted ratings per session; it is not an identity.

[Measured] 41/41 tests pass; choosing a style takes 3–6 ms with 2,000 logged requests; local peak memory is about 210 MB. Deployed on Render on 27 Nov 2026; memory there is not measured yet.

**Pending.**
1. WP3 must make action, priority and review independent of the style (an adversarial-agent test records the gap).
2. The real campaign: 150 eligible ratings if quota allows. No clear winner is an acceptable result.
3. Persistence of the new fields on PostgreSQL, and the evidence export.

## How to run it

*Anything beyond SETUP.md that the professor needs to run it: which keys, any download step, how long the first run takes.*

## AI use

*To be completed at the end of the project. Kept up to date as we work.*

- **Claude Code (Anthropic):** used to analyse the template repository, compare product ideas, draft the project plan, draft the text of the business case in this report, and help write code. All decisions, assumptions and criteria were reviewed and approved by team members, as recorded in CONTRIBUTIONS.md.
- **ChatGPT (OpenAI):** used by Manuela to review the project plan and to draft the prompts given to Claude Code.
- **Bandit (WP6):** Claude Code helped write the bandit, the simulator, the tests and the dashboard. Manuela chose the neutral prior, the reward, the exclusion rules and the safety scope, and reviewed every result.

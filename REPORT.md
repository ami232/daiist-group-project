# Group Assignment Report

- **Product**: MaintAI – Intelligent Maintenance Advisor
- **Team** (Blackboard group name):
- **Section**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A]
- **App URL** (on Render):
- **Final commit** (tag `final`):

*At most about 3,000 words, tables excluded. Members and who did what go in CONTRIBUTIONS.md.*

## Business case

> **Status: written before any model was trained (25 Nov 2026).** Every number in this section is either a fact about the dataset, an economic **assumption**, or a metric that is **still to be measured**. Each one is labelled. The assumptions and criteria were approved by Manuela on 25 Nov 2026 and are pending confirmation by the rest of the team. Results will be added in later sections and will not overwrite this pre-registration.

Labels used in this report:

| Label | Meaning |
|---|---|
| **[Fact]** | Checked in the dataset documentation or in the repository |
| **[Assumption]** | A value chosen to make the economics concrete. Not taken from a published source. Tested in the sensitivity analysis |
| **[To measure]** | Will be measured in the evaluation. No value yet |
| **[Simulated]** | Produced by a simulator, not by real use of the app |

### Value proposition

MaintAI helps a maintenance manager decide which machines need attention now and turns that decision into a work order that is ready to approve. It estimates the risk that the current process on a machine ends in a failure, explains which failure mode is most likely, and drafts a work order with the action, spare parts, maintenance window and cost, checked against the plant's (simulated) stock, calendar and policies. The aim is fewer unplanned breakdowns without paying for many unnecessary inspections.

### User and decision

- **User:** the maintenance manager (or shift supervisor) of a machining plant.
- **Decision we want to improve:** for each machine and process, *do nothing, schedule an inspection, or stop and intervene*, and with which priority, parts and time slot.
- **Secondary task:** writing the work order. Today this is done by hand, using several separate systems.

MaintAI is a decision-support tool. It does not control machines, it does not predict the exact time of a failure, and every critical or uncertain case is sent to a person before anything happens.

### AS-IS and TO-BE

| | AS-IS (today, without MaintAI) | TO-BE (with MaintAI) |
|---|---|---|
| Trigger | Fixed tool-life rule and single-sensor alarms with fixed limits | Calibrated failure risk for every reading, plus the most likely failure mode |
| Work order | Written by a technician, who checks stock and calendar in separate systems | Drafted by an agent that queries stock, maintenance windows, costs and policies, then validated by code |
| Human role | Decides everything, usually after an alarm | Approves orders. Reviews every critical, uncertain or out-of-range case |
| Learning | None | Technicians rate orders (👍/👎). A bandit learns which work-order style they find most useful. Safety rules never change |

### Baselines

The comparison must be fair, so the baselines get the same data and the same cost model as MaintAI, and their thresholds are **tuned on the training and validation data**, not set by hand. A baseline that is easy to beat would make the result meaningless.

| Policy | Rule | Role |
|---|---|---|
| **B0 · Reactive** | Never intervene. Repair after a failure | Reference only, to show the cost of doing nothing. Not the baseline to beat |
| **B1 · Tool-life rule** | Intervene when tool wear ≥ *T* minutes. *T* is tuned to minimise expected cost | Common practice in machining |
| **B2 · Condition-monitoring rules** | Intervene if any single sensor crosses its limit: tool wear, torque, low rotational speed or a small air–process temperature difference. Limits tuned to minimise expected cost | A realistic alarm system without machine learning |
| **Main baseline** | The better of B1 and B2 on the validation split | The bar MaintAI has to clear |
| **Oracle (not a baseline)** | The rules that generated the synthetic dataset | Upper bound only. No real plant knows these rules, so beating them is not expected |

For the work order itself, the baseline is a **rule-based template**: the same order the app produces in degraded mode, when the LLM is not available. This lets us measure what the agent adds over a fixed template.

### KPIs

| Type | KPI | Status |
|---|---|---|
| Technical (component A) | PR-AUC, recall and precision of failures at the chosen threshold, Brier score (calibration) | [To measure] on the test split |
| Technical (component B) | Valid work orders, grounding (no invented part, window or cost), correct action for the case, correct review flag, judge pass rate, judge agreement with our own labels | [To measure] |
| Economic | Expected cost per 1,000 processes compared with the main baseline; missed failures and unnecessary interventions per 1,000 | [To measure] with the cost model below |
| Operational | Cases sent to review per day, p50/p95 latency, tokens and € per request, share of requests in degraded mode, share of 👎 | [To measure] in evaluation and in the live logs |

### Economic model

We compare policies by their **expected cost per 1,000 processes**. One process is one row of the dataset.

| Symbol | Meaning | Working value |
|---|---|---|
| *C_I* | Cost of one preventive intervention: inspection, planned work and planned downtime | €300 [Assumption] |
| *C_F* | Cost of one unplanned failure: repair plus unplanned downtime | €4,000 [Assumption] |
| *e* | Probability that a preventive intervention really prevents a failure that would otherwise happen | 0.8 [Assumption] |
| *C_R* | Cost of one human review (about 15 minutes of a supervisor) | €15 [Assumption] |
| *V* | Processes (machine cycles) assessed per day | 1,000 [Assumption] |
| *C_LLM* | Cost of the AI per request, at the paid Gemini price | [To measure] from logged tokens |

Cost of each outcome:

| Outcome | What happens | Cost |
|---|---|---|
| True positive | Intervention on a process that would fail | *C_I* + (1 − *e*) · *C_F* |
| False positive | Unnecessary intervention | *C_I* |
| False negative | Failure not caught | *C_F* |
| True negative | Nothing | 0 |
| Any case sent to review | A supervisor checks the order | + *C_R* |
| Any MaintAI request | Model and API cost | + *C_LLM* |

The term (1 − *e*) · *C_F* means that an intervention does **not** always prevent the failure.

**Break-even probability (simplified model).** If we ignore review and AI costs, intervening pays off when the failure probability *p* satisfies *p* · *e* · *C_F* > *C_I*. The break-even probability is then *p\** = *C_I* / (*e* · *C_F*) = 300 / (0.8 × 4,000) = 0.09375, about 9.4%. This explains why component A must give **calibrated** probabilities. It is only a starting point: the final decision threshold will be chosen on the validation split with the **full** cost model, which also includes the review cost *C_R*, the measured AI cost *C_LLM* and the effect of the review band.

**Monthly value.** Savings per 1,000 processes are converted to euros per month with an assumed volume of **1,000 processes per day** [Assumption]. A process is one machine cycle that MaintAI assesses, so 1,000 processes a day may come from far fewer machines. They are not 1,000 different machines. The costs of review and of the AI are included, so the figure is net.

**Planned sensitivity analysis.** The working values are not sourced, so the conclusion must hold across a range, not only at one point.

| Assumption | Range tested |
|---|---|
| *C_F* / *C_I* ratio | 5× to 50× |
| *e* (prevention effectiveness) | 0.5 to 1.0 |
| *C_R* (review cost) | €0 to €50 |
| Daily volume | 200 to 5,000 processes |

For each setting we recompute the cost of MaintAI and of the main baseline and report where MaintAI stops being cheaper. When a setting needs new thresholds, both MaintAI's and the baselines' are re-tuned **only on the training and validation splits, never on the test split**. The test split is then used once per setting to report the cost. The results will be labelled as model outputs under assumptions, not as real savings.

**Where the economic numbers come from.** Costs per 1,000 processes, savings and cases sent to review per day are computed only on the **test split**, which keeps the natural failure rate of the dataset. `eval/cases.csv` is different: we chose its cases on purpose to over-represent hard, review and out-of-scope inputs. It is used to check behaviour by kind of case, not to estimate plant-level costs, and its costs are never extrapolated to the whole plant.

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

- **Synthetic data.** [Fact] AI4I 2020 is synthetic and its failures come from a few known rules. Models may look better than they would on real machines, so results do not transfer directly to a real plant.
- **No time to failure.** [Fact] Each row is a single snapshot with no machine history. MaintAI estimates the risk that *this process* fails under the current conditions, not *when* a machine will fail.
- **Few failures.** Failures are rare, so test metrics will be uncertain. We will report bootstrap confidence intervals.
- **Random failures.** [Fact] One failure mode (RNF) is random by design and cannot be predicted.
- **Assumed costs.** All euro values are assumptions. Savings are model outputs, not observed savings.
- **Simulated plant systems.** Stock, calendar, costs and policies used by the agent are simulated.
- **Feedback.** Ratings in the live app will come mainly from the team and classmates, not from real technicians. Any learning from this feedback shows that the loop works, not what real technicians would prefer.

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

## Failures

*Failed end-to-end cases grouped by cause (which component, or the hand-off between them), with counts, and what you would fix first.*

## Decision

*Which cases go to a person: the rule, and how many cases a day that would be. Go / no-go / go with conditions: the conditions, and the evidence behind them.*

## Bonus: reinforcement learning

*Delete this section if you did not do it. The loop (feedback → reward → update → behaviour), the reward and what it could reward by accident, the exploration strategy and its cost, the evidence from real logs (and which part was simulated), the risks.*

## How to run it

*Anything beyond SETUP.md that the professor needs to run it: which keys, any download step, how long the first run takes.*

## AI use

*To be completed at the end of the project. Kept up to date as we work.*

- **Claude Code (Anthropic):** used to analyse the template repository, compare product ideas, draft the project plan, draft the text of the business case in this report, and help write code. All decisions, assumptions and criteria were reviewed and approved by team members, as recorded in CONTRIBUTIONS.md.
- **ChatGPT (OpenAI):** used by Manuela to review the project plan and to draft the prompts given to Claude Code.

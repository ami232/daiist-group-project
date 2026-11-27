# Group Assignment Report

- **Product**: MaintAI – Intelligent Maintenance Advisor
- **Team** (Blackboard group name):
- **Section**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A]
- **App URL** (on Render): https://maintai-238a.onrender.com
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

### Component A — Machine Learning

Component A estimates the probability of machine failure under the current operating conditions using a Gradient Boosting classifier with physics-informed features and sigmoid probability calibration.

The component receives six machine readings through the `MachineInput` schema: product type, air temperature, process temperature, rotational speed, torque and tool wear. These are transformed into ten features using the same feature engineering pipeline during training and inference.

The selected model was trained using scikit-learn with the following configuration:

| Parameter | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Number of estimators | 100 |
| Learning rate | 0.1 |
| Maximum tree depth | 3 |
| Random seed | 42 |
| Calibration | Sigmoid, five-fold stratified cross-validation |
| Decision threshold | 0.19 |
| Model version | `maintai-gb-sigmoid-v1` |

In addition to the overall failure classifier, four independent binary classifiers estimate the probabilities of tool wear failure (TWF), heat dissipation failure (HDF), power failure (PWF) and overstrain failure (OSF). Random failure (RNF) is excluded because it is unpredictable by design.

The trained models are serialized using joblib and loaded on the first inference request. They are subsequently cached in memory to avoid repeated disk access and do not require retraining during application execution.

The `component_a.assess()` function returns a structured `RiskAssessment` containing:

- Calibrated probability of machine failure.
- Risk classification and economic decision threshold.
- Uncertainty-band indicator.
- Estimated failure-mode probabilities, ranked from highest to lowest.
- Physical indicators derived from machine readings.
- Out-of-distribution (OOD) flags.
- Model version.

This structured assessment is passed to Component B, which uses the predicted risk, physical indicators and possible failure modes to generate an actionable maintenance recommendation supported by the business tools.

The machine learning component applies concepts from the AI Design module, including supervised classification, feature engineering, model comparison, data leakage prevention and cost-sensitive decision thresholds. Model serialization, reproducibility and versioning also apply principles from the MLOps module.

*The two components and how one feeds the other: a diagram plus a short walk-through of one request from input to output. Which models or APIs, and why those. What you reused from assignments 1–3, and whose it was.*

## Data

### Component A — Dataset and preprocessing

Component A uses the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i), published by the UCI Machine Learning Repository under the Creative Commons Attribution 4.0 International license (CC BY 4.0). Dataset DOI: https://doi.org/10.24432/C5HS5C.

The dataset contains 10,000 synthetic industrial observations and 14 columns, representing machine operating conditions and associated failure outcomes. It contains no personal data.

#### Data quality

| Metric | Result |
|---|---:|
| Total observations | 10,000 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Unique observation identifiers | 10,000 |
| Machine failures | 339 |
| Failure rate | 3.39% |
| Non-failure observations | 9,661 |

The dataset is highly imbalanced. A classifier that always predicts no failure would achieve 96.61% accuracy while detecting none of the actual failures. Consequently, accuracy alone is not considered an appropriate model selection metric.

Data quality analysis identified 27 inconsistencies between the overall machine failure label and the individual failure-mode labels: 18 observations with a failure-mode flag but no overall failure and nine observations with an overall failure but no recorded failure mode. The original labels were preserved without modification.

The UDI and Product ID columns were excluded from the input features because they are identifiers. The overall machine failure label and all five individual failure-mode labels were also excluded from the input features to prevent target leakage.

#### Dataset splits

A fixed stratified random 60/20/20 split was generated using random seed 42, stratifying by the overall machine failure target.

| Split | Observations | Failures | Failure rate |
|---|---:|---:|---:|
| Training | 6,000 | 203 | 3.38% |
| Validation | 2,000 | 68 | 3.40% |
| Test | 2,000 | 68 | 3.40% |
| Total | 10,000 | 339 | 3.39% |

All five failure modes are represented in every split. Observation identifiers were checked to ensure that no observation appears in more than one subset.

A chronological split was considered but not selected because all heat dissipation failures occur within the first 6,000 observations, leaving that failure mode absent from the chronological validation and test subsets.

Stratified random sampling provides better representation of failure modes but may produce optimistic generalization estimates because of correlations between sequential observations. This limitation is acknowledged.

The training set is used for model fitting and internal probability calibration. The validation set is used for model comparison, calibration-method selection and economic threshold optimization. The test set is reserved for final evaluation and was not used for any of these decisions.

#### Feature engineering

Two feature sets were compared: the six original machine readings and an expanded set containing four additional physics-informed features.

| Engineered feature | Calculation | Purpose |
|---|---|---|
| Temperature difference | Process temperature − air temperature | Captures heat dissipation conditions |
| Mechanical power | Torque × rotational speed × 2π / 60 | Captures power-related failure conditions |
| Mechanical strain | Tool wear × torque | Captures the combined effect of wear and mechanical load |
| Overstrain ratio | Mechanical strain / product-specific strain limit | Normalizes strain according to product type |

The product-specific strain limits are 11,000 minNm for L, 12,000 minNm for M and 13,000 minNm for H, following the documented AI4I data-generation rules.

These variables are calculated exclusively from observable machine readings, without using the failure labels.

The preprocessing pipeline uses one-hot encoding for the categorical product type and standardization for numerical features. All fitted preprocessing transformations are learned from training data only.

The same feature engineering functions are used during training and live inference to prevent training-serving skew.

The original dataset, documentation and fixed splits are stored in `data/`, and the reproducible preparation and feature engineering scripts are stored in `training/`.

*Where each component's data comes from, with links, licences and sizes. Confirm that there is no personal data in your inputs or logs.*

## Evaluation

### Components
#### Component A — Machine learning evaluation

**Evaluation status:** model selection and integrated validation completed. Final held-out test evaluation pending.

##### Experiment 1 — Algorithm and feature comparison

Three classification algorithms were evaluated using both raw and physics-informed features, resulting in six experiments. All models were fitted on the same training set of 6,000 observations and evaluated on the same validation set of 2,000 observations, including 68 failures.

Logistic Regression was included as a simple ML baseline. Random Forest and Gradient Boosting were selected for their ability to capture nonlinear relationships and interactions between machine readings.

The initial comparison used a conventional classification threshold of 0.5.

| Model | Features | Precision | Recall | F1 | ROC-AUC | Average precision | Brier score |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Raw | 0.7273 | 0.2353 | 0.3556 | 0.8737 | 0.4321 | 0.0249 |
| Logistic Regression | Engineered | 0.6800 | 0.2500 | 0.3656 | 0.8942 | 0.4614 | 0.0242 |
| Random Forest | Raw | 0.8276 | 0.3529 | 0.4948 | 0.9633 | 0.7197 | 0.0174 |
| Random Forest | Engineered | 0.9815 | 0.7794 | 0.8689 | 0.9799 | 0.8696 | 0.0085 |
| Gradient Boosting | Raw | 0.8667 | 0.5735 | 0.6903 | 0.9618 | 0.7672 | 0.0137 |
| Gradient Boosting | Engineered | 0.9643 | 0.7941 | 0.8710 | 0.9654 | 0.8830 | 0.0071 |

Physics-informed feature engineering substantially improved the performance of both tree-based models. Random Forest recall increased from 35.29% to 77.94%, while Gradient Boosting recall increased from 57.35% to 79.41%.

Gradient Boosting with engineered features achieved the highest average precision and F1-score and the lowest Brier score. Random Forest with engineered features achieved the highest ROC-AUC.

Both models were retained for further calibration and economic evaluation. Average precision is reported as the summary measure of precision-recall performance because of the imbalanced target.

Results are stored in `training/model_comparison.csv`.

##### Experiment 2 — Probability calibration

Reliable probability estimates are necessary because maintenance decisions depend on the predicted failure probability and the expected costs of alternative actions.

Random Forest and Gradient Boosting were compared using three probability estimation methods: uncalibrated probabilities, sigmoid calibration and isotonic calibration.

Calibration was performed using five-fold stratified cross-validation within the training set. The ensemble option was disabled to retain a single fitted base estimator per calibrated candidate, reducing deployment complexity.

| Model | Calibration | ROC-AUC | Average precision | Brier score |
|---|---|---:|---:|---:|
| Gradient Boosting | Uncalibrated | 0.9654 | 0.8830 | 0.0071 |
| Gradient Boosting | Sigmoid | 0.9654 | 0.8830 | 0.0071 |
| Gradient Boosting | Isotonic | 0.9644 | 0.8519 | 0.0074 |
| Random Forest | Uncalibrated | 0.9799 | 0.8696 | 0.0085 |
| Random Forest | Sigmoid | 0.9799 | 0.8696 | 0.0076 |
| Random Forest | Isotonic | 0.9776 | 0.8557 | 0.0075 |

Calibration improved Random Forest's Brier score. Gradient Boosting already achieved a comparatively low Brier score, which remained unchanged to four decimal places after sigmoid calibration.

Isotonic calibration reduced average precision for both models, potentially because its stepwise probability mapping introduces tied prediction scores.

A low Brier score does not establish perfect calibration, particularly with rare outcomes. The calibration results should therefore be interpreted in the context of the synthetic dataset and limited number of positive observations.

Results are stored in `training/calibration_comparison.csv`.

##### Experiment 3 — Economic threshold optimization

The conventional 0.5 classification threshold does not reflect the asymmetric economic consequences of missed failures and unnecessary interventions.

Using the working economic assumptions defined in the business case, the simplified break-even probability is 0.09375. However, the threshold was selected empirically using the validation set, accounting for intervention costs, residual failure costs and human review costs.

All six candidates from the calibration experiment were evaluated at thresholds from 0.01 to 0.98 in increments of 0.01, including the theoretical break-even probability of 0.09375.

The optimization minimized expected cost per 1,000 processes subject to a minimum validation recall of 0.80.

Human review was included whenever a prediction was critical, within the uncertainty band or outside the training ranges. LLM costs and additional agent-triggered reviews were excluded at this stage because the complete agent had not yet been evaluated.

| Model | Threshold | Precision | Recall | Reviews | Cost per 1,000 |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting — Uncalibrated | 0.18 | 0.9322 | 0.8088 | 68 | €57,360.00 |
| Gradient Boosting — Sigmoid | 0.19 | 0.9322 | 0.8088 | 71 | €57,382.50 |
| Random Forest — Isotonic | 0.09 | 0.9167 | 0.8088 | 51 | €57,382.50 |
| Random Forest — Sigmoid | 0.11 | 0.9167 | 0.8088 | 56 | €57,420.00 |
| Random Forest — Uncalibrated | 0.27 | 0.9167 | 0.8088 | 57 | €57,427.50 |
| Gradient Boosting — Isotonic | 0.07 | 0.4091 | 0.9265 | 154 | €59,455.00 |

Although the uncalibrated Gradient Boosting model achieved the lowest estimated cost, the sigmoid-calibrated version was selected because it maintained the same classification decisions at its optimized threshold while explicitly incorporating probability calibration.

The additional cost was only €22.50 per 1,000 processes, arising from three additional human reviews across the 2,000 validation observations.

The selected configuration is therefore Gradient Boosting with physics-informed features, sigmoid calibration and a decision threshold of 0.19.

Results are stored in `training/threshold_comparison.csv` and `training/economic_comparison.csv`.

##### Selected model — Integrated validation

The trained model was serialized and integrated into `component_a.assess()`. The application-facing prediction function was then evaluated over all 2,000 validation observations to verify consistency between training and inference.

| Actual / Predicted | No failure | Failure |
|---|---:|---:|
| No failure | 1,928 | 4 |
| Failure | 13 | 55 |

| Metric | Validation result |
|---|---:|
| Accuracy | 99.15% |
| Precision | 93.22% |
| Recall | 80.88% |
| F1-score | 86.61% |
| Average precision | 88.30% |
| ROC-AUC | 96.54% |
| Brier score | 0.007142 |
| Recommended interventions | 59 |
| Human reviews | 71 |
| Human review rate | 3.55% |
| Estimated cost per 1,000 processes | €57,382.50 |

The integrated validation reproduced the classification metrics, review count and economic cost obtained during threshold optimization.

The model met the minimum recall target of 80% on the validation set. Whether it also meets the held-out test recall requirement and outperforms the main baseline remains to be established through final evaluation.

##### Experiment 4 — Failure-mode classification

Four additional independent Gradient Boosting classifiers were trained to estimate individual failure-mode probabilities. Each uses 50 estimators, a learning rate of 0.1, maximum tree depth of 3, engineered features and five-fold sigmoid calibration.

The classifiers were evaluated on the validation set using a threshold of 0.5.

| Failure mode | Training positives | Validation positives | Precision | Recall | Average precision |
|---|---:|---:|---:|---:|---:|
| TWF | 27 | 12 | 0.0000 | 0.0000 | 0.0630 |
| HDF | 67 | 22 | 1.0000 | 0.9091 | 0.9962 |
| PWF | 64 | 16 | 1.0000 | 1.0000 | 1.0000 |
| OSF | 57 | 21 | 1.0000 | 1.0000 | 1.0000 |

Heat dissipation, power and overstrain classifiers performed strongly on validation because their labels are generated according to physical relationships captured by the engineered features.

Perfect validation performance for power and overstrain does not establish equivalent performance in real industrial environments.

Tool wear failure was considerably harder to predict. Despite a ROC-AUC of 0.8833, its average precision was only 0.0630 and it detected none of the 12 validation positives at the 0.5 threshold. This reflects the limited positive examples and stochastic nature of tool failure in the dataset.

TWF probabilities are retained as advisory signals only and should not independently determine a maintenance action.

RNF was excluded from individual classification because it occurs randomly by design.

The failure-mode estimates are independent probabilities of their respective labels, not mutually exclusive diagnostic categories. They are not required to sum to one or to the overall machine failure probability.

Results are stored in `training/failure_mode_comparison.csv`.

##### Component A — Functional smoke tests

Five manually constructed machine-reading scenarios were evaluated through the integrated `assess()` function.

| Scenario | Failure probability | Risk band | Observed behaviour |
|---|---:|---|---|
| Normal operation | 0.17% | Low | Low estimated failure risk |
| Heat dissipation risk | 90.80% | Critical | HDF ranked first at 97.33% |
| Power failure risk | 97.45% | Critical | PWF ranked first at 98.48% |
| Overstrain risk | 99.58% | Critical | Both PWF and OSF identified |
| Out-of-distribution input | 0.14% | Low | Three out-of-range feature flags generated |

All five smoke tests completed successfully.

The out-of-distribution case demonstrates why probability estimates cannot be interpreted independently of data quality and operating conditions. Despite its low predicted failure probability, the observation must be routed to human review by the downstream guardrails.

These smoke tests verify the behaviour of Component A only. They do not establish that the complete agent or application satisfies all safety requirements.

##### Final held-out evaluation

[Pending] The selected model, calibration method and threshold must be frozen before evaluating the 2,000 held-out test observations. The final results will be reported alongside the strongest economic baseline, confidence intervals and the predefined go/no-go criteria.
*Each component on its own: the metric, the result, the test data and its size.*

### End to end

*How you wrote `eval/cases.csv` (counts by kind, two examples). Results against the baseline. Where the output is free text: the judge's rubric and your agreement with it on your own labels.*

### Cost and latency
**Component A — Resource requirements and provisional economic performance.**

The machine learning component runs locally within the application and does not require external API calls or consume LLM tokens.

The serialized production artifacts are approximately 40 KB for the overall failure classifier, 57 KB for the four failure-mode classifiers and 5.9 KB for the model card, with approximately 108 KB of total disk usage.

The models are cached in memory after their initial load. The small serialized artifact sizes support lightweight deployment, although actual application memory consumption and inference latency must be measured independently.

For model selection purposes, the validation cost was calculated using 55 true positives, four false positives, 13 false negatives and 71 human reviews.

| Cost contribution | Total for 2,000 validation observations |
|---|---:|
| True positives: 55 × €1,100 | €60,500 |
| False positives: 4 × €300 | €1,200 |
| False negatives: 13 × €4,000 | €52,000 |
| Human reviews: 71 × €15 | €1,065 |
| Total estimated cost | €114,765 |

This corresponds to an estimated cost of €57,382.50 per 1,000 processes, compared with €136,000 for the reactive no-intervention policy (B0). The provisional validation cost reduction is therefore 57.81%.

These figures are used for model and threshold selection only. They exclude LLM costs and additional agent-triggered reviews, rely on assumed industrial costs and prevention effectiveness, and do not represent observed savings.

The final economic comparison against the main baseline, plant-level extrapolations, sensitivity analysis and measured latency are reported separately following the coordinated test evaluation.

*Per request: tokens, euros and seconds, for each component and in total. What it would cost per month at the volume in your business case.*

## Operations

*What you log. The quality signal and the drift check you monitor, the reference set, and the alert threshold with its justification. The drift demo: what you shifted, and the result with the alert firing (screenshot or numbers). The rate of false alarms on normal inputs.*

**Deployment (25 Nov 2026).** The app runs on Render's free plan at https://maintai-238a.onrender.com, created from `render.yaml` as a Blueprint: one web service and one free PostgreSQL database for the logs, connected through `DATABASE_URL`. `GEMINI_API_KEY` is stored only in the service's environment variables. On Render, `LLM_CACHE=off`, so live requests never come from the LLM cache, and `DAILY_LLM_BUDGET=400` keeps a margin below the free tier's 500 requests a day.

**Records excluded from analysis.** The first deployment still used stub components. The 2 requests and 2 👍 logged on 25 Nov 2026 were smoke tests of the deployment. They are not evidence for the evaluation or for the bandit's learning. They can be identified by their versions, which start with `stub-` (for example `component_a = stub-a-v0`), and every analysis and the bandit exclude records with stub versions.

## Failures
### Component A — Failure analysis and limitations

The machine learning experiments identified several limitations that affect the interpretation and potential deployment of MaintAI.

| Issue | Observed evidence | Mitigation or next step |
|---|---|---|
| Missed machine failures | 13 of 68 validation failures were not detected at the selected threshold | Monitor false negatives, investigate missed cases and evaluate on real plant data |
| Unnecessary interventions | Four false positives on validation | Use cost-sensitive threshold selection and human review |
| Unreliable tool wear diagnosis | TWF recall and precision of zero at a 0.5 threshold | Treat TWF predictions as advisory signals and consider physical wear indicators |
| Out-of-distribution predictions | A smoke-test case produced a low failure probability despite three out-of-range features | Require human review independently of the predicted probability |
| Inconsistent ground-truth labels | 27 discrepancies between overall failure and individual failure-mode labels | Preserve original labels and document their limitations |
| Synthetic failure rules | Perfect validation results for PWF and OSF | Avoid extrapolating synthetic-data performance to real machinery |
| Potential sequential correlations | Stratified random splitting of sequentially generated observations | Conduct chronological or independent real-data robustness evaluation |
| Limited evidence of calibration | Small number of positive observations | Assess reliability using larger, independent datasets |

The model estimates failure risk under current operating conditions. It does not predict remaining useful life, the exact timing of a future breakdown or the causal effect of maintenance interventions.

The economic evaluation assumes an 80% prevention effectiveness. This is a business assumption rather than an effect learned from the dataset, which contains no observed maintenance intervention outcomes.

Before real-world deployment, the main priority would be validation using historical and prospective industrial data, including actual failures, maintenance actions and resulting outcomes.

End-to-end failure analysis, including agent and hand-off failures, is reported separately.
*Failed end-to-end cases grouped by cause (which component, or the hand-off between them), with counts, and what you would fix first.*

## Decision
### Component A — Decision policy

The selected machine learning decision threshold is 0.19. Preventive intervention is recommended by the ML threshold policy when the estimated probability of failure is greater than or equal to 0.19.

The component assigns risk categories using the following fixed boundaries:

| Risk band | Failure probability |
|---|---|
| Low | p < 0.19 |
| Medium | 0.19 ≤ p < 0.30 |
| High | 0.30 ≤ p < 0.80 |
| Critical | p ≥ 0.80 |

The uncertainty band is defined as:

`threshold / 2 ≤ p < 2 × threshold`

At the selected threshold, this corresponds to probabilities between 0.095 (inclusive) and 0.38 (exclusive).

Component A exposes the information necessary for the downstream guardrails to require human review whenever at least one of the following conditions is met:

- Critical failure risk: p ≥ 0.80.
- Prediction within the uncertainty band: 0.095 ≤ p < 0.38.
- One or more out-of-distribution feature flags.

Out-of-distribution flags are generated when numerical raw or engineered features fall outside their observed training minimum and maximum values. These checks identify range violations but cannot detect every possible distribution shift.

The ML-triggered review policy identified 71 of the 2,000 validation observations for human review, representing 3.55% of the validation sample. Final daily review estimates will be calculated using the held-out test set.

The agent may introduce additional review requirements. Human review is therefore not determined solely by the ML risk category, and a low predicted probability does not guarantee safe operating conditions.

The selected configuration is suitable for integration and further evaluation as a university prototype. The final go/no-go decision remains subject to the predefined economic, technical, operational and safety criteria, including comparison with the strongest baseline and final held-out evaluation.


*Which cases go to a person: the rule, and how many cases a day that would be. Go / no-go / go with conditions: the conditions, and the evidence behind them.*

## Bonus: reinforcement learning

**Status (26 Nov 2026): implemented and tested; no real learning evidence yet.**
- **Labels.** [Measured] = local tests and measurements. [Simulated] = simulated users, never used by the live policy. [Pending] = not done yet.
- **Why there is no real evidence yet.** Real ratings only start to count once the real risk model (WP1) and the real agent with one prompt per style (WP3) are deployed. We do not claim the bonus until there is evidence from real logs.

**What learns, and what it may not touch.**
- A multi-armed bandit chooses the *writing style* of each work order: `checklist` (short numbered steps), `cost` (opens with preventive vs breakdown cost) or `root_cause` (explains the physical indicators).
- The style is passed to the agent, and nothing else. The risk model and the deterministic safety rules never receive it (`guardrails.finalise` has no arm argument).

**The loop: feedback → reward → update → behaviour.**
1. **Feedback.** The user rates the order without knowing its style; the style is revealed only after the vote.
2. **Reward.** Useful = 1, not useful = 0. A request with no rating gives no reward.
3. **Update.** The posterior for each style is Beta(α, β), with α = 1 + positive ratings and β = 1 + negative ratings. It is rebuilt from the logs on every request.
4. **Behaviour.** Thompson sampling: draw one sample from each posterior and choose the highest. A style with little evidence sometimes draws a high sample and gets tried; a style with good evidence usually wins.

**Exploration and exploitation.**
- In Thompson sampling both come from the same rule: every choice is a posterior sample, and there is no separate exploration step.
- For monitoring, we label a choice *exploration* when the chosen style is not the one with the highest posterior mean at that moment. The label is undefined while several styles share the highest mean, as under the neutral prior.
- The cost of exploration is the regret below.

**Prior: neutral Beta(1, 1)** (decided by Manuela; it revises decision D6).
- We have no evidence that one style is better, so production starts with Beta(1, 1) for every style and no pseudo-observations. Every learned preference therefore comes from real ratings.
- The master plan's capped simulated prior (10 pseudo-observations per style) was studied in simulation only.

**Simulation results** [Simulated]. Source: `eval/simulate_bandit.py`, 100 runs × 300 ratings, fixed seeds. The simulated users like `root_cause` 75% of the time and the other styles 45%. Regret is the expected positive ratings lost against always choosing the best style.

| Prior | P(best) after 50 | P(best) after 300 | Ratings to 80% | Regret after 100 | Regret after 300 |
|---|---|---|---|---|---|
| Neutral Beta(1, 1) | 82% | 98% | 45 | 7.0 | 9.4 |
| Capped plan prior (10 pseudo-obs.) | 81% | 99% | 50 | 7.5 | 9.3 |
| Wrong prior (favours `checklist` at 90%) | 63% | 99% | 80 | 13.0 | 15.9 |
| Uniform random selection | 33% | 33% | never | 20.0 | 60.0 |

- **Regret.** Thompson's cumulative regret flattens once the best style dominates, while random selection's grows linearly. After 300 ratings Thompson loses about 9 positive ratings against 60 for random.
- **Wrong prior.** It roughly doubles the early regret but is corrected by the data.
- **No false winner.** With three equally good styles (50% each), no style dominates: final selection probabilities are 38% / 29% / 33% with the neutral prior.
- **Limits of the simulation.** Users are independent coin flips with fixed rates. Real ratings are fewer, noisier, correlated within a session, and may change over time.
- **Calibration of the 95% intervals** [Simulated] (`eval/bandit_calibration.py`, 200 runs per scenario). With rates drawn from the prior, coverage is 95–96%, which checks the implementation. Adaptive sampling bends it: the best style is under-covered while it is being found (about 90% at 25–100 ratings), and a style starved after early bad luck can be badly under-covered (31% of 29 intervals and 0% of 16 with equal styles: small samples, kept in the results). The dashboard intervals are a guide, not a guarantee.

**Risks.**
- **Reward hacking.** The policy optimises what raters like, not what is correct. A style could win by sounding confident or being short. Mitigations: the safety rules and costs never come from the arm; the arm does not change the risk or the review rules; order quality is judged separately (evaluation and judge).
- **Feedback bias.** Raters will mostly be the team and classmates, not technicians, and early ratings come from demo scenarios. Ratings are blind to the style, but the learned preference only shows that the loop works, not what technicians would prefer.
- **Delayed and missing rewards.** A rating can arrive later, change, or never arrive. Unrated requests do not count; a changed rating replaces the earlier one; the history replays ratings in the order given. The dashboard measures the response rate and the reward delay (median and p95).
- **Feedback loop.** The policy decides which style collects evidence, so a style with early bad luck is served less. Thompson sampling keeps serving it with its selection probability; we log those probabilities and show an exploratory propensity-weighted (IPS) estimate beside the observed reward. It is not causal and does not correct non-response.
- **Non-response.** Only rated orders teach the policy. The dashboard compares response rates by risk band, decision and style, and flags gaps of 20 points or more.
- **Abuse.** Feedback needs no login, so check.py keeps working. A random, temporary, pseudonymous session tag (no personal data) caps each session at 15 counted ratings and 3 clicks per order. It is not an identity: a new page is a new session.

**Persistence and versioning.**
- There is no state file: the posterior is rebuilt from the logs (PostgreSQL on Render) on every request, so it survives restarts and redeploys.
- Each request logs the chosen arm, the selection probabilities (Monte Carlo, summing to 1), the exploration label and the policy version (`ts-v1`).
- Only eligible ratings count: real traffic, handled, not degraded, no `stub` version, the current prompt version and a compatible policy version. A new prompt version starts a new policy.
- If the logs cannot be read, the request is served from the prior alone and the record says so (`bandit_policy_from = "prior_only"`).

**Tests** [Measured]. 41/41 pass (`uv run python tests/test_bandit.py`): updates, exclusions, duplicate and changed ratings, history without future information, restarts, log failures, session limits, reward delay, missing data, evidence export, simulation isolation, `/run` compatibility and a performance budget (list in `docs/WP6_BANDIT.md`).

**Latency and memory** [Measured locally]. Choosing an arm takes 3–6 ms with up to 2,000 logged requests and about 60 ms with 5,000. A request served end to end with the stubs takes 4–7 ms. Peak memory is about 156 MB after serving and 205–210 MB after the first Monitoring refresh, which imports scipy (about 2 s); scipy is never loaded inside `/run`. `/run` serves two requests at a time from a bounded queue. [Pending] Measurement on Render (512 MB, 0.15 CPU) with the real models.

**Pending and limitations.**
- [Pending] **Safety with the real agent (WP3).** Today the action, priority and the agent's own review request come from the LLM draft, so a real LLM could change them with the style. A test with an adversarial agent records this gap. The fix belongs to WP3: decide those fields without the arm, use the arm only to write the text, and have the guardrails reject any difference (`docs/WP6_BANDIT.md`).
- [Pending] **Real feedback campaign** and validation of persistence on PostgreSQL. Persistence is tested locally with log files, which use the same `logbook.load()`.
- [Pending] **Real-feedback evidence for this report**: `eval/export_rl_evidence.py` exports it (aggregates only), once real ratings exist.

## How to run it
### Reproducing Component A

The trained production models are stored in `models/`. No external model download, API key or retraining is required for Component A inference.

To reproduce the machine learning experiments from the original dataset, run the following commands from the repository root:

```bash
uv sync
uv run python training/prepare_data.py
uv run python -m training.train
uv run python -m training.calibrate
uv run python -m training.optimize_threshold
uv run python -m training.train_failure_modes
uv run python -m training.export_model
```

The calibration script generates the temporary candidate model collection required for threshold optimization and model export.

The final production artifacts are:

- `models/failure_model.joblib`
- `models/failure_modes.joblib`
- `models/model_card.json`

The model card records the dataset hash, scikit-learn version, model configuration, calibration method, decision threshold, training feature ranges, validation metrics and known limitations.

The public inference interface is `maintai.component_a.assess()`, which accepts a `MachineInput` and returns a `RiskAssessment`.

All model selection scripts use the training and validation datasets only. The held-out test dataset is reserved for the final evaluation.
*Anything beyond SETUP.md that the professor needs to run it: which keys, any download step, how long the first run takes.*

## AI use

*To be completed at the end of the project. Kept up to date as we work.*

- **Claude Code (Anthropic):** used to analyse the template repository, compare product ideas, draft the project plan, draft the text of the business case in this report, and help write code. All decisions, assumptions and criteria were reviewed and approved by team members, as recorded in CONTRIBUTIONS.md.
- **ChatGPT (OpenAI):** used by Manuela to review the project plan and to draft the prompts given to Claude Code.
- **Bandit (WP6):** Claude Code helped write the bandit, the simulator, the tests and the dashboard. Manuela chose the neutral prior, the reward, the exclusion rules and the safety scope, and reviewed every result.

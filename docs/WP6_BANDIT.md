# WP6 · Writing-style bandit (MaintAI)

Owner: Manuela. Status: algorithm, simulator, tests, dashboard and hardening (session limits, feedback-quality metrics, interval calibration, evidence export) done. Real learning starts once WP1 (risk model) and WP3 (real agent with one prompt per style) are deployed.

## What it does

Thompson sampling with a Beta-Bernoulli posterior over three work-order writing styles (`checklist`, `cost`, `root_cause`).

- **Reward**: a thumbs-up = 1, a thumbs-down = 0. A request without a rating does not count.
- **Selection**: one sample from each Beta, and the highest sample wins.
- **Persistence**: no state file. The policy is rebuilt from `logbook.load()`, which reads Postgres on Render.
- **Code**: `maintai/bandit.py`.

## Prior: neutral in production (decided by Manuela, 2026-11-26)

- **Production prior.** Beta(1, 1) for every style, with no pseudo-observations. Every learned preference therefore comes from real ratings.
- **The master plan's capped prior.** It was studied in simulation (`eval/simulate_bandit.py`, 100 runs × 300 ratings; the best style has a 75% like-rate and the others 45%).

| Prior | P(best) after 50 ratings | Ratings to reach 80% | Regret after 300 (random: 60) |
|---|---|---|---|
| Neutral Beta(1, 1) | 82% | 45 | 9.4 |
| Capped plan prior (10 pseudo-obs.) | 81% | 50 | 9.2 |
| Wrong prior (strongly favours another style) | 63% | 80 | 15.9 |

- **Why neutral.** The capped prior brings no measurable benefit, and a wrong belief costs about 70% more regret before real ratings correct it. A wrong prior is still corrected: 99% after 300 ratings.
- **Equal styles.** With three equally good styles, the neutral policy stays spread out and picks no false winner (final probabilities 38% / 29% / 33%).
- **Simulation only.** All simulated data is labelled `simulated` and is never fed to the live policy.

## Methodological safeguards (all tested in `tests/test_bandit.py`)

- **Selection probabilities**: estimated with 4,000 posterior draws, rounded so they sum exactly to 1.
- **Exploration**: an *operational* monitoring label, not a step of Thompson sampling. A choice counts as exploration when the chosen style is not the one with the highest posterior mean at that moment. The label is undefined (`None`) while several styles share the highest mean, for example under the neutral prior.
- **Eligible feedback**: real traffic served by the live bandit; handled and not degraded; no `stub` version; same prompt version; compatible policy version (`ts-v1`). Only the latest rating per request counts, so a changed rating replaces the earlier one. At most 15 rated requests per session count (the first ones by request time); see "Abuse protection".
- **History without future information**: `policy_history()` replays the rating events in the order they were given. Each point uses only what was known at that moment, and the last point equals the live policy.
- **Log read failure**: the request is still served using the prior only, and the record logs `bandit_policy_from = "prior_only"`. The dashboard counts these decisions.
- **Logged with every request**: `arm`, `arm_probs`, `bandit_explore`, `bandit_greedy_arm`, `bandit_n_feedback`, `bandit_policy_from`, `session_tag`, `versions.bandit`.

## Abuse protection (session tags)

Decided by Manuela on 2026-11-26. Code: `maintai/feedback_guard.py`, `app.py`, `bandit.apply_session_cap()`.

- **Session tag.** Gradio gives each page load a random `session_hash`. We log only `sha256(per-process random salt + session_hash)`, cut to 16 characters, as `session_tag` on the request record.
  - No name, e-mail, IP address, user agent, cookie or device fingerprint is read or stored. The code reads `request.session_hash` and nothing else from `gr.Request`.
  - The salt is never written down, so tags cannot be linked to a browser after a restart.
- **Limits (assumptions):**

| Limit | Value | Where it is enforced | Survives a restart |
|---|---|---|---|
| Rating clicks per order | 3 | UI, in memory (further clicks refused with a message) | No |
| Rating clicks per session written to the logs | 60 | UI, in memory | No |
| Rated requests per session that train the policy | 15, the first by request time | When the policy is rebuilt from the logs (`eligible_feedback`, `policy_history`) | Yes |

- **What it is not.** It is pseudonymous, not anonymous: within one server process, the requests and ratings of a page load share a tag and can be linked to each other (that is what the cap needs). It is not an identity and not authentication. Reloading the page, a private window or a script gives a new session, so the limits bound casual or accidental abuse (a user clicking repeatedly, one session flooding the policy). A determined person with many sessions is not stopped. Requests without a tag (`check.py`, tests, records from before this change) are not capped.
- **Unchanged.** No login, so check.py and the grader reach `/run` freely. `logbook.REQUIRED_FIELDS` is unchanged, and the tag is an extra field. The feedback record is unchanged (`logbook.log_feedback` is a fixed file); the tag lives on the request record.
- **Quota.** `/run` is still protected by the daily LLM budget (WP3, `DAILY_LLM_BUDGET=400`), the queue size (16) and the concurrency limit (2 at a time, `MAINTAI_RUN_CONCURRENCY` / `MAINTAI_QUEUE_MAX_SIZE`). A per-session cap on `/run` would be weak, because every API client opens a new session, so there is none.

## Internal dependency on `logbook._read`

`bandit.load_feedback_events()` reads `logbook._read("feedback")`, a private helper of the fixed template module. `logbook.load()` keeps only the latest rating per request and drops its time, and the rating times are needed for the history and the reward delay.

- **Read-only.** Nothing on the `/run` path calls it.
- **Guarded.**
  - A missing helper, an exception (for example the database is unreachable) or malformed rows give an empty or partial frame, with the reason in `frame.attrs["error"]`.
  - The dashboard then shows "N/A" or "not available".
  - Tests simulate all three cases.
- **If the template changes `_read`**, only the history chart and the reward-delay KPI degrade.

## Feedback-quality metrics (`maintai/rl_monitor.py`)

Measurements only, from the logs. A missing value is `None` and is shown as "N/A", never 0.

- **Response rate**: rated / feedback-eligible requests, overall and per style.
- **Reward delay**: seconds from the logged order to its first rating (median, p95). Also counted:
  - changed ratings;
  - ratings faster than 3 s (assumption);
  - events dated before their request (clock problems, which are dropped).
- **Coverage by style**: served, eligible, rated, counted after the session cap, and share of counted ratings.
- **Served probabilities**: the `arm_probs` logged when each request was served, in time order, as block means beyond 60 points. These are the probabilities the policy really used, not a replay.
- **Non-response**: the response rate by risk band, decision and style. A flag is raised when two groups with at least 10 eligible requests differ by 20 points or more (assumption). The **risk mix per style** checks that no style got easier cases by chance.
- **Belief comparison**: per style,
  - the observed reward;
  - the posterior mean with its 95% interval;
  - a self-normalised IPS estimate (each rating weighted by 1 / the logged probability of the served style). This is **exploratory**: it re-weights for the policy serving some styles more often over time. It is not a causal proof, it is noisy with few ratings, and it does **not** correct non-response, because it assumes ratings are missing at random given the style;
  - P(selected), which is the probability of being the best style, not a reward.
  
  A flag is raised when the posterior and the observed reward differ by 10 points or more (assumption).

All of them appear in Adaptive Intelligence with a provenance badge: **Real logs**, **Simulated**, **Assumption** or **Pending**.

## Calibration of the 95% credible intervals (simulated)

`eval/bandit_calibration.py`: 4 scenarios × 200 runs × 300 ratings, seed 20261127, the production code. Outputs: `eval/bandit_calibration.csv` and `eval/bandit_calibration_summary.json`. Coverage for all styles:

| Scenario (declared rates) | 10 | 25 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|
| Rates drawn from the prior | 95.0% | 95.7% | 96.2% | 96.0% | 96.5% | 94.8% |
| One better (0.45 / 0.45 / 0.75) | 98.2% | 95.5% | 96.3% | 96.5% | 97.3% | 97.8% |
| All equal (0.50) | 99.0% | 97.3% | 97.7% | 95.8% | 92.2% | 92.0% |
| Low rates (0.10 / 0.15 / 0.25) | 97.3% | 97.3% | 96.2% | 97.5% | 97.0% | 97.2% |

Monte Carlo error is about ±1 point (600 intervals per cell).

- **Implementation check.** With rates drawn from the prior, coverage is close to 95%, as Bayesian calibration requires.
- **Adaptive sampling.** The best style is under-covered while it is being found: 89.5–90.5% between 25 and 100 ratings in "one better". This is the worst cell with at least 100 intervals. Rarely served styles are over-covered, because their intervals are wide.
- **Starved styles.** With equal styles, a style starved after early bad luck has a pessimistic interval: 31% coverage of 29 intervals at 200 ratings, and 0% of 16 at 300. These samples are small but the direction is clear, and it is the feedback loop the dashboard warns about.
- **Conclusion.** The intervals are a reasonable guide, not a guarantee. Do not report a style as "worse" from its interval alone when it has few ratings.

## Evidence export

`uv run python eval/export_rl_evidence.py [--logs-dir logs] [--out eval/rl_evidence]`. After `main.py export-logs`, this exports the Render logs.

- **Aggregates only.** No request ids, inputs, session tags or single rating times; a test checks this.
- **Labelled.** Every file is labelled `measured` or `simulated`, and `manifest.json` lists the assumptions.
- **No eligible ratings.** The manifest then says that the files show the prior and are not evidence of learning.
- **Do not commit** the output from stub or local data. Commit it once the real campaign has run, before the free database expires.

## Safety: what is guaranteed and what WP3 must guarantee

Guaranteed today, by construction: the arm never reaches component A or `guardrails.finalise()`. So the risk assessment and the deterministic review rules (critical risk, out of training range, uncertain prediction, failed grounding) cannot change with the style. A test with an adversarial fake agent, which writes a different order for each style, confirms it.

**Not yet guaranteed (requirement for WP3).** The *action*, *priority* and the agent's own review request come from the LLM draft. A real LLM could change them depending on the writing style. Proposed fix, in WP3's files:

1. The agent decides the action, priority, failure mode and review request **without the arm**: in the tool/decision call, or from deterministic rules.
2. The arm is used only in the final writing call (steps and justification).
3. Guardrails reject a draft whose decision fields differ from the arm-free decision.

Once the real agent exists, the invariance can be checked offline with the LLM cache: run the same readings with each arm and compare those fields. That takes 3 calls per reading, so it should run once with the cache and never on the test split.

## Instructions for WP3 (agent, P3)

The bandit only learns once these are in place. Until then no rating is eligible, which is correct.

1. **Prompts.** One prompt per style in `maintai/prompts/`, for `checklist`, `cost` and `root_cause`. Each prompt changes only *how* the order is written: order of sections, length, emphasis.
2. **Arm-free decisions.** `agent.plan(machine, risk, arm)` must decide `priority`, `action`, `failure_mode_addressed`, `requires_human_review` and `review_reason` **without using `arm`**: in the tool/decision call, or from deterministic rules. Use `arm` only in the final writing step (`steps`, `justification`).
3. **Enforce it in the guardrails.** `guardrails.finalise()` should reject a draft whose decision fields differ from the arm-free decision. Keep `finalise()` without an `arm` parameter; a test checks this.
4. **Prompt version.** Set `agent.PROMPT_VERSION` to a version that does not start with `stub`, for example `agent-v1`. Bump it whenever a prompt changes: that restarts the policy, as intended (D6).
5. **Degraded mode.** When the agent falls back to the template, set `degraded=True`. Those ratings are excluded automatically.
6. **Offline check, once, with the LLM cache.** Run a few validation readings with each arm and confirm that the decision fields are identical. Never use the test split.

## Instructions for WP5 (monitoring, P6)

1. **The RL panel is done** (`ui/rl_dashboard.py`, inside Monitoring). Please do not build a second one. If you deliver `monitoring_ui.render()`, call `ui.rl_dashboard.render(logbook.load())` for the RL section, or keep it where it is.
2. **Data contracts you can use** (also `maintai/rl_monitor.py`: `response_rate`, `reward_delay`, `coverage_by_arm`, `served_probabilities`, `non_response`, `belief_comparison`, `evidence`):
   - `bandit.policy_snapshot(logs)`: arm, alpha, beta, mean, ci_low, ci_high, n_real, n_prior, successes, failures, p_select;
   - `bandit.policy_history(logs, bandit.load_feedback_events())`;
   - `bandit.eligible_feedback(logs)`;
   - the logged fields `arm`, `arm_probs`, `bandit_explore`, `bandit_policy_from`.
3. **Simulations.** `eval/bandit_simulation*.{csv,json}` are simulated. Show them only in a section labelled as simulation, never in charts of real activity.
4. **Suggested alert.** If `bandit_policy_from == "prior_only"` appears in recent requests, the logs were unreadable while serving.

## Latency and memory (measured locally)

Hardening round, 5,000 synthetic requests in memory:

| Step | Time |
|---|---|
| `decide()` | about 60 ms |
| All `rl_monitor` metrics | about 0.9 s |
| Whole Adaptive Intelligence section | about 0.35 s |

- **Whole app.** The first Monitoring refresh takes 2.1–2.4 s, almost all of it importing `scipy.stats`; the new code adds about 0.05 s. Peak memory is 205–210 MB.
- **Before Python bytecode is cached**, the first refresh can take about 4 s. Render's build does not precompile bytecode, so expect this once after each deploy.
- **Six parallel `/run` calls** through the queue all succeed.

Earlier measurements (round 2, best of 5 runs):

| Logged requests | `logbook.load()` | `decide()` | Peak RSS of the process |
|---|---|---|---|
| 500 | 5 ms | 3 ms | 165 MB |
| 2,000 | 21 ms | 6 ms | 173 MB |
| 10,000 | 123 ms | 22 ms | 242 MB |

The expected project volume is hundreds of requests. Render has 0.15 CPU, so expect several times the local times. These figures are still estimates: Render has not been measured yet. Above about 5,000 requests, consider reading only the bandit columns.

Whole app, measured locally with the stub components:

| Moment | Time | Peak memory |
|---|---|---|
| Import `app` | 1.3 s | 158 MB |
| 20 requests served end to end | median 4 ms each | 161 MB |
| First Monitoring refresh (imports scipy.stats for the Beta curves) | 1.4 s | 227 MB |
| Later Monitoring refreshes | 0.02 s | – |

scipy is never loaded inside `/run`. It is declared as a direct dependency in `pyproject.toml` (it was already installed through scikit-learn, so `uv.lock` only records the declaration). A clean `uv sync --frozen --no-dev` install passes the 23 tests.

**Pending:** measure on Render (512 MB, 0.15 CPU) once the real model (WP1) and the Gemini client (WP3) are loaded. Both add memory, and scikit-learn imports scipy anyway.

## Files

- `maintai/bandit.py`: policy, filters, snapshot and history.
- `eval/simulate_bandit.py` with its outputs `eval/bandit_simulation.csv` and `eval/bandit_simulation_summary.json`: simulated only.
- `ui/rl_dashboard.py`: the Adaptive Intelligence section in Monitoring.
- `maintai/rl_monitor.py`: feedback-quality metrics for the dashboard and the evidence export.
- `maintai/feedback_guard.py`: session tags and feedback limits.
- `eval/bandit_calibration.py` with its outputs `eval/bandit_calibration.csv` and `eval/bandit_calibration_summary.json`: simulated only.
- `eval/export_rl_evidence.py`: aggregated evidence export.
- `eval/select_rl_scenarios.py` with its output `eval/rl_scenarios.csv`: 30 validation readings for the real feedback campaign (never the test split), with risk bands from the real component A (`status = final` since 27 Nov 2026); see `docs/RL_FEEDBACK_PROTOCOL.md`.
- `tests/test_bandit.py`: 41 tests, run with `uv run python tests/test_bandit.py`. They cover:
  - posterior updates, the neutral prior and selection probabilities;
  - exclusions, and compatibility with a future prompt version;
  - duplicate and changed ratings;
  - session limits, and their expiry and memory bound;
  - history without future information, including the session cap;
  - persistence and log or reader failures;
  - missing data shown as N/A;
  - reward delay, response rate, IPS, served probabilities and non-response flags;
  - evidence export, without ids or tags;
  - simulation and calibration isolation and reproducibility;
  - an adversarial agent;
  - `/run` and `serve()` compatibility;
  - a performance budget.
- `app.py`: `serve()` uses `bandit.decide()` and logs the decision fields and the session tag. Its signature `serve(raw_input, session_tag=None)` is compatible with `serve(raw_input)`. `/run` still takes one text argument and is the only public endpoint.


## Appendix: full report text of the bonus section (moved on 27 Nov 2026)

`REPORT.md` was shortened on 27 Nov 2026 to respect the 3,000-word limit. This is its earlier bonus section, word for word (copied at commit `87f38ba`), so no detail, limitation or result is lost.


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

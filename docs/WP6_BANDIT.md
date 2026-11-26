# WP6 · Writing-style bandit (MaintAI)

Owner: Manuela. Status: algorithm, simulator, tests and dashboard done. Real learning starts once WP1 (risk model) and WP3 (real agent with one prompt per style) are deployed.

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
- **Eligible feedback**: real traffic served by the live bandit; handled and not degraded; no `stub` version; same prompt version; compatible policy version (`ts-v1`). Only the latest rating per request counts, so a changed rating replaces the earlier one.
- **History without future information**: `policy_history()` replays the rating events in the order they were given. Each point uses only what was known at that moment, and the last point equals the live policy.
- **Log read failure**: the request is still served using the prior only, and the record logs `bandit_policy_from = "prior_only"`. The dashboard counts these decisions.
- **Logged with every request**: `arm`, `arm_probs`, `bandit_explore`, `bandit_greedy_arm`, `bandit_n_feedback`, `bandit_policy_from`, `versions.bandit`.

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
2. **Data contracts you can use:**
   - `bandit.policy_snapshot(logs)`: arm, alpha, beta, mean, ci_low, ci_high, n_real, n_prior, successes, failures, p_select;
   - `bandit.policy_history(logs, bandit.load_feedback_events())`;
   - `bandit.eligible_feedback(logs)`;
   - the logged fields `arm`, `arm_probs`, `bandit_explore`, `bandit_policy_from`.
3. **Simulations.** `eval/bandit_simulation*.{csv,json}` are simulated. Show them only in a section labelled as simulation, never in charts of real activity.
4. **Suggested alert.** If `bandit_policy_from == "prior_only"` appears in recent requests, the logs were unreadable while serving.

## Latency and memory (measured locally, best of 5 runs)

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
- `tests/test_bandit.py`: 23 tests, run with `uv run python tests/test_bandit.py`.
- `app.py`: `serve()` uses `bandit.decide()` and logs the decision fields. The signature and `/run` are unchanged.

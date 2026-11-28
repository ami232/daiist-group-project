# WP5 · Operations monitoring: what is measured and why

Code: `monitor.py` (functions `real_traffic`, `operations`, `quality_counts`, `alerts`). Tests: `tests/test_monitor.py` (20, synthetic logs, no Gemini). Drift is in `docs/WP5_DRIFT.md`.

## Which requests count
`real_traffic()` keeps requests with `source = real` (or no source) and drops any request whose model or prompt version starts with `stub`. Smoke tests, `check.py` runs and simulations never reach the drift, quality or operations numbers. Requests answered from the LLM cache or refused as out of scope are not "Gemini requests": latency, cost and degradation are measured on requests that really called Gemini. A 👍 is a rating of usefulness, not a measure of accuracy.

## Operations table (`monitor.operations`)
Real requests; Gemini requests; tokens in / out / thinking; mean cost per Gemini request; latency p50 / p95 for all served requests and for Gemini requests; degraded share over the last 50 Gemini requests; number of 429; retries; Gemini requests today against the daily budget (Pacific-time day, like the Gemini quota). `quality_counts()` gives the denominators behind the review and 👎 rates per period.

## Alerts (each one is a message in the Monitoring tab)
| Alert | Rule | Why this threshold |
|---|---|---|
| Drift | PSI > 0.25 and > 5 x the PSI expected by chance, per feature, last 100 requests (at least 20) | Calibrated on validation: `docs/WP5_DRIFT.md` |
| Quota | Gemini requests today > 80% of the daily budget | Leaves 20% (80 requests of 400) to react before the app falls back to degraded mode. Counts every log, tests included, because they spend the same quota |
| Degradation | More than 20% of the last 50 Gemini requests in degraded mode (at least 10 requests) | A spent quota or an API outage degrades every request, so it has to fire after 10, not 20 |
| Review | More than 50% of the last 100 requests sent to review (at least 20) | Measured on 150 reference inputs: about 4% are reviewed (validation: 71 of 2,000, 3.6%); the 30 feedback-campaign scenarios: 30%. 50% is far above both, so it fires if the model starts flagging half of the traffic, not during the campaign |
| 👎 | More than 40% of the last rated requests (at least 20) | Not calibrated: there is no real rating yet. Kept as in the template and to be reported as such |

None of these thresholds has been tested on real production traffic yet: there is none. The tests use synthetic logs only, so they show that the rules fire and stay quiet as written, not that the numbers are right for real users.

## Draft for REPORT.md, section Operations (to complete with real numbers after the campaign)
Every request is logged with its features, both component outputs, model and prompt versions, latency, tokens, cost and arm; ratings are linked by request id. Monitoring only counts real traffic: stub versions and test sources are excluded, and cache hits are not counted as Gemini requests. We track temporal quality (share sent to human review and share of 👎 among rated requests, always with their denominators), latency p50/p95, tokens, cost, degraded-mode share, 429 errors and the daily Gemini quota.

Drift is measured with the population stability index on nine features (the five inputs, temperature difference, power, product type and `p_fail`) against 300 normal training inputs. The template rule raised false alarms in 31-47% of normal windows of 16-20 requests, so we calibrated it on validation windows only: additive smoothing and an alert only above five times the PSI expected by chance. This gives 0-1% false alarms on random normal windows of 16-100 requests and detects hotter, more worn shifted windows (100%; 99.5% for a half-size shift at 16 requests). The demo is quiet on the 16 normal cases and alerts on the 15 shifted inputs (air temperature, process temperature and `p_fail`).

Other alerts: quota above 80% of the daily budget, degraded mode in more than 20% of the last 50 Gemini requests, and more than 50% of requests sent to review (about 4% on validation-like inputs). Limits: the shifted set is synthetic, the review and degradation thresholds are reasoned from validation rates and the quota, not tuned on live traffic, and the 👎 alert is not calibrated. [Add here: number of real requests, latency p50/p95, cost per request and any alert that fired during the campaign.]

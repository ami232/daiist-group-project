# Integration checklist (27–30 Nov 2026)

How the remaining work packages go into `main`, what to run after each merge, and the submission checklist. Owner: Manuela (coordination). Status of each package: `docs/TEAM_PLAN.md`, section 0.

**Progress (27 Nov, 21:30):** steps 1 (WP1, PR #9) and 2 (WP2, PR #8) are in `main` and deployed. WP4 is finished on its branch `feat/wp4-evaluation` (no PR yet); it merges with `main` without code conflicts and its tests pass on the combined tree. WP3 and WP5 have not started in GitHub. The live status is in `docs/TEAM_PLAN.md`, section 0.

**Rule for every merge:** one PR per package, reviewed and approved by someone else, merged only if every check in section 2 passes. Every merge into `main` redeploys Render, so a broken `main` is a broken public app.

## 1. Recommended order

| Step | Package | Why in this position | Blocks |
|---|---|---|---|
| 1 | **WP1 · real component A** (`component_a.assess()`, chosen model in `models/`, `model_card.json`, threshold from the full cost model on validation) | Every other result depends on the real risk and its `p_fail` | WP3 tests, WP5 reference, WP4 evaluation, RL scenarios |
| 2 | **WP2 · tools data and economics** (`maintai/data/*.json`, `tools.py`, `economics.py`, sensitivity script) | The agent's grounding is checked against these data | WP3 grounding, WP4 cost results |
| 3 | **WP3 · LLM client, agent, prompts, guardrails** (one prompt per style, fixed prompt version, arm only changes the text) | Needs real risk and real tools; closes the safety gap of `docs/WP6_BANDIT.md` | WP4 end-to-end, RL campaign |
| 4 | **WP5 · drift reference and shifted set** (`main.py reference`, `monitoring/reference_features.csv`, `eval/reference.csv`, `eval/shifted.csv`, threshold calibrated on validation) | `p_fail` is a drift feature, so the reference must be built **after** the real model | check.py step 4 |
| 5 | **WP4 · evaluation** (`eval/cases.csv` ≥ 30 with all kinds, baselines B1/B2, `evaluate.py`, judge, `human_labels.csv` ≥ 30) | Needs the final pipeline; uses the test split once | check.py step 2, REPORT Evaluation, Decision |
| 6 | **WP6 · real feedback campaign** (`docs/RL_FEEDBACK_PROTOCOL.md`) | Needs real model and real agent deployed, frozen prompts | REPORT bonus |
| 7 | **WP8 · REPORT, CONTRIBUTIONS, tag `final`** | Last numbers from steps 1–6 | Submission |

WP4's hand-written `hard`, `review` and `out_of_scope` cases and WP5's design can be merged earlier if they do not depend on the model. If WP2 is late, WP3 can merge against the stub tools, but the grounding results must be re-run after WP2.

## 2. Checks after every merge

Run on the merged `main`, locally, before announcing it:

```bash
git switch main && git pull
```
```bash
uv sync --frozen && uv lock --check
```
```bash
uv run python tests/test_bandit.py
```
```bash
uv run python check.py
```

| Check | Expected | If not |
|---|---|---|
| `uv lock --check` | Lock file matches `pyproject.toml` | The PR added a dependency without `uv add`: fix in the PR |
| `tests/test_bandit.py` | 41/41 (more if tests were added) | The merge broke the pipeline contract or the bandit: revert or fix before anything else |
| `check.py` | Steps that the merged packages make possible pass (step 2 needs WP4, step 4 needs WP5) | Fix before the next merge |
| App starts: `uv run python app.py` | App and Monitoring tabs load; one sample reading gives an order | — |
| Render redeploy | "Live" in the Render dashboard; the app opens; banner lists only the stubs still pending | Read the Render logs; revert the merge if the app does not start |
| Render memory (section 6.1) | Below about 400 MB at rest and after a Monitoring refresh | See section 3, "Memory" |

## 3. Known incompatibilities to watch

| Area | Risk | Check |
|---|---|---|
| **Contracts** (`maintai/schemas.py`) | A package changes a field and breaks another module or `logbook.REQUIRED_FIELDS` | Contract changes only in their own PR, announced; run the tests and `check.py` step 3 |
| **Training/serving skew** (WP1) | `component_a` builds features differently from `training/feature_engineering.py`, or maps the dataset columns (`Air temperature [K]`, …) to `MachineInput` fields wrongly | Re-score the validation split through `component_a.assess()` and compare with the training script's validation metrics (same numbers expected) |
| **Model file** (WP1) | `training/candidate_models.joblib` holds several candidates (about 0.7 MB). Production should load **one** chosen model from `models/`, pickled with the same scikit-learn as `uv.lock` (1.9.1) | Load it in a clean clone; check its size and load time; `MODEL_VERSION` must not start with `stub` |
| **Calibration and threshold** (WP1) | Threshold or calibration chosen on the test split | Only train and validation; the test split is used once, for the report |
| **Prompt versions** (WP3) | Any prompt change creates a new prompt version, which starts a new bandit policy (earlier ratings stop counting) | Freeze prompts before the RL campaign |
| **Arm and safety** (WP3/WP6) | The real LLM changes action, priority or the review request with the style | The adversarial-agent test in `tests/test_bandit.py` must pass with the real agent |
| **LLM cache** (WP3) | Live traffic served from the cache | `LLM_CACHE=off` on Render (already set in `render.yaml`); the cache is allowed only in local tests and evaluation re-runs |
| **Quota** (WP3/WP4/WP6) | Evaluation, demos and the RL campaign share 500 calls a day | `DAILY_LLM_BUDGET=400` on Render; evaluation uses each member's own key and the cache; RL budget in the protocol, section 8 |
| **Drift reference** (WP5) | Reference built before the real model, so `p_fail` drifts at once | Re-run `main.py reference` after WP1 merges |
| **API** (WP7) | A new handler becomes a public endpoint | `/?view=api` shows only `/run` with `raw_input` |
| **Logs** | New fields missing from Postgres, or personal data in a record | Section 6.3; records keep only readings, outputs and versions |
| **Dependencies** | A heavy library (e.g. a second ML framework) pushes Render past 512 MB | Announce before `uv add`; measure memory after the merge |

## 4. Acceptance criteria per package

- **WP1:** validation metrics and the chosen threshold documented (`model_card.json`); `assess()` returns calibrated `p_fail`, band and likely modes; no test-split use before the final evaluation; app latency for component A under 1 s on Render.
- **WP2:** every part, window, cost and policy the agent can cite exists in `maintai/data/`; the sensitivity script reproduces the grid of the business case.
- **WP3:** at most 3 Gemini calls per request; 429 and budget handled with the degraded mode; tokens and cost logged; zero ungrounded orders after the guardrails; review rules applied in 100% of the cases that need them.
- **WP4:** `eval/cases.csv` ≥ 30 with ≥ 1 normal, ≥ 5 hard, ≥ 5 review, ≥ 3 out_of_scope (check.py minimums; target 90); baselines tuned on train/validation; judge agreement (kappa) reported against `human_labels.csv`.
- **WP5:** drift alert fires on `eval/shifted.csv` and stays quiet on normal inputs; false-alarm rate reported.
- **WP6:** campaign run as pre-registered, or reported as not run, with the evidence export.
- **All:** `CONTRIBUTIONS.md` line in the same PR; no keys, personal data or logs committed.

## 5. Contracts, dependencies, memory, API, cache and logs (final pass)

| Item | Command or place | Pass |
|---|---|---|
| Contracts | `git diff <first-commit>..main -- maintai/schemas.py` reviewed by all owners | ☐ |
| Dependencies | `uv lock --check`; `pyproject.toml` lists only what the code imports | ☐ |
| Memory | Render → service → Metrics (section 6.1) | ☐ |
| API | `https://maintai-238a.onrender.com/?view=api`: only `/run`, parameter `raw_input` | ☐ |
| Cache | Render → Environment: `LLM_CACHE=off`; logs show no cached live calls | ☐ |
| Logs | Section 6.3: required fields, `session_tag`, `arm`, `arm_probs`, feedback rows | ☐ |

## 6. Render and PostgreSQL (manual steps)

### 6.1 Memory and CPU in Render Metrics

1. Open https://dashboard.render.com and sign in.
2. Click the web service **maintai** (not the database).
3. In the left menu, click **Metrics**.
4. Choose the period that covers your test (for example, the last hour).
5. **Memory:** read the line at rest, then open the app, run one reading in the App tab and press **Refresh** in Monitoring (this imports scipy). Note the highest value. The limit is 512 MB; keep a margin (below about 400 MB). Locally the app peaks at about 205–210 MB with the stubs.
6. **CPU:** the free plan has 0.15 CPU. Expect short peaks during a request or a Monitoring refresh; a line stuck at the top means requests are queuing.
7. Write down the date, the version (commit) and both values, and repeat after WP1 and after WP3 are merged. Put the final numbers in REPORT.md, "Cost and latency".

If the charts are empty, the service may have been asleep: open the app, wait a minute and reload the Metrics page.

### 6.2 LOGS_DATABASE_URL, safely

1. Render dashboard → click the database **maintai-logs** → **Connect** → **External** tab.
2. Copy the **External Database URL**. It contains the password: treat it like a key.
3. On your laptop, open `.env` (git-ignored; never `.env.example`) and set `LOGS_DATABASE_URL=<paste>`.
4. Check that it will not be committed: `git status` must not list `.env`.
5. Export: `uv run python main.py export-logs`. It writes `logs/requests-export.jsonl` and `logs/feedback-export.jsonl` (git-ignored).
6. Do not paste the URL in chats, issues, PRs, screenshots or the report. When the campaign is over, delete the line from `.env`. If the URL is ever exposed, rotate the password in Render (database → **Info** → reset credentials).
7. Free Render databases are deleted 30 days after creation (about 25 Dec 2026, plus 14 days of grace): export the final logs before the presentation.

### 6.3 New fields persist in PostgreSQL

After `export-logs`, run:

```bash
uv run python -c "import pandas as pd; r=pd.read_json('logs/requests-export.jsonl', lines=True); f=pd.read_json('logs/feedback-export.jsonl', lines=True); cols=['session_tag','arm','arm_probs','bandit_policy_from','source']; print(len(r), 'requests;', len(f), 'feedback rows'); print({c: int(r[c].notna().sum()) if c in r else 'missing' for c in cols}); print(r[[c for c in ['timestamp','versions','session_tag','arm'] if c in r]].tail(3).to_string())"
```

Expected after the hardening deploy (27 Nov 2026):

- requests made **before** the deploy have no `session_tag` (the field did not exist yet);
- a request made from the App tab **after** the deploy has a 16-character `session_tag`, plus `arm`, `arm_probs` and `bandit_policy_from`;
- a request made through `/run` from a script (check.py, the professor) has `session_tag` empty: that is correct, it has no browser session;
- no name, e-mail, IP address or user agent anywhere in the records.

To create the record to check, make **one** request from the App tab of the online app. Do not rate it unless you would genuinely rate it: today it is a `stub` request, so it is excluded from the bandit and the evaluation either way, and no artificial feedback is needed for this check.

### 6.4 Results of the check on 27 Nov 2026

Logs exported by Manuela at 20:55 with `main.py export-logs` (the database URL stays in `.env`; `logs/` is git-ignored and nothing from it is committed):

| Check | Result |
|---|---|
| Rows in PostgreSQL | 3 requests and 2 ratings, all from 25 Nov 2026 (the deployment smoke tests) |
| Fields of the request rows | `id`, `timestamp`, `versions`, `arm`, `arm_probs`, `source`, `status`, `degraded`, `outputs`, `features`, `latency_s`, cost and token fields: all present |
| `versions` | `component_a = stub-a-v0`, `component_b = none\|stub-agent-v0`, `bandit = stub-bandit-v0` |
| `arm_probs` | Present, but from the first (deterministic) bandit: `checklist` 1.0, the others 0.0 |
| `session_tag`, `bandit_policy_from` | Absent in all 3 rows. **Expected:** these fields were added by the hardening (27 Nov), after these rows were written |
| Feedback rows | `id`, `feedback`, `timestamp`. By design they carry no session tag: the tag is joined from the request |
| Personal data | None: no name, e-mail, IP address or user agent |
| Stub records | 3 of 3 (all) |
| Eligible for RL (`bandit.eligible_requests` / `eligible_feedback` on the export) | **0 requests, 0 ratings**, as expected: stub versions, old bandit version, degraded |

**Still to verify:** that a request made **after** the hardening deploy stores `session_tag`, `bandit_policy_from` and the `ts-v1` `arm_probs` in PostgreSQL. Make one request from the App tab of the online app (no rating needed), export again and run the command of section 6.3. Locally the same record is complete (`check.py` step 3 simulation on 27 Nov: all required fields, `bandit = ts-v1`, `bandit_policy_from = logs`).

## 7. Full check.py in a clean clone

Run this on the final candidate commit, not in your working folder (which has `.env`, `logs/` and local files the professor will not have):

```bash
cd "$(mktemp -d)" && git clone https://github.com/manueladominguez/daiist-group-project.git && cd daiist-group-project
```
```bash
uv sync --frozen && cp .env.example .env
```

Put your own `GEMINI_API_KEY` in the new `.env` (it stays in the temporary folder), then:

```bash
uv run python check.py --url https://maintai-238a.onrender.com
```

- Expected: all steps pass (5 + 1 with `--url`). Warnings about `human_labels.csv` or empty member rows must be fixed, not ignored.
- `check.py` without `--url` logs only to a temporary folder. With `--url`, it sends **one** real request to Render, which is logged in PostgreSQL as real traffic (unrated, so it does not train the bandit) and may use one Gemini call.
- Delete the temporary folder afterwards.

## 8. Submission checklist (Mon 30 Nov, 23:59)

| # | Item | Who | Done |
|---|---|---|---|
| 1 | All package PRs merged; `main` green in sections 2 and 7 | All | ☐ |
| 2 | REPORT.md complete, about 3,000 words without tables; pre-registered criteria unchanged; App URL; final commit filled in after tagging | WP8 | ☐ |
| 3 | CONTRIBUTIONS.md: every member's row (first name, GitHub username, areas), log complete | All | ☐ |
| 4 | Render memory/CPU measured with the final version; app live | Manuela | ☐ |
| 5 | RL evidence exported (or "campaign not run" stated) | WP6 | ☐ |
| 6 | Logs exported before the database expires; nothing from `logs/` committed | Manuela | ☐ |
| 7 | Secret scan: no keys, database URLs or personal data in the repo (`git grep -nE "AIza|postgres(ql)?://" -- ':!docs/INTEGRATION_CHECKLIST.md'` returns nothing) | Manuela | ☐ |
| 8 | Tag: `git tag final && git push --tags` on the agreed commit (only fixes that keep the app running after the deadline) | Manuela | ☐ |
| 9 | PR from `manueladominguez/daiist-group-project:main` to https://github.com/ami232/daiist-group-project | Manuela | ☐ |
| 10 | Blackboard: PR link, app link and a zip of the repo (`git archive --format=zip -o maintai.zip final`) | Manuela | ☐ |
| 11 | Presentation and backup video ready (1–3 Dec) | WP8 | ☐ |

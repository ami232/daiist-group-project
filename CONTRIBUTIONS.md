# Contributions

*Fill this in as you go, not on the last day. It is checked against the commits and against the peer assessment. A first name and a GitHub username are enough: this repo is public.*

## Members

| First name | GitHub username | Main areas |
|---|---|---|
| Manuela | manueladominguez | Repository setup, business case, interface and branding (WP7) |
| | | |
| | | |
| | | |
| | | |

## Log

*One line per piece of work: the date, who did it (more than one name is fine), and what. Code done in pair programming counts for both, even if only one of you committed it.*

| Date | Who | What |
|---|---|---|
| 2026-11-25 | Manuela | Forked the template, added the team as collaborators, set up the local environment with uv and checked that the template app, feedback and monitoring run |
| 2026-11-25 | Manuela | Compared product ideas, chose MaintAI (predictive maintenance + work-order agent) and approved the master plan, adding conditions on live mode, test-set use and the economic model. Comparison and plan drafted with AI assistance (Claude Code); ChatGPT used to review the plan and draft prompts. Pending confirmation by the rest of the team |
| 2026-11-25 | Manuela | Wrote the business case in REPORT.md: value proposition, AS-IS/TO-BE, baselines, KPIs, economic model, sensitivity plan, go/no-go criteria and limitations. Chose and approved the economic assumptions and go/no-go targets. Text drafted with AI assistance (Claude Code); ChatGPT used to review it. Pending review by the rest of the team |
| 2026-11-25 | Manuela | WP0 setup: maintai/ package with the contracts between modules (schemas.py) and stubs for every component, pipeline wired with deterministic offline arm, minimal App tab, .env.example, render.yaml (maintai, live mode on Render), dependencies (scikit-learn, google-genai, pydantic). Code written with AI assistance (Claude Code) and tested locally without API calls |
| 2026-11-25 | Manuela | Created the Render account and Blueprint, configured GEMINI_API_KEY on Render and deployed the first version (web service + PostgreSQL logs) at https://maintai-238a.onrender.com. Smoke-tested the App and Monitoring tabs online (2 requests, 2 ratings, stub components only) |
| 2026-11-25 | Manuela | Merged PR #1 (WP0) into main |
| 2026-11-26 | Manuela | WP7 interface (initial implementation against the stub components):
  - chose the Aerospace Blue identity with accessible colour variants, and designed the original logo and favicon;
  - built the CSS design system;
  - App tab: machine-reading form with sample readings, empty / loading / error / out-of-scope / development states, executive summary, risk gauge, failure-mode bars, physical indicators against the AI4I dataset rules, work order, financial insight consistent with economics.py, agent trace and feedback that reveals the bandit's writing style only after the vote;
  - Monitoring tab: executive dashboard built on monitor.py;
  - API: /run is now the only public endpoint.

  Design decisions, review of every iteration and visual approval by Manuela. Code written with AI assistance (Claude Code); ChatGPT used to review the requirements and draft prompts. Tested locally without API calls (docs/WP7_UI.md) |

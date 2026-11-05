# Group assignment: Ship it

As a team, build a small AI product for one business process, put it online, and run it like a product: it logs its own use, watches itself for drift, and you decide, with evidence, whether anyone should rely on it.

The product chains **two AI components from different modules of the course**: the output of one is the input of the other. Getting each piece to run is the easy part, and you have done most of it in A1–A3. The hard part is making the chain work end to end, measuring it, and keeping an eye on it once it is live.

Start with [SETUP.md](SETUP.md).

## Teams

- Record who built what in [CONTRIBUTIONS.md](CONTRIBUTIONS.md) as you go, not on the last day.

## The task

1. **Business process and decision.** Who uses the product, which decision or task it supports, what happens today without it (your **baseline**), how you measure success, and roughly what it is worth in euros. Write this before you choose any model.
2. **Two connected AI components** from different modules:
   - tabular ML (A1);
   - LLMs, RAG or agents (module 2, A3);
   - vision (module 3, A2).

   One component's output must be the next one's input, in one flow. Two demos in two tabs do not count. Pretrained models and APIs are fine, and you do not have to train anything from scratch. You may reuse your own A1–A3 code; say whose and where in REPORT.md.

   Examples, to give you the idea (don't copy them):
   - photo of a damaged parcel → a vision model classifies the damage → an agent drafts the claim decision with RAG over the carrier's policy;
   - product photo → a vision model extracts its attributes → an LLM writes the listing and checks it against the brand's guidelines;
   - a churn model flags customers at risk → an agent writes the retention offer from the customer's contract terms.
3. **Deploy it** as a Gradio app on a public Hugging Face Space, with an **App** tab and a **Monitoring** tab. API keys go in the Space's secrets, never in the repo.
4. **Log every request**: the inputs (summarised), the output of each component, model and prompt versions, latency, cost, and the user's 👍/👎. **No personal data**, neither in the inputs you demo with nor in the logs. The template's logger sends the logs to a private Hugging Face dataset, because a Space's disk is wiped when it restarts.
5. **Monitor it.** The Monitoring tab shows, from the logs:
   - at least one **quality signal over time** (share of 👎, share of cases sent to review, confidence…);
   - one **drift check** between live inputs and a reference set: PSI on a feature, the distribution of model confidence, or embedding distance;
   - an **alert rule**, with a threshold you justify.

   `main.py drift-demo` replays inputs that you shifted on purpose (darker photos, another language, another customer segment…). Your alert must fire on them and stay quiet on normal inputs.
6. **Evaluate.**
   - Each component, with the metric that fits it (as in A1–A3).
   - The whole product, on **30+ cases** you write in `eval/cases.csv`, compared with the baseline from step 1.
   - Where the output is free text, an LLM judge that you check against 20+ of your own labels, as in A3.
   - The cost per request, in tokens, euros and seconds.
7. **Decide.** Back to step 1, now with the numbers:
   - which cases go to a person: the rule that sends a case to review, and how many cases a day that would be;
   - **go / no-go / go with conditions**, the conditions, and the evidence behind them.
8. **Write REPORT.md, submit, and present.**

### Bonus: reinforcement learning (up to +10 %)

Add an RL layer that **changes how the app behaves based on the feedback the app itself collects**. A bandit is enough, for example:
- choosing among three prompt variants or answer styles and learning from 👍/👎;
- choosing the confidence threshold or the review rule and learning from reviewers' corrections;
- choosing which offer to show and learning from accept/reject.

The bonus counts only if REPORT.md and the demo show all of this:
1. **The loop closes:** feedback in the app → reward → policy update → behaviour visibly changes. Updates can be online or a batch job over the logs.
2. **The reward is justified:** what it rewards, what it could reward by accident (think of reward hacking), and how long the reward takes to arrive.
3. **Exploration** (ε-greedy, UCB or Thompson sampling) is chosen and justified, including what exploring costs the users who get the worse option.
4. **Some of the learning comes from your app's real logs.** You will have little traffic, so you may warm the policy up with a simulator. Say which part is simulated and which part is real.
5. **The risks:** feedback loops, who gives feedback and who doesn't, and calibration.

A bandit that is wired in but never learns from real feedback earns nothing.

## Code layout

```bash
uv run python app.py                 # the app, at http://127.0.0.1:7860
uv run python main.py evaluate       # component and end-to-end evaluation
uv run python main.py drift-demo     # replays shifted inputs; the alert must fire
uv run python deploy.py              # uploads the app to your Space
uv run python check.py               # what the grader runs
```

Put your components wherever suits you, but `app.py` only wires them together: the logic lives in modules that `main.py evaluate` can also import. Evaluation may be a notebook if `main.py evaluate` runs it.

**Your app must run on someone else's machine with someone else's keys.** That is how it is graded. Use only the APIs listed in SETUP.md (ask first if you need another one), read every key from the environment, list them in `.env.example`, and commit whatever models, indexes or reference sets the app needs (under 90 MB in total), or download them in a step that SETUP.md describes.

## Deliverables and submission

Deadlines are on Blackboard.

1. **Team registration** in the Blackboard groups.
2. **M1, one-page proposal:** the process, the user, the baseline, the two components and how they connect, the data, and an architecture sketch. One per team, on Blackboard. Pass/fail, with feedback.
3. **M2, walking skeleton:** the Space runs end to end (the components may still be dumb) and logs requests. Submit the Space URL on Blackboard. Pass/fail.
4. **Final submission** (see below).
5. **Presentation** in class. Your slot is published after the final submission.
6. **Peer assessment:** a form, filled in by each student.

M1 and M2 do not carry marks, but **a team that misses either one meets the professor** before going on. In the integration workshop, bring your live Space.

**Final submission.** One member forks this repo and adds the rest of the team as collaborators. Tag the final commit `final` (`git tag final && git push --tags`), open a PR to https://github.com/ami232/daiist-group-project, and submit on Blackboard, once per team: the PR link, the Space URL, and a zip of the repo as backup. After the deadline you may only fix things that keep the Space running; anything else is ignored.

## Presentation

- **BBA:** 6 minutes, then 4 minutes of questions. **PPLE:** 12 minutes, then 8 of questions.
- Show the **live product on the Space**, not screenshots: one normal case, one case it gets wrong or sends to review, and the Monitoring tab. A recorded backup video is allowed if the network fails.
- **The professor picks who answers each question**, so every member must be able to explain every part.
- The other students fill in a short form after each presentation, as potential users or buyers: did they understand the problem, does it work, would they trust it, would they buy it. You get their anonymised comments afterwards.

## Grading

**Hard requirement:** the professor can run the app from your `final` commit, with their own API keys, by following SETUP.md, and `check.py` passes. If your Space happens to be down when it is graded, that is not penalised as long as this holds.

| Component | Weight |
|---|---|
| Business case, baseline, value and go/no-go decision | 15% |
| Integration and technical execution: is the chain real, and does it work? | 30% |
| Evaluation: each component, end to end, cost | 20% |
| Operations: logging, monitoring, drift alert | 15% |
| Presentation, live demo and Q&A | 20% |
| *Bonus: reinforcement learning* | *up to +10%* |

The team grade cannot go above 100%.

**Your individual grade** is the team grade multiplied by a factor from the peer assessment. Each of you splits 100 points among all the members of your team, yourself included, according to how much each person contributed, and writes a sentence or two about each one. Your factor comes from the points your **teammates** give you; your own points do not count towards it. If the team splits evenly, everyone's factor is 1. Otherwise it ranges from 0.8 to 1.1, and your grade still cannot go above 100%. A factor below 1 needs at least two teammates to agree, and it is always checked against CONTRIBUTIONS.md, the commits and the Q&A.

## AI use

Allowed for code; disclose it in REPORT.md. The decisions (the problem, the components and how they connect, the metrics, the drift threshold, the review rule, go/no-go), the test cases, your own labels and the report's analysis must be the team's.

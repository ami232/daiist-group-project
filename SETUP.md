# Setup

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). One member forks this repo and adds the rest of the team as collaborators (*Settings → Collaborators*). Each of you clones the fork and runs `uv sync`.

Then:

```bash
uv run python app.py
```

You should see the template's placeholder app at <http://127.0.0.1:7860>, with its App and Monitoring tabs.

## API keys

You may use these services. If you need another one, ask first: the grader has to run your app with their own keys.

| Service | Variable | Free key |
|---|---|---|
| Gemini (LLMs, embeddings, vision) | `GEMINI_API_KEY` | <https://aistudio.google.com/apikey> |
| Roboflow (datasets and hosted models) | `ROBOFLOW_API_KEY` | <https://app.roboflow.com> → Settings → API Keys |
| Hugging Face (Space, logs, models) | `HF_TOKEN` | <https://huggingface.co/settings/tokens>, with *write* permission |

Local models (Hugging Face `transformers`, Ultralytics YOLO, scikit-learn, PyTorch) need no key.

```bash
cp .env.example .env    # paste your keys into .env
```

Never commit `.env`. Read keys only from the environment (`os.environ`). In `.env.example`, list **every** variable your app needs, with a comment saying what it is for. If a key is missing, the app should say which one, not crash with a stack trace.

Each member can use their own free keys. Free tiers have daily limits (for example, about 1,000 Gemini embeddings a day per project), so cache what you can: embeddings, model outputs in the evaluation, and anything else you would otherwise recompute.

## Your Space

1. One member creates a Hugging Face account and a new Space: <https://huggingface.co/new-space>, SDK **Gradio**, hardware **CPU basic** (free), visibility **public**. Add the others as members of the Space or of an organisation.
2. In the Space's *Settings → Variables and secrets*, add each key from `.env.example` as a **secret**.
3. In your `.env`, set `HF_SPACE=<user>/<space-name>`, then:

   ```bash
   uv run python deploy.py
   ```

   This uploads the app, writes the `requirements.txt` that Spaces needs from your `uv.lock`, and adds the Space's configuration. Do not edit the Space's files by hand: the repo is the source of truth.
4. Open the Space and check the build log (*Logs*) if it does not start.

A Space on free hardware **goes to sleep when nobody uses it** and takes a minute or more to wake up. Open it before your presentation.

A model that needs a GPU will not run on CPU basic. Use a smaller model, an API, or precompute its outputs for the cases you demo and say so in REPORT.md.

## Logs

A Space's disk is wiped every time it restarts, so the logger in the template (`logbook.py`) writes each request to `logs/` and, if `HF_LOG_REPO` is set, pushes that folder every few minutes to a **private** Hugging Face dataset.

1. Create the dataset: <https://huggingface.co/new-dataset>, visibility **private**, for example `<user>/<space-name>-logs`.
2. Add `HF_LOG_REPO=<user>/<space-name>-logs` to `.env` and to the Space's secrets.

Locally, without `HF_LOG_REPO`, logs stay in `logs/`, which is git-ignored. The Monitoring tab reads them from wherever they are.

Do not change the log fields the template defines. `check.py` and the grader rely on them. You may add your own.

## Dependencies

`uv add <package>`, then commit `pyproject.toml` and `uv.lock`. `deploy.py` takes the Space's `requirements.txt` from them.

## Check your submission

```bash
uv run python check.py                    # local: keys, app starts, one request end to end, log schema, evaluation files
uv run python check.py --space <url>      # also wakes your Space and sends it one request
```

`check.py` uses the keys in your environment, exactly as the grader will use theirs. A failure here is a failure for the grader too. Before you tag `final`, run it in a **fresh clone** of your fork, with a `.env` holding only the keys listed in `.env.example`. That is the quickest way to find a file you forgot to commit.

"""
Upload the app to your Hugging Face Space. `uv run python deploy.py [--secrets]`

Needs HF_TOKEN (write permission) and HF_SPACE (<user>/<space-name>) in .env.
It writes requirements.txt from uv.lock, adds the Space's configuration, and
uploads the repo (without .env, logs or the virtual environment). With
--secrets it also copies every variable in .env.example that is set in your
environment into the Space's secrets.

The repo is the source of truth: don't edit files on the Space by hand.
"""

import argparse
import os
import subprocess
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import HfApi

import env

ROOT = Path(__file__).resolve().parent
IGNORE = [
    ".env", ".venv/*", ".git/*", "logs/*", "logs_archive/*", "**/__pycache__/*",
    "**/.ipynb_checkpoints/*", ".github/*", "README.md", "*.pyc", ".DS_Store",
]
# The Space installs Gradio itself at sdk_version; pinning it again can clash.
SKIP_REQUIREMENTS = ("gradio==", "gradio-client==")

SPACE_README = """---
title: {title}
sdk: gradio
sdk_version: {gradio}
python_version: "3.12"
app_file: app.py
pinned: false
---

Deployed from {repo}.
"""


def write_requirements() -> None:
    exported = subprocess.run(
        ["uv", "export", "--format", "requirements-txt", "--no-hashes", "--no-dev",
         "--no-emit-project", "--no-header", "--no-annotate"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    lines = [line for line in exported.splitlines() if line and not line.startswith(SKIP_REQUIREMENTS)]
    (ROOT / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote requirements.txt ({len(lines)} packages).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--secrets", action="store_true", help="copy the variables in .env.example to the Space's secrets")
    args = parser.parse_args()

    env.load_env()
    token, space = os.environ.get("HF_TOKEN", "").strip(), os.environ.get("HF_SPACE", "").strip()
    if not token or not space:
        raise SystemExit("Set HF_TOKEN and HF_SPACE in .env first. See SETUP.md.")
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)

    write_requirements()
    origin = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    readme = SPACE_README.format(title=space.split("/")[-1], gradio=version("gradio"), repo=origin or "a local copy")
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=space, repo_type="space")
    api.upload_folder(folder_path=ROOT, repo_id=space, repo_type="space", ignore_patterns=IGNORE,
                      commit_message="Deploy from deploy.py")

    if args.secrets:
        for name, _ in env.declared():
            if name == "HF_SPACE":
                continue
            value = os.environ.get(name, "").strip()
            if value:
                api.add_space_secret(space, name, value)
                print(f"Secret {name} set.")
            else:
                print(f"Secret {name} not set: it is empty in your environment.")

    print(f"\nDeployed: https://huggingface.co/spaces/{space}")
    print("The Space rebuilds now; watch its Logs tab if it does not start.")


if __name__ == "__main__":
    main()

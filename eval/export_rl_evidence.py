"""
Export the bandit's evidence for the report and the presentation (WP6).

    uv run python eval/export_rl_evidence.py                     # logs as logbook reads them
    uv run python eval/export_rl_evidence.py --logs-dir logs     # a folder of JSON-lines logs,
                                                                 # e.g. after `main.py export-logs`
    uv run python eval/export_rl_evidence.py --out some/folder

AGGREGATES ONLY. No request ids, inputs, session tags or timestamps of single
ratings leave the logs: counts, rates, intervals and block means. No API calls.

Every file says what kind of data it holds:
    measured    computed from the logged requests and ratings (real traffic only
                counts for learning; see bandit.eligible_requests)
    simulated   copied from eval/bandit_*simulation*/calibration*, never mixed with
                measured numbers
    assumption  thresholds chosen by the team (listed in manifest.json)
If no rating is eligible yet, manifest.json says so in `warning`, and the files
show the neutral prior: that is not evidence that the policy learned anything.

Files (default folder eval/rl_evidence/):
    manifest.json            what, when, versions, counts by source, provenance, warning
    metrics.json             rl_monitor.evidence(): response rate, reward delay, sessions,
                             non-response, belief comparison, ...
    policy_snapshot.csv      current posterior per style (measured)
    policy_history.csv       posterior after each real rating event (measured)
    served_probabilities.csv arm_probs logged when requests were served, block means (measured)
    coverage_by_arm.csv      served / eligible / rated / counted per style (measured)
    belief_comparison.csv    observed reward vs posterior mean vs IPS vs P(selected) (measured)
    non_response.csv         response rate by risk band, review flag and style (measured)
    exclusions.csv           why live requests do not train the policy (measured)
    simulation_summary.json  the simulator's and the calibration's summaries (simulated)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _parse(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export aggregated RL evidence (no personal data, no API calls).")
    parser.add_argument("--logs-dir", help="read JSON-lines logs from this folder instead of logbook's default")
    parser.add_argument("--out", default=str(ROOT / "eval" / "rl_evidence"), help="output folder")
    return parser.parse_args(argv)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 - not a git checkout
        return None


def export(out: Path) -> dict:
    """Write every evidence file into `out`. Returns the manifest."""
    import pandas as pd

    import logbook
    from maintai import bandit, rl_monitor
    from ui import rl_dashboard

    out.mkdir(parents=True, exist_ok=True)
    logs = logbook.load()
    events = bandit.load_feedback_events()
    prompt_version = bandit.current_prompt_version()
    metrics = rl_monitor.evidence(logs, events, prompt_version)
    measured = {"data_type": "measured"}

    def csv(name: str, df: pd.DataFrame) -> None:
        df.assign(**measured).to_csv(out / name, index=False)

    csv("policy_snapshot.csv", bandit.policy_snapshot(logs, prompt_version=prompt_version))
    history = bandit.policy_history(logs, events, prompt_version=prompt_version)
    csv("policy_history.csv", history.drop(columns=["timestamp"], errors="ignore"))  # no single rating times
    served = rl_monitor.served_probabilities(logs).drop(columns=["timestamp"], errors="ignore")
    csv("served_probabilities.csv", served)
    csv("coverage_by_arm.csv", rl_monitor.coverage_by_arm(logs, prompt_version))
    csv("belief_comparison.csv", rl_monitor.belief_comparison(logs, prompt_version))
    csv("non_response.csv", rl_monitor.non_response(logs, prompt_version)["table"])
    live = rl_monitor.live_requests(logs)
    csv("exclusions.csv", pd.DataFrame(rl_dashboard.exclusions(live, prompt_version, logs), columns=["reason", "requests"]))
    (out / "metrics.json").write_text(json.dumps({"data_type": "measured", **metrics}, indent=2, default=str) + "\n",
                                      encoding="utf-8")

    sim = {"data_type": "simulated", "note": "Simulated users with declared like-rates. Never real feedback."}
    for key, name in (("simulation", "bandit_simulation_summary.json"), ("calibration", "bandit_calibration_summary.json")):
        path = ROOT / "eval" / name
        sim[key] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    (out / "simulation_summary.json").write_text(json.dumps(sim, indent=2) + "\n", encoding="utf-8")

    n_eligible = metrics["n_eligible_ratings"]
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "logs_backend": "postgres" if logbook.database_url() else "files",
        "policy_version": bandit.POLICY_VERSION, "prompt_version": prompt_version, "prior": "neutral Beta(1, 1)",
        "n_requests": metrics["n_requests"], "requests_by_source": metrics["requests_by_source"],
        "n_eligible_ratings": n_eligible,
        "warning": None if n_eligible else ("No eligible real rating yet: the snapshot and history show the neutral "
                                            "prior. This is not evidence that the policy learned anything."),
        "events_error": metrics["events_error"],
        "provenance": {
            "measured": ["metrics.json", "policy_snapshot.csv", "policy_history.csv", "served_probabilities.csv",
                         "coverage_by_arm.csv", "belief_comparison.csv", "non_response.csv", "exclusions.csv"],
            "simulated": ["simulation_summary.json"],
            "assumption": metrics["assumptions"],
        },
        "privacy": "Aggregates only: no request ids, inputs, session tags or single rating timestamps.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> None:
    args = _parse(argv)
    if args.logs_dir:
        os.environ["LOG_DIR"] = str(Path(args.logs_dir).resolve())  # before logbook is imported
    os.environ.setdefault("GEMINI_API_KEY", "")
    manifest = export(Path(args.out))
    print(f"Exported RL evidence to {args.out}: {manifest['n_requests']} request(s), "
          f"{manifest['n_eligible_ratings']} eligible real rating(s).")
    if manifest["warning"]:
        print("WARNING:", manifest["warning"])


if __name__ == "__main__":
    main()

"""Blind human labels for the work orders, for the judge's Cohen's kappa.  OWNER: WP4.

    uv run python main.py evaluate      # writes eval/to_label.csv (the orders, no verdicts)
    uv run python eval/label.py         # shows one order at a time; you type 1 (pass) or 0 (fail)

The judge's verdicts are NOT shown, so the label is ours and not a copy of the
judge's. Grade with the same rubric as the judge (eval/judge.py): an order passes
only if it addresses the risk, a technician could follow it, every fact comes
from the inputs or the tool results, and it is safe and consistent with the
action, priority and review.

Each label is appended to eval/human_labels.csv (id,label,comment) straight away,
so you can stop with q and go on later; ids already labelled are skipped.
Split the orders between the team (e.g. --start 0 --count 10) and aim for 20+.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

EVAL = Path(__file__).resolve().parent
TO_LABEL = EVAL / "to_label.csv"
LABELS = EVAL / "human_labels.csv"

RUBRIC = """Pass (1) only if ALL four are true, otherwise fail (0):
  R1 deals with the failure mode and risk the model reported
  R2 a technician could follow the steps without asking
  R3 every number, part, window or rule comes from the inputs or tool results
  R4 safe and consistent with action, priority and review"""


def show(row: dict) -> None:
    outputs = json.loads(row["outputs"])
    risk = outputs["component_a"]
    order = outputs["component_b"]["work_order"]
    draft = order["draft"]
    print("\n" + "=" * 72)
    print(f"Case {row['id']} ({row['kind']})")
    print(f"Reading: {row['input']}")
    modes = ", ".join(f"{m['mode']} {m['p']:.0%}" for m in risk.get("likely_modes", []))
    print(f"Risk: p_fail {risk['p_fail']:.1%}, band {risk['risk_band']}, modes [{modes}], "
          f"out of range {risk.get('ood_flags') or 'no'}")
    for call in outputs["component_b"].get("tool_calls", []):
        print(f"Tool {call['name']}({call['arguments']}) -> {json.dumps(call['result'], ensure_ascii=False)[:300]}")
    print(f"\nORDER  action {draft['action']}, priority {draft['priority']}, mode {draft['failure_mode_addressed']}, "
          f"review {'yes' if order['review'] else 'no'} {order.get('review_reasons') or ''}")
    for i, step in enumerate(draft.get("steps", []), 1):
        print(f"  {i}. {step}")
    print(f"  Why: {draft.get('justification', '')}")
    print(f"  Parts {draft.get('part_ids') or '-'}, window {draft.get('window_id') or '-'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Blind human labels for the work orders.")
    ap.add_argument("--start", type=int, default=0, help="first order of the shuffled list (to split the work)")
    ap.add_argument("--count", type=int, default=1000, help="how many orders to show")
    args = ap.parse_args()

    if not TO_LABEL.is_file():
        raise SystemExit("eval/to_label.csv is missing: run `uv run python main.py evaluate` first.")
    with TO_LABEL.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    random.Random(0).shuffle(rows)  # same order for everyone, so --start/--count split it cleanly
    rows = rows[args.start:args.start + args.count]

    done = set()
    if LABELS.is_file():
        with LABELS.open(newline="", encoding="utf-8") as f:
            done = {r["id"] for r in csv.DictReader(f) if (r.get("label") or "").strip() in ("0", "1")}
    else:
        LABELS.write_text("id,label,comment\n", encoding="utf-8")

    todo = [r for r in rows if r["id"] not in done]
    print(RUBRIC)
    print(f"\n{len(todo)} orders to label ({len(done)} already in human_labels.csv). q = stop, s = skip.")
    for row in todo:
        show(row)
        answer = ""
        while answer not in ("0", "1", "s", "q"):
            answer = input("\nLabel 1 = pass, 0 = fail (s skip, q quit): ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            continue
        comment = input("Short comment (what is wrong, or why it passes): ").strip()
        with LABELS.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row["id"], answer, comment])
    print(f"\nSaved in {LABELS.relative_to(EVAL.parent)}.")


if __name__ == "__main__":
    main()

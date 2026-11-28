"""Builds eval/reference.csv and eval/shifted.csv reproducibly (WP5).

reference.csv : 300 inputs sampled from data/train.csv, stratified by product type (L/M/H keep the train proportions),
                natural failure rate, fixed seed, excluding any input that is in eval/cases.csv.
                (60 was tried first: too noisy, see eval/drift_calibration.py)
shifted.csv   : 15 inputs from data/validation.csv with a deliberate, physically plausible shift:
                hotter workshop / weaker cooling (air and process temperature up) and more worn tools.
The test split is never read.

    uv run python eval/make_drift_sets.py
"""
import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED, N_REFERENCE, N_SHIFTED = 20261128, 300, 15
COLS = {"Type": "type", "Air temperature [K]": "air_temperature_k", "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rotational_speed_rpm", "Torque [Nm]": "torque_nm", "Tool wear [min]": "tool_wear_min"}


def to_input(row: pd.Series) -> str:
    d = {new: (row[old] if new == "type" else float(row[old])) for old, new in COLS.items()}
    d["rotational_speed_rpm"] = int(d["rotational_speed_rpm"])
    return json.dumps(d)


def write(path: Path, inputs: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["input"])
        for i in inputs:
            w.writerow([i])


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    valid = pd.read_csv(ROOT / "data" / "validation.csv")
    in_cases = {json.dumps(json.loads(r["input"]), sort_keys=True)
                for r in csv.DictReader((ROOT / "eval" / "cases.csv").open(encoding="utf-8")) if r["input"].startswith("{")}
    train = train[[json.dumps(json.loads(to_input(r)), sort_keys=True) not in in_cases for _, r in train.iterrows()]]
    ref = train.groupby("Type", group_keys=False).sample(frac=N_REFERENCE / len(train), random_state=SEED)
    write(ROOT / "eval" / "reference.csv", [to_input(r) for _, r in ref.iterrows()])

    shifted = valid.sample(N_SHIFTED, random_state=SEED).copy()
    shifted["Air temperature [K]"] = (shifted["Air temperature [K]"] + 6.0).round(1)
    shifted["Process temperature [K]"] = (shifted["Process temperature [K]"] + 7.0).round(1)
    shifted["Tool wear [min]"] = (shifted["Tool wear [min]"] + 90).clip(upper=290)
    write(ROOT / "eval" / "shifted.csv", [to_input(r) for _, r in shifted.iterrows()])
    print(f"reference.csv: {len(ref)} inputs; shifted.csv: {len(shifted)} inputs")


if __name__ == "__main__":
    main()

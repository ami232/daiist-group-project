"""Baselines B0, B1 and B2 of REPORT.md ("Baselines").  OWNER: WP4.

    uv run python eval/baselines.py      # tunes on train + validation, prints the test cost

- B0 · Reactive: never intervene.
- B1 · Tool-life rule: intervene when tool wear >= T. T tuned to minimise expected cost.
- B2 · Condition-monitoring rules: intervene if ANY single sensor crosses its limit:
  tool wear >= w, torque >= q, rotational speed <= r, or air-process temperature
  difference <= d. The four limits are tuned to minimise expected cost.

Tuning uses only data/train.csv + data/validation.csv; the test split is only used
to report. The cost of every outcome comes from maintai/economics.py, the same
model MaintAI is judged with, so the comparison is fair.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maintai import economics

DATA = ROOT / "data"
COLS = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
}


def load(split: str) -> pd.DataFrame:
    """One split with short column names and the temperature difference."""
    df = pd.read_csv(DATA / f"{split}.csv").rename(columns=COLS)
    df["temp_diff_k"] = df["process_temperature_k"] - df["air_temperature_k"]
    return df


def cost_per_1000(intervene: np.ndarray, failed: np.ndarray, reviewed: np.ndarray | None = None,
                  llm_eur: float = 0.0, a: economics.CostAssumptions = economics.DEFAULT) -> float:
    """Expected cost per 1,000 processes with the outcome table of REPORT.md."""
    intervene = np.asarray(intervene, bool)
    failed = np.asarray(failed, bool)
    reviewed = np.zeros_like(intervene) if reviewed is None else np.asarray(reviewed, bool)
    tp = intervene & failed
    fp = intervene & ~failed
    fn = ~intervene & failed
    cost = (tp * (a.intervention_eur + (1 - a.prevention_effectiveness) * a.failure_eur)
            + fp * a.intervention_eur + fn * a.failure_eur + reviewed * a.review_eur + llm_eur)
    return float(1000 * cost.mean())


def confusion(intervene: np.ndarray, failed: np.ndarray) -> dict:
    intervene = np.asarray(intervene, bool)
    failed = np.asarray(failed, bool)
    tp, fp = int((intervene & failed).sum()), int((intervene & ~failed).sum())
    fn = int((~intervene & failed).sum())
    n = len(failed)
    return {
        "recall": round(tp / max(tp + fn, 1), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "interventions_per_1000": round(1000 * (tp + fp) / n, 1),
        "missed_failures_per_1000": round(1000 * fn / n, 1),
        "unnecessary_interventions_per_1000": round(1000 * fp / n, 1),
    }


# ----------------------------------------------------------------- policies ---

@dataclass(frozen=True)
class B1:
    wear_min: float

    name = "B1 tool-life rule"

    def decide(self, df: pd.DataFrame) -> np.ndarray:
        return (df["tool_wear_min"] >= self.wear_min).to_numpy()

    def describe(self) -> str:
        return f"intervene when tool wear >= {self.wear_min:.0f} min"


@dataclass(frozen=True)
class B2:
    wear_min: float
    torque_nm: float
    rpm_max: float
    temp_diff_k: float

    name = "B2 condition-monitoring rules"

    def decide(self, df: pd.DataFrame) -> np.ndarray:
        return ((df["tool_wear_min"] >= self.wear_min) | (df["torque_nm"] >= self.torque_nm)
                | (df["rotational_speed_rpm"] <= self.rpm_max) | (df["temp_diff_k"] <= self.temp_diff_k)).to_numpy()

    def describe(self) -> str:
        parts = []
        if self.wear_min < OFF["wear_min"]:
            parts.append(f"tool wear >= {self.wear_min:.0f} min")
        if self.torque_nm < OFF["torque_nm"]:
            parts.append(f"torque >= {self.torque_nm:.1f} Nm")
        if self.rpm_max > OFF["rpm_max"]:
            parts.append(f"speed <= {self.rpm_max:.0f} rpm")
        if self.temp_diff_k > OFF["temp_diff_k"]:
            parts.append(f"temperature difference <= {self.temp_diff_k:.1f} K")
        return "intervene if " + " or ".join(parts) if parts else "never intervene"


def b0_decide(df: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(df), bool)


# ------------------------------------------------------------------ tuning ---

OFF = {"wear_min": 10_000.0, "torque_nm": 10_000.0, "rpm_max": -1.0, "temp_diff_k": -100.0}
GRIDS = {
    "wear_min": np.arange(150, 255, 1.0),
    "torque_nm": np.arange(45, 80, 0.5),
    "rpm_max": np.arange(1150, 1500, 5.0),
    "temp_diff_k": np.arange(7.5, 10.05, 0.05),
}


def tune_b1(df: pd.DataFrame, a: economics.CostAssumptions = economics.DEFAULT) -> B1:
    failed = df["failure"].to_numpy(bool)
    costs = {t: cost_per_1000(B1(t).decide(df), failed, a=a) for t in [OFF["wear_min"]] + list(GRIDS["wear_min"])}
    return B1(min(costs, key=costs.get))


def tune_b2(df: pd.DataFrame, a: economics.CostAssumptions = economics.DEFAULT, rounds: int = 4) -> B2:
    """Coordinate descent: each limit in turn, the others fixed. Any limit may be switched off."""
    failed = df["failure"].to_numpy(bool)
    params = dict(OFF)
    for _ in range(rounds):
        changed = False
        for name, grid in GRIDS.items():
            candidates = [OFF[name]] + list(grid)  # OFF first: on a tie, keep the rule simpler
            costs = {v: cost_per_1000(B2(**{**params, name: v}).decide(df), failed, a=a) for v in candidates}
            best = min(costs, key=costs.get)
            if best != params[name]:
                params[name], changed = best, True
        if not changed:
            break
    return B2(**{k: float(v) for k, v in params.items()})


def tuned(a: economics.CostAssumptions = economics.DEFAULT) -> tuple[B1, B2, str]:
    """B1 and B2 tuned on train + validation, and the name of the main baseline.

    The main baseline is the better of B1 and B2 on the validation split (REPORT.md).
    Pass other cost assumptions `a` to re-tune them for a sensitivity setting
    (eval/economics_sensitivity.py). Baselines are fixed rules: no human review
    and no AI cost, so `reviewed` stays empty and `llm_eur` is 0 for them.
    """
    dev = pd.concat([load("train"), load("validation")], ignore_index=True)
    val = load("validation")
    b1, b2 = tune_b1(dev, a), tune_b2(dev, a)
    failed = val["failure"].to_numpy(bool)
    main = "B1" if cost_per_1000(b1.decide(val), failed, a=a) <= cost_per_1000(b2.decide(val), failed, a=a) else "B2"
    return b1, b2, main


def report() -> pd.DataFrame:
    b1, b2, main = tuned()
    test = load("test")
    failed = test["failure"].to_numpy(bool)
    rows = []
    for name, decide, rule in (("B0 reactive", b0_decide, "never intervene"),
                               (b1.name, b1.decide, b1.describe()),
                               (b2.name, b2.decide, b2.describe())):
        intervene = decide(test)
        rows.append({"policy": name, "rule": rule, "cost_per_1000_eur": round(cost_per_1000(intervene, failed), 1),
                     **confusion(intervene, failed)})
    out = pd.DataFrame(rows)
    out["main_baseline"] = out["policy"].str.startswith(main)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_colwidth", 120)
    print(report().to_string(index=False))

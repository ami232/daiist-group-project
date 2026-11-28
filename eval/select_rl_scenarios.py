"""
Scenarios for the real feedback campaign (WP6, docs/RL_FEEDBACK_PROTOCOL.md).

    uv run python eval/select_rl_scenarios.py               # writes eval/rl_scenarios.csv
    uv run python eval/select_rl_scenarios.py --order 3     # sheet for participant 3: 15 readings, random order, no labels

VALIDATION SPLIT ONLY (data/validation.csv). The test split is never read: it is
kept for the final evaluation. No API calls, no feedback, nothing written to the logs.

How the 30 readings are chosen (fixed seed, so anyone gets the same file):
    - 10 readings whose process failed (label `Machine failure` = 1), one per failure
      mode first (TWF, HDF, PWF, OSF, RNF), then the rest at random;
    - 20 readings that did not fail, spread over the machine conditions below and
      over product types L / M / H, so participants see calm and stressed machines.
Machine conditions come from the physical rules in the AI4I documentation
(maintai/features.py), not from a model:
    tool_end      tool wear >= 200 min
    overstrain    tool wear x torque >= 90% of the type's overstrain limit
    heat          process - air temperature < 8.6 K and speed < 1380 rpm
    power         power < 3500 W or > 9000 W
    normal        none of the above

Risk bands. The band shown to participants comes from component A. While
component_a is a stub (MODEL_VERSION starts with "stub"), the bands are NOT real,
so the file is marked `status = provisional` and the selection only uses labels and
conditions. When WP1's model is merged, re-run this script: it then reads the real
band of each reading and requires at least 4 readings per band (low, medium, high,
critical) where the split has them; the file is marked `status = final`.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maintai import component_a, features  # noqa: E402
from maintai.schemas import MachineInput, MachineType  # noqa: E402

SEED = 20261128
N_FAILED, N_OK = 10, 20
MIN_PER_BAND = 4
OUT = ROOT / "eval" / "rl_scenarios.csv"
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
COLUMNS = {"Type": "type", "Air temperature [K]": "air_temperature_k", "Process temperature [K]": "process_temperature_k",
           "Rotational speed [rpm]": "rotational_speed_rpm", "Torque [Nm]": "torque_nm", "Tool wear [min]": "tool_wear_min"}


def _machine(row: pd.Series) -> MachineInput:
    return MachineInput.model_validate({v: row[k] for k, v in COLUMNS.items()})


def condition(m: MachineInput) -> str:
    ind = features.indicators(m)
    if ind.tool_wear_min >= 200:
        return "tool_end"
    if ind.strain_min_nm >= 0.9 * ind.strain_limit_min_nm:
        return "overstrain"
    if ind.temp_diff_k < 8.6 and m.rotational_speed_rpm < 1380:
        return "heat"
    if ind.power_w < 3500 or ind.power_w > 9000:
        return "power"
    return "normal"


def load_validation() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "validation.csv")
    machines = [_machine(r) for _, r in df.iterrows()]
    df["condition"] = [condition(m) for m in machines]
    real_model = not component_a.MODEL_VERSION.startswith("stub")
    df["risk_band"] = [component_a.assess(m).risk_band.value if real_model else "pending" for m in machines]
    df.attrs["real_model"] = real_model
    return df


def select(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick: list[int] = []

    def take(pool: pd.DataFrame, k: int) -> None:
        pool = pool[~pool.index.isin(pick)]
        if k > 0 and len(pool):
            pick.extend(pool.index[rng.permutation(len(pool))[:k]].tolist())

    failed = df[df["Machine failure"] == 1]
    for mode in MODES:
        take(failed[failed[mode] == 1], 1)
    take(failed, N_FAILED - len(pick))

    ok = df[df["Machine failure"] == 0]
    stressed = [c for c in ("tool_end", "overstrain", "heat", "power") if (ok["condition"] == c).any()]
    for c in stressed:  # two of each stressed condition that exists among non-failures
        take(ok[ok["condition"] == c], 2)
    for t in (x.value for x in MachineType):  # the rest: normal machines, balanced by type
        take(ok[(ok["condition"] == "normal") & (ok["Type"] == t)], (N_FAILED + N_OK - len(pick)) // 3)
    take(ok, N_FAILED + N_OK - len(pick))

    if df.attrs.get("real_model"):  # make sure every band is represented, swapping in readings if needed
        for band in ("low", "medium", "high", "critical"):
            have = int((df.loc[pick, "risk_band"] == band).sum())
            extra = df[(df["risk_band"] == band) & ~df.index.isin(pick)]
            for idx in extra.index[rng.permutation(len(extra))[:max(0, MIN_PER_BAND - have)]]:
                counts = df.loc[pick, "risk_band"].value_counts()
                donor = [i for i in pick if counts[df.loc[i, "risk_band"]] > MIN_PER_BAND]
                if donor:
                    pick.remove(donor[-1])
                    pick.append(idx)

    out = df.loc[pick].copy()
    out["raw_input"] = [json.dumps({v: (r[k] if k == "Type" else float(r[k])) for k, v in COLUMNS.items()})
                        for _, r in out.iterrows()]
    out["failure_mode"] = [next((m for m in MODES if r[m] == 1), "none") for _, r in out.iterrows()]
    out = out.sort_values("UDI").reset_index(drop=True)
    out.insert(0, "scenario_id", [f"S{i + 1:02d}" for i in range(len(out))])
    out["split"] = "validation"
    out["status"] = "final" if df.attrs.get("real_model") else "provisional"
    out["model_version"] = component_a.MODEL_VERSION
    return out[["scenario_id", "UDI", "split", "Type", "condition", "Machine failure", "failure_mode",
                "risk_band", "status", "model_version", "raw_input"]]


def participant_order(n_scenarios: int, participant: int, seed: int = SEED) -> list[int]:
    """A different random order for each participant (reproducible from the participant number)."""
    return (np.random.default_rng([seed, participant]).permutation(n_scenarios) + 1).tolist()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Select validation readings for the feedback campaign.")
    parser.add_argument("--order", type=int, help="print the scenario order for this participant number")
    args = parser.parse_args(argv)
    if args.order is not None:  # the participant's sheet: inputs only, never the label or the failure mode
        table = pd.read_csv(OUT).set_index("scenario_id")
        ids = [f"S{i:02d}" for i in participant_order(len(table), args.order)][:15]
        print(f"Participant {args.order}: enter these 15 readings in this order, one request each, and rate every order.")
        for sid in ids:
            print(f"  {sid}  {table.loc[sid, 'raw_input']}")
        return
    table = select(load_validation())
    table.to_csv(OUT, index=False)
    print(table.drop(columns=["raw_input"]).to_string(index=False))
    print(f"\nWrote {OUT.relative_to(ROOT)}: {len(table)} validation readings, status = {table['status'].iloc[0]}.")
    if table["status"].iloc[0] == "provisional":
        print("Risk bands are pending: component A is still a stub. Re-run after WP1's model is merged.")


if __name__ == "__main__":
    main()

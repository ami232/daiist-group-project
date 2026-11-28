"""WP5: calibrate the drift alert on VALIDATION only (the test split is never read).

For each candidate rule (smoothing, PSI_ALERT, NOISE_MULTIPLE) and each window size n it estimates
  - false alarm rate: share of random windows of n NORMAL validation inputs that raise any alert
    (any of the 9 features, so the multiple-feature effect is included);
  - detection rate: share of windows of n SHIFTED validation inputs (two shift sizes) that raise an alert.
The reference set is the real one (monitoring/reference_features.csv); repeated also over 20 other
random 300-input reference sets from train, to show the false alarm rate does not depend on one lucky draw.

    uv run python eval/drift_calibration.py        # writes eval/drift_calibration.csv and ..._summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import monitor  # noqa: E402
from maintai import component_a, features, inputs  # noqa: E402

SEED, REPS, WINDOWS = 20261128, 200, (16, 20, 50, 100)
COLS = {"Type": "type", "Air temperature [K]": "air_temperature_k", "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rotational_speed_rpm", "Torque [Nm]": "torque_nm", "Tool wear [min]": "tool_wear_min"}
SHIFTS = {"large": (6.0, 7.0, 90), "moderate": (3.0, 3.5, 45)}  # (air K, process K, tool wear min)


def feats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        raw = {new: (r[old] if new == "type" else float(r[old])) for old, new in COLS.items()}
        raw["rotational_speed_rpm"] = int(raw["rotational_speed_rpm"])
        m = inputs.parse(json.dumps(raw))
        rows.append(features.drift_features(m, component_a.assess(m)))
    return pd.DataFrame(rows)


def alarm(reference: pd.DataFrame, window: pd.DataFrame, smoothing: float, psi_alert: float, multiple: float) -> bool:
    monitor.SMOOTHING, monitor.PSI_ALERT, monitor.NOISE_MULTIPLE = smoothing, psi_alert, multiple
    return bool(monitor.drift(reference, window)["alert"].any())


def main() -> None:
    rng = np.random.default_rng(SEED)
    valid = pd.read_csv(ROOT / "data" / "validation.csv")
    train = pd.read_csv(ROOT / "data" / "train.csv")
    normal_v = feats(valid)
    shifted_v = {}
    for name, (da, dp, dw) in SHIFTS.items():
        s = valid.copy()
        s["Air temperature [K]"] += da
        s["Process temperature [K]"] += dp
        s["Tool wear [min]"] = (s["Tool wear [min]"] + dw).clip(upper=290)
        shifted_v[name] = feats(s)
    reference = monitor.load_reference()
    alt_refs = [feats(train.sample(300, random_state=int(rng.integers(1e9)))) for _ in range(10)]

    rules = [("clip (template)", 0.0, 0.25, 2), ("smoothing 0.5", 0.5, 0.25, 2), ("smoothing 0.5", 0.5, 0.25, 3),
             ("smoothing 0.5", 0.5, 0.25, 4), ("smoothing 0.5", 0.5, 0.25, 5), ("smoothing 0.5", 0.5, 0.25, 6)]
    out = []
    for label, sm, pa, mult in rules:
        for n in WINDOWS:
            idx = [rng.choice(len(normal_v), n, replace=False) for _ in range(REPS)]
            far = np.mean([alarm(reference, normal_v.iloc[i], sm, pa, mult) for i in idx])
            far_alt = np.mean([alarm(alt_refs[k % 10], normal_v.iloc[i], sm, pa, mult) for k, i in enumerate(idx)])
            row = {"rule": label, "smoothing": sm, "psi_alert": pa, "noise_multiple": mult, "window": n,
                   "false_alarm_rate": round(far, 4), "false_alarm_rate_other_refs": round(far_alt, 4)}
            for name, sv in shifted_v.items():
                row[f"detection_{name}"] = round(np.mean([alarm(reference, sv.iloc[i], sm, pa, mult) for i in idx]), 4)
            out.append(row)
            print({k: float(v) if hasattr(v, 'item') else v for k, v in row.items()}, flush=True)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "eval" / "drift_calibration.csv", index=False)
    print(f"{REPS} windows per cell; reference n={len(reference)}; validation rows={len(valid)}")


if __name__ == "__main__":
    main()

# WP5 · Drift monitoring: design, calibration and results

All numbers come from `eval/drift_calibration.py` (validation split only; the test split is never read) and `main.py drift-demo`. Reproduce with `uv run python eval/make_drift_sets.py && uv run python main.py reference && uv run python eval/drift_calibration.py && uv run python main.py drift-demo`. None of it calls Gemini (features come from component A; the agent is not needed).

## What is watched
Nine numeric features (`maintai/features.py`): air and process temperature, rpm, torque, tool wear, temperature difference, power, product type code and `p_fail`. Out-of-scope inputs have no features and are not counted.

## Data
- `eval/reference.csv`: 300 inputs from `data/train.csv`, stratified by product type, natural failure rate, fixed seed, none of them in `eval/cases.csv`. A first version with 60 inputs gave 40% false alarms even on windows of 100, so it was enlarged (the brief asks for at least 30).
- `eval/shifted.csv`: 15 validation inputs shifted on purpose: workshop hotter (air +6 K, process +7 K, i.e. weaker cooling) and tools 90 min more worn. All values stay inside the physical limits of `MachineInput`.
- `monitoring/reference_features.csv`: features of the reference, built with the real model (`maintai-gb-sigmoid-v1`). It must be rebuilt if the model or the feature list changes.

## Rule and why
PSI with 5 reference-quantile bins. Two changes to the template: (1) additive smoothing of 0.5 per bin instead of clipping at 1e-4, because one empty bin in a window of 16 added about 1.5 to the PSI on its own; (2) the alert needs PSI > max(0.25, 5 x the PSI expected by chance), instead of 2 x. With 9 features and small windows the template rule raised false alarms in 31-47% of normal windows of 16-20 requests, 12% at 50 and 1% at 100. Calibration grid on 200 random validation windows per cell, 4 window sizes, and 10 alternative random references:

| Rule | False alarm, n=16 / 20 / 50 / 100 | Detection of the large shift | Detection of the moderate shift (half size) |
|---|---|---|---|
| template (clip, x2) | 31% / 47% / 12% / 1% | 100% | 100% |
| smoothing, x3 | 8% / 6% / 13% / 0.5% | 100% | 100% |
| smoothing, x4 | 0% / 1% / 0.5% / 1% | 100% | 100% |
| **smoothing, x5 (chosen)** | **0% / 0% / 0.5% / 0.5%** | 100% | 99.5% (n=16) to 100% |
| smoothing, x6 | 0% / 0% / 0% / 0% | 100% | 97.5% (n=16) |

x5 is the smallest multiple that keeps false alarms at or below about 1% per window in every window size while still detecting the moderate shift. The 1% target is an internal goal, estimated on 200 windows (so +/- about 1 point), not a guarantee. Windows overlap in time in production, so the real alarm rate per day can be higher than per independent window.

## drift-demo result (`main.py drift-demo`)
- 16 `normal` cases of `eval/cases.csv`: quiet (largest PSI 0.52 against thresholds around 1.3).
- 15 shifted inputs: alert on `air_temp_k` (PSI 2.21), `process_temp_k` (2.33) and `p_fail` (1.51).

## Limits (to state in the REPORT)
- The shifted set is synthetic and was built by us, so the detection rate shows the check works for shifts of that size, not for real drift. Small shifts may not alert.
- The 16 `normal` cases were chosen by WP4 and 6 of them are real failures; they are not random traffic. The 0% false alarm rate on random validation windows is the more reliable number.
- `p_fail` as a feature means drift also fires when the model's risk output changes, which is intended.
- With fewer than 20 requests in the window the app stays silent (`MIN_WINDOW`).
- The thumbs-down alert (`THUMBS_DOWN_ALERT = 0.4`) is not calibrated yet.

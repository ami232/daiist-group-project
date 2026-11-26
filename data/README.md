# MaintAI — Dataset Documentation

## Dataset Source

- **Name:** AI4I 2020 Predictive Maintenance Dataset
- **Source:** UCI Machine Learning Repository
- **Year:** 2020
- **URL:** https://archive.ics.uci.edu/dataset/601/ai4i

## Dataset Overview
The dataset contains 10,000 synthetic industrial observations and 14 columns. It represents machine operating conditions and associated failure outcomes.

### Input Features

| Feature | Description |
|---|---|
| Type | Product quality variant (L, M, H) |
| Air temperature [K] | Ambient temperature |
| Process temperature [K] | Operating temperature |
| Rotational speed [rpm] | Machine rotational speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Accumulated tool wear |

The primary prediction target is `Machine failure`.

UDI, Product ID and the five failure-mode labels are excluded from the model's input features to prevent identifier dependence and target leakage.

## Data Quality

- Total observations: 10,000
- Missing values: 0
- Duplicate rows: 0
- Unique identifiers: 10,000
- Machine failures: 339
- Failure rate: 3.39%
- Failure-mode flags without a machine failure: 18
- Machine failures without a failure-mode flag: 9

The original labels are preserved without modification.

## Dataset Splits

A reproducible stratified random split was generated using scikit-learn with random seed 42, stratifying by the Machine failure target.

| Split | Observations | Failures |
|---|---:|---:|
| Training | 6,000 | 203 |
| Validation | 2,000 | 68 |
| Test | 2,000 | 68 |

The training set is used for model fitting and internal probability calibration. The validation set is used for model selection and threshold optimization. The test set is reserved for final evaluation.

All five failure modes are represented in every split.

## Reproducibility

To regenerate the dataset splits, run:

`uv run python training/prepare_data.py`

## Limitations
- The dataset is synthetic and may not fully represent real industrial environments.
- The target is highly imbalanced.
- Inconsistencies exist between the overall failure label and individual failure-mode labels.
- Random splitting may introduce optimistic performance estimates due to sequential correlations between observations.
- The dataset does not contain time-to-failure or maintenance intervention outcomes.
- The model estimates failure risk under observed operating conditions rather than predicting the exact timing of future breakdowns.

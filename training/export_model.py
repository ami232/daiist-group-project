from pathlib import Path
import json
import hashlib

import joblib
import pandas as pd
import sklearn

from training.feature_engineering import build_features
from training.optimize_threshold import get_training_ranges

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAINING_DIR = ROOT / "training"
MODELS_DIR = ROOT / "models"

MODEL_NAME = "Gradient Boosting - Sigmoid"
MODEL_VERSION = "maintai-gb-sigmoid-v1"
THRESHOLD = 0.19
SEED = 42


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    candidates = joblib.load(
        TRAINING_DIR / "candidate_models.joblib"
    )

    model = candidates[MODEL_NAME]

    joblib.dump(
        model,
        MODELS_DIR / "failure_model.joblib",
        compress=3
    )

    train = pd.read_csv(DATA_DIR / "train.csv")

    X_train = build_features(
        train, engineered=True
    )

    ranges = get_training_ranges(X_train)

    results = pd.read_csv(
        TRAINING_DIR / "economic_comparison.csv"
    )

    selected = results[
        results["candidate"] == MODEL_NAME
    ].iloc[0]

    mode_results = pd.read_csv(
        TRAINING_DIR / "failure_mode_comparison.csv"
    )

    dataset_hash = hashlib.sha256(
        (DATA_DIR / "ai4i2020.csv").read_bytes()
    ).hexdigest()

    model_card = {
        "model_version": MODEL_VERSION,
        "algorithm": "Gradient Boosting",
        "calibration": {
            "method": "sigmoid",
            "cv_folds": 5,
            "ensemble": False
        },
        "hyperparameters": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": SEED
        },
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "name": "AI4I 2020 Predictive Maintenance",
            "source": "UCI Machine Learning Repository",
            "license": "CC BY 4.0",
            "sha256": dataset_hash,
            "split_seed": SEED,
            "split_method": "stratified random",
            "training_rows": 6000,
            "validation_rows": 2000,
            "test_rows": 2000
        },
        "target": "Machine failure",
        "features": list(X_train.columns),
        "training_ranges": ranges,
        "threshold": THRESHOLD,
        "critical_threshold": 0.8,
        "uncertainty_band": {
            "lower": THRESHOLD / 2,
            "upper": THRESHOLD * 2,
            "lower_inclusive": True,
            "upper_inclusive": False
        },
        "review_policy": [
            "critical_risk",
            "uncertain_prediction",
            "out_of_training_range"
        ],
        "cost_assumptions": {
            "intervention_eur": 300,
            "failure_eur": 4000,
            "prevention_effectiveness": 0.8,
            "review_eur": 15,
            "llm_eur": 0
        },
        "validation_metrics": selected.to_dict(),
        "failure_mode_metrics": mode_results.to_dict(
            orient="records"
        ),
        "failure_modes": {
            "TWF": "Low reliability; advisory only",
            "HDF": "Predictable in synthetic data",
            "PWF": "Predictable in synthetic data",
            "OSF": "Predictable in synthetic data",
            "RNF": "Excluded: random by design"
        },
        "test_evaluated": False,
        "limitations": [
            "Synthetic dataset",
            "Highly imbalanced target",
            "27 inconsistencies between failure labels",
            "Potential sequential correlation across random splits",
            "No time-to-failure information",
            "No observed maintenance intervention outcomes",
            "Limited reliability of tool wear failure predictions",
            "Economic assumptions are simulated",
            "Validation costs exclude LLM costs",
            "Requires human oversight"
        ]
    }

    with open(
        MODELS_DIR / "model_card.json",
        "w"
    ) as file:
        json.dump(model_card, file, indent=4)

    print("Selected model:", MODEL_NAME)
    print("Model version:", MODEL_VERSION)
    print("Decision threshold:", THRESHOLD)
    print("Model and model card saved successfully.")

    print("\nMODEL FILES")

    for path in MODELS_DIR.iterdir():
        size = path.stat().st_size / 1024
        print(f"{path.name}: {size:.2f} KB")


if __name__ == "__main__":
    main()
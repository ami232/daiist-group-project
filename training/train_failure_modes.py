from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from training.feature_engineering import build_features
from training.train import create_preprocessor, evaluate_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
TRAINING_DIR = ROOT / "training"

SEED = 42

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF"]


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = build_features(train, engineered=True)

    X_validation = build_features(
        validation, engineered=True
    )

    MODELS_DIR.mkdir(exist_ok=True)

    mode_models = {}
    results = []

    for mode in FAILURE_MODES:
        print(f"\nTraining failure mode: {mode}")

        y_train = train[mode]
        y_validation = validation[mode]

        pipeline = Pipeline([
            ("preprocessor", create_preprocessor(X_train)),
            ("classifier", GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                random_state=SEED
            ))
        ])

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED
        )

        model = CalibratedClassifierCV(
            estimator=pipeline,
            method="sigmoid",
            cv=cv,
            ensemble=False
        )

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model, X_validation, y_validation
        )

        mode_models[mode] = model

        results.append({
            "failure_mode": mode,
            "training_positives": int(y_train.sum()),
            "validation_positives": int(y_validation.sum()),
            **metrics
        })

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    joblib.dump(
        mode_models,
        MODELS_DIR / "failure_modes.joblib",
        compress=3
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        TRAINING_DIR / "failure_mode_comparison.csv",
        index=False
    )

    columns = [
        "failure_mode",
        "training_positives",
        "validation_positives",
        "precision",
        "recall",
        "roc_auc",
        "average_precision",
        "brier_score"
    ]

    print("\nFAILURE MODE COMPARISON")

    print(
        results_df[columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nFailure-mode models saved successfully.")


if __name__ == "__main__":
    main()
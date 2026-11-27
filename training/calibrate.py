from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from training.feature_engineering import build_features
from training.train import (
    create_preprocessor,
    get_models,
    evaluate_model
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAINING_DIR = ROOT / "training"

SEED = 42
TARGET = "Machine failure"


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = build_features(train, engineered=True)
    X_validation = build_features(validation, engineered=True)

    y_train = train[TARGET]
    y_validation = validation[TARGET]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    results = []
    candidates = {}

    for model_name in [
        "Random Forest",
        "Gradient Boosting"
    ]:
        for method in [
            "Uncalibrated",
            "Sigmoid",
            "Isotonic"
        ]:
            print(f"\n{model_name} - {method}")

            classifier = get_models()[model_name]

            pipeline = Pipeline([
                ("preprocessor", create_preprocessor(X_train)),
                ("classifier", classifier)
            ])

            if method == "Uncalibrated":
                model = pipeline
            else:
                model = CalibratedClassifierCV(
                    estimator=pipeline,
                    method=method.lower(),
                    cv=cv,
                    ensemble=False
                )

            start = time.perf_counter()

            model.fit(X_train, y_train)

            training_time = time.perf_counter() - start

            metrics = evaluate_model(
                model,
                X_validation,
                y_validation
            )

            result = {
                "model": model_name,
                "calibration": method,
                "training_time_seconds": training_time,
                **metrics
            }

            results.append(result)

            candidates[f"{model_name} - {method}"] = model

            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "brier_score",
        ascending=True
    )

    results_df.to_csv(
        TRAINING_DIR / "calibration_comparison.csv",
        index=False
    )

    joblib.dump(
        candidates,
        TRAINING_DIR / "candidate_models.joblib",
        compress=3
    )

    print("\nCALIBRATION COMPARISON")

    columns = [
        "model",
        "calibration",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "brier_score"
    ]

    print(
        results_df[columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
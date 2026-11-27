from pathlib import Path
import time

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)

from training.feature_engineering import build_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAINING_DIR = ROOT / "training"

SEED = 42
TARGET = "Machine failure"


def create_preprocessor(X):
    categorical = ["type"]
    numerical = [
        col for col in X.columns
        if col not in categorical
    ]

    return ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical
        ),
        (
            "numerical",
            StandardScaler(),
            numerical
        )
    ])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=2
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=SEED
        )
    }


def evaluate_model(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(
            y, probabilities
        ),
        "brier_score": brier_score_loss(
            y, probabilities
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    y_train = train[TARGET]
    y_validation = validation[TARGET]

    results = []

    for feature_set, engineered in [
        ("Raw", False),
        ("Engineered", True)
    ]:
        X_train = build_features(
            train, engineered=engineered
        )

        X_validation = build_features(
            validation, engineered=engineered
        )

        for model_name, classifier in get_models().items():
            print(f"\nTraining {model_name} - {feature_set}")

            model = Pipeline([
                ("preprocessor", create_preprocessor(X_train)),
                ("classifier", classifier)
            ])

            start = time.perf_counter()

            model.fit(X_train, y_train)

            training_time = time.perf_counter() - start

            metrics = evaluate_model(
                model, X_validation, y_validation
            )

            result = {
                "model": model_name,
                "features": feature_set,
                "training_time_seconds": training_time,
                **metrics
            }

            results.append(result)

            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "average_precision",
        ascending=False
    )

    results_df.to_csv(
        TRAINING_DIR / "model_comparison.csv",
        index=False
    )

    print("\nMODEL COMPARISON")

    columns = [
        "model",
        "features",
        "accuracy",
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

    print("\nResults saved to training/model_comparison.csv")


if __name__ == "__main__":
    main()
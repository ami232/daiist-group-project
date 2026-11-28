from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    average_precision_score
)

from maintai.economics import (
    DEFAULT,
    outcome_cost,
    break_even_probability
)

from training.feature_engineering import build_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAINING_DIR = ROOT / "training"

TARGET = "Machine failure"
MIN_RECALL = 0.80
CRITICAL_THRESHOLD = 0.80

THRESHOLDS = np.unique(
    np.append(
        np.round(np.arange(0.01, 0.99, 0.01), 2),
        break_even_probability()
    )
)


def get_training_ranges(X):
    numerical = X.select_dtypes(include="number")

    return {
        column: {
            "min": float(numerical[column].min()),
            "max": float(numerical[column].max())
        }
        for column in numerical.columns
    }


def detect_ood(X, ranges):
    flags = np.zeros(len(X), dtype=bool)

    for column, limits in ranges.items():
        flags |= (
            (X[column] < limits["min"]) |
            (X[column] > limits["max"])
        ).to_numpy()

    return flags


def evaluate_threshold(y, probabilities, threshold, ood):
    intervened = probabilities >= threshold

    uncertainty = (
        (probabilities >= threshold / 2) &
        (probabilities < threshold * 2)
    )

    critical = probabilities >= CRITICAL_THRESHOLD

    reviewed = uncertainty | critical | ood

    costs = np.array([
        outcome_cost(
            intervened=bool(i),
            failed=bool(f),
            reviewed=bool(r),
            llm_eur=0.0,
            a=DEFAULT
        )
        for i, f, r in zip(intervened, y, reviewed)
    ])

    tn, fp, fn, tp = confusion_matrix(
        y,
        intervened,
        labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": precision_score(
            y, intervened, zero_division=0
        ),
        "recall": recall_score(
            y, intervened, zero_division=0
        ),
        "f1": f1_score(
            y, intervened, zero_division=0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "interventions": int(intervened.sum()),
        "reviews": int(reviewed.sum()),
        "review_rate": float(reviewed.mean()),
        "total_cost_eur": float(costs.sum()),
        "cost_per_1000": float(costs.mean() * 1000)
    }


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = build_features(train, engineered=True)

    X_validation = build_features(
        validation, engineered=True
    )

    y = validation[TARGET].to_numpy()

    ranges = get_training_ranges(X_train)
    ood = detect_ood(X_validation, ranges)

    candidates = joblib.load(
        TRAINING_DIR / "candidate_models.joblib"
    )

    baseline_cost = float(
        y.mean() * DEFAULT.failure_eur * 1000
    )

    results = []

    for name, model in candidates.items():
        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        brier = brier_score_loss(y, probabilities)

        ap = average_precision_score(y, probabilities)

        for threshold in THRESHOLDS:
            metrics = evaluate_threshold(
                y, probabilities, threshold, ood
            )

            savings = (
                baseline_cost -
                metrics["cost_per_1000"]
            )

            results.append({
                "candidate": name,
                "brier_score": brier,
                "average_precision": ap,
                **metrics,
                "savings_vs_no_action": savings,
                "savings_pct_vs_no_action": (
                    savings / baseline_cost * 100
                )
            })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "cost_per_1000"
    )

    results_df.to_csv(
        TRAINING_DIR / "threshold_comparison.csv",
        index=False
    )

    eligible = results_df[
        results_df["recall"] >= MIN_RECALL
    ]

    best_per_model = (
        eligible
        .drop_duplicates(subset="candidate")
        .sort_values("cost_per_1000")
    )

    columns = [
        "candidate",
        "threshold",
        "precision",
        "recall",
        "false_positives",
        "false_negatives",
        "reviews",
        "cost_per_1000",
        "savings_pct_vs_no_action"
    ]

    print("\nECONOMIC ASSUMPTIONS")
    print(DEFAULT)

    print(
        f"\nBreak-even probability: "
        f"{break_even_probability():.4%}"
    )

    print(f"OOD observations: {ood.sum()}")

    print(
        f"No-action cost per 1000: "
        f"EUR {baseline_cost:,.2f}"
    )

    print("\nBEST THRESHOLD PER CANDIDATE")

    print(
        best_per_model[columns]
        .round(4)
        .to_string(index=False)
    )

    if eligible.empty:
        print(
            "\nNo candidate meets the minimum recall."
        )
        return

    best = eligible.iloc[0]

    print("\nPROVISIONAL BEST MODEL")
    print(best.to_string())

    best_per_model.to_csv(
        TRAINING_DIR / "economic_comparison.csv",
        index=False
    )

    print("\nEconomic comparison saved successfully.")


if __name__ == "__main__":
    main()
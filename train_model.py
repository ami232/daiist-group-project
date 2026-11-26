"""Train an offer-count estimator and benchmark it on later notices."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NUMERIC = ["budget_eur", "duration_months"]
CATEGORIES = ["cpv", "buyer_type", "award_basis"]
FEATURES = NUMERIC + CATEGORIES


def candidates():
    for name, estimator in {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=90, max_leaf_nodes=9, l2_regularization=5,
            min_samples_leaf=30, random_state=17,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=90, max_depth=7, min_samples_leaf=20, random_state=17, n_jobs=1,
        ),
    }.items():
        preprocess = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), NUMERIC),
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORIES),
        ])
        yield name, make_pipeline(preprocess, estimator)


def train():
    rows = pd.read_csv(DATA / "ted_construction.csv", dtype={"cpv": str, "buyer_type": str, "notice_id": str, "award_id": str})
    rows = rows.loc[rows.procedure.eq("OPE") & rows.budget_eur.notna() & rows.duration_months.notna()].copy()
    rows["award_basis"] = rows.award_basis.fillna("unknown")
    rows["cpv"] = rows.cpv.str[:4]
    early = rows.loc[rows.year.le(2021) & rows.award_year.le(2021)]
    validation = rows.loc[rows.year.eq(2022) & rows.award_year.eq(2022)]
    test = rows.loc[rows.year.eq(2023)]
    if min(len(early), len(validation), len(test)) < 30:
        raise ValueError("The chronological cohorts need at least 30 usable cases each.")
    baseline = DummyRegressor(strategy="median").fit(early[FEATURES], early.offers)
    baseline_val = mean_absolute_error(validation.offers, baseline.predict(validation[FEATURES]))
    comparisons = {}
    fitted = {}
    for name, candidate in candidates():
        candidate.fit(early[FEATURES], early.offers)
        comparisons[name] = mean_absolute_error(validation.offers, candidate.predict(validation[FEATURES]))
        fitted[name] = candidate
    best = min(comparisons, key=comparisons.get)
    calibration = np.abs(validation.offers - np.maximum(1, np.rint(fitted[best].predict(validation[FEATURES]))))
    error_radius = int(np.ceil(np.quantile(calibration, 0.8)))
    development = rows.loc[rows.year.le(2022) & rows.award_year.le(2022)]
    model = dict(candidates())[best].fit(development[FEATURES], development.offers)
    baseline = DummyRegressor(strategy="median").fit(development[FEATURES], development.offers)
    test_predictions = np.maximum(1, np.rint(model.predict(test[FEATURES])))
    metrics = {
        "version": "ted-es-construction-2018-2023-v1",
        "estimator": best,
        "development_rows": len(development),
        "training_rows": len(early),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_year": 2023,
        "validation_mae": {k: round(v, 3) for k, v in comparisons.items()},
        "validation_median_baseline_mae": round(baseline_val, 3),
        "test_mae": round(mean_absolute_error(test.offers, test_predictions), 3),
        "test_median_baseline_mae": round(mean_absolute_error(test.offers, baseline.predict(test[FEATURES])), 3),
        "test_within_radius": round(float((np.abs(test.offers - test_predictions) <= error_radius).mean()), 3),
        "historical_median": float(baseline.constant_[0, 0]),
        "error_radius": error_radius,
        "cpv_support": development.cpv.value_counts().to_dict(),
        "budget_bounds": [float(development.budget_eur.min()), float(development.budget_eur.max())],
        "duration_bounds": [float(development.duration_months.min()), float(development.duration_months.max())],
    }
    joblib.dump(model, DATA / "model.joblib", compress=3)
    (DATA / "model_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()

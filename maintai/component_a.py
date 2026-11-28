"""Component A: failure-risk prediction (tabular ML).  OWNER: WP1.

STUB. Returns a rough risk from the physical indicators so the app runs end to
end. WP1 replaces `assess()` with the trained, calibrated scikit-learn model
loaded from models/ (joblib), keeping the same signature and return type.
"""

from pathlib import Path
from functools import lru_cache
import json

import joblib

from maintai import features
from maintai.schemas import (
    FailureMode,
    MachineInput,
    ModeProbability,
    RiskAssessment,
    RiskBand
)

from training.feature_engineering import (
    single_prediction_features
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODEL_VERSION = "maintai-gb-sigmoid-v1"
THRESHOLD = 0.19
CRITICAL_THRESHOLD = 0.8


@lru_cache(maxsize=1)
def _load_models():
    failure_model = joblib.load(
        MODELS_DIR / "failure_model.joblib"
    )

    failure_modes = joblib.load(
        MODELS_DIR / "failure_modes.joblib"
    )

    with open(MODELS_DIR / "model_card.json") as file:
        model_card = json.load(file)

    if model_card["threshold"] != THRESHOLD:
        raise ValueError("Model threshold mismatch")

    if model_card["model_version"] != MODEL_VERSION:
        raise ValueError("Model version mismatch")

    return failure_model, failure_modes, model_card


def _band(p: float) -> RiskBand:
    if p >= CRITICAL_THRESHOLD:
        return RiskBand.CRITICAL
    if p >= 0.3:
        return RiskBand.HIGH
    if p >= THRESHOLD:
        return RiskBand.MEDIUM
    return RiskBand.LOW


def _ood_flags(X, ranges):
    flags = []

    for column, limits in ranges.items():
        value = float(X[column].iloc[0])

        if value < limits["min"]:
            flags.append(f"{column}: below training minimum")

        elif value > limits["max"]:
            flags.append(f"{column}: above training maximum")

    return flags


def assess(machine: MachineInput) -> RiskAssessment:
    failure_model, failure_modes, card = _load_models()

    X = single_prediction_features(
        machine, engineered=True
    )

    p = float(failure_model.predict_proba(X)[0, 1])

    likely_modes = []

    for mode, model in failure_modes.items():
        probability = float(
            model.predict_proba(X)[0, 1]
        )

        likely_modes.append(
            ModeProbability(
                mode=FailureMode(mode),
                p=probability
            )
        )

    likely_modes.sort(
        key=lambda item: item.p,
        reverse=True
    )

    ood = _ood_flags(
        X, card["training_ranges"]
    )

    return RiskAssessment(
        p_fail=p,
        risk_band=_band(p),
        threshold=THRESHOLD,
        in_uncertainty_band=(
            THRESHOLD / 2 <= p < THRESHOLD * 2
        ),
        likely_modes=likely_modes,
        indicators=features.indicators(machine),
        ood_flags=ood,
        model_version=MODEL_VERSION
    )
"""Component A: failure-risk prediction (tabular ML).  OWNER: WP1.

STUB. Returns a rough risk from the physical indicators so the app runs end to
end. WP1 replaces `assess()` with the trained, calibrated scikit-learn model
loaded from models/ (joblib), keeping the same signature and return type.
"""

from maintai import features
from maintai.schemas import FailureMode, MachineInput, ModeProbability, RiskAssessment, RiskBand

MODEL_VERSION = "stub-a-v0"
THRESHOLD = 0.09375  # simplified break-even from REPORT.md; WP1 sets the real one on validation


def _band(p: float) -> RiskBand:
    if p >= 0.8:
        return RiskBand.CRITICAL
    if p >= 0.3:
        return RiskBand.HIGH
    if p >= THRESHOLD:
        return RiskBand.MEDIUM
    return RiskBand.LOW


def assess(machine: MachineInput) -> RiskAssessment:
    ind = features.indicators(machine)
    strain_ratio = ind.strain_min_nm / ind.strain_limit_min_nm
    wear_ratio = machine.tool_wear_min / 240
    p = min(0.95, max(0.01, 0.6 * max(strain_ratio, wear_ratio) ** 4))  # placeholder, not a model
    mode = FailureMode.OSF if strain_ratio >= wear_ratio else FailureMode.TWF
    return RiskAssessment(
        p_fail=round(p, 3),
        risk_band=_band(p),
        threshold=THRESHOLD,
        in_uncertainty_band=THRESHOLD / 2 <= p < THRESHOLD * 2,
        likely_modes=[ModeProbability(mode=mode, p=round(p, 3))],
        indicators=ind,
        ood_flags=[],
        model_version=MODEL_VERSION,
    )

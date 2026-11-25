"""Physical indicators and the features the drift check watches.

The formulas follow the AI4I 2020 documentation (UCI). They only use values that
are known at prediction time, so they are not data leakage.
"""

import math

from maintai.schemas import Indicators, MachineInput, MachineType, RiskAssessment

# Overstrain limit (tool wear x torque, in min·Nm) by product type, from the AI4I documentation.
STRAIN_LIMIT_MIN_NM = {MachineType.L: 11_000, MachineType.M: 12_000, MachineType.H: 13_000}
TYPE_CODE = {MachineType.L: 0.0, MachineType.M: 1.0, MachineType.H: 2.0}


def indicators(machine: MachineInput) -> Indicators:
    return Indicators(
        temp_diff_k=round(machine.process_temperature_k - machine.air_temperature_k, 3),
        power_w=round(machine.torque_nm * machine.rotational_speed_rpm * 2 * math.pi / 60, 1),
        strain_min_nm=round(machine.tool_wear_min * machine.torque_nm, 1),
        strain_limit_min_nm=STRAIN_LIMIT_MIN_NM[machine.type],
        tool_wear_min=machine.tool_wear_min,
    )


def drift_features(machine: MachineInput, risk: RiskAssessment) -> dict[str, float]:
    """Numbers logged in `features` and compared with monitoring/reference_features.csv.

    WP5 owns this list. Changing it means re-running `main.py reference`.
    """
    ind = risk.indicators
    return {
        "air_temp_k": float(machine.air_temperature_k),
        "process_temp_k": float(machine.process_temperature_k),
        "rpm": float(machine.rotational_speed_rpm),
        "torque_nm": float(machine.torque_nm),
        "tool_wear_min": float(machine.tool_wear_min),
        "temp_diff_k": float(ind.temp_diff_k),
        "power_w": float(ind.power_w),
        "type_code": TYPE_CODE[machine.type],
        "p_fail": float(risk.p_fail),
    }

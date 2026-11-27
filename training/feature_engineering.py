import pandas as pd
from maintai.features import indicators
from maintai.schemas import MachineInput

RAW_FEATURES = [
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min"
]

ENGINEERED_FEATURES = [
    "temp_diff_k",
    "power_w",
    "strain_min_nm",
    "strain_ratio"
]

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

COLUMN_MAPPING = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min"
}


def machine_to_features(machine: MachineInput) -> dict:
    ind = indicators(machine)

    return {
        "type": machine.type.value,
        "air_temperature_k": machine.air_temperature_k,
        "process_temperature_k": machine.process_temperature_k,
        "rotational_speed_rpm": machine.rotational_speed_rpm,
        "torque_nm": machine.torque_nm,
        "tool_wear_min": machine.tool_wear_min,
        "temp_diff_k": ind.temp_diff_k,
        "power_w": ind.power_w,
        "strain_min_nm": ind.strain_min_nm,
        "strain_ratio": (
            ind.strain_min_nm / ind.strain_limit_min_nm
        )
    }


def build_features(
    df: pd.DataFrame,
    engineered: bool = True
) -> pd.DataFrame:
    raw = df.rename(columns=COLUMN_MAPPING)[RAW_FEATURES]

    rows = []

    for record in raw.to_dict(orient="records"):
        machine = MachineInput(**record)
        rows.append(machine_to_features(machine))

    result = pd.DataFrame(rows, index=df.index)

    if engineered:
        return result[ALL_FEATURES]

    return result[RAW_FEATURES]


def single_prediction_features(
    machine: MachineInput,
    engineered: bool = True
) -> pd.DataFrame:
    result = pd.DataFrame([machine_to_features(machine)])

    if engineered:
        return result[ALL_FEATURES]

    return result[RAW_FEATURES]
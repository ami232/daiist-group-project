"""Sample readings for the form.

Hand-written to show each scenario. They are NOT plant data and NOT results:
the form only fills in the inputs, and nothing is shown until the user runs it.
Value ranges follow the AI4I 2020 documentation (UCI); the scenario names use the
dataset's failure rules (tool wear 200-240 min, heat dissipation when the
temperature difference is < 8.6 K and speed < 1380 rpm, power outside
3500-9000 W, overstrain when wear x torque exceeds the type limit).
"""

from maintai.inputs import EXAMPLE_INPUT

FIELDS = ("type", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min")

PRESETS: dict[str, dict] = {
    "Normal operation": {"type": "M", "air_temperature_k": 298.4, "process_temperature_k": 308.7,
                         "rotational_speed_rpm": 1520, "torque_nm": 38.0, "tool_wear_min": 40},
    "Worn tool": {"type": "L", "air_temperature_k": 300.1, "process_temperature_k": 310.2,
                  "rotational_speed_rpm": 1450, "torque_nm": 42.0, "tool_wear_min": 228},
    "Heat dissipation": {"type": "H", "air_temperature_k": 302.8, "process_temperature_k": 310.9,
                         "rotational_speed_rpm": 1310, "torque_nm": 48.0, "tool_wear_min": 90},
    "Power anomaly": {"type": "M", "air_temperature_k": 299.0, "process_temperature_k": 309.5,
                      "rotational_speed_rpm": 2850, "torque_nm": 4.5, "tool_wear_min": 60},
    "Overstrain": dict(EXAMPLE_INPUT),  # the reference example of maintai/inputs.py
    "Out of range": {"type": "L", "air_temperature_k": 365.0, "process_temperature_k": 310.2,
                     "rotational_speed_rpm": 1450, "torque_nm": 40.0, "tool_wear_min": 30},
}

DEFAULT = "Normal operation"


def values(name: str) -> list:
    """Form values in FIELDS order."""
    preset = PRESETS[name]
    return [preset[field] for field in FIELDS]

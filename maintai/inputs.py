"""Parse the raw input and refuse out-of-scope inputs before any model is called.

The raw input is JSON text, the same in the App tab, in eval/*.csv and on /run:

    {"type": "L", "air_temperature_k": 300.1, "process_temperature_k": 310.2,
     "rotational_speed_rpm": 1450, "torque_nm": 62.0, "tool_wear_min": 215}
"""

import json

from pydantic import ValidationError

from maintai.schemas import MachineInput, OutOfScope

EXAMPLE_INPUT = {
    "type": "L",
    "air_temperature_k": 300.1,
    "process_temperature_k": 310.2,
    "rotational_speed_rpm": 1450,
    "torque_nm": 62.0,
    "tool_wear_min": 215,
}


def parse(raw_input) -> MachineInput | OutOfScope:
    """MachineInput if the input is a valid machine reading, otherwise OutOfScope with the reason."""
    text = str(raw_input or "").strip()
    if not text:
        return OutOfScope(reason="empty_input")
    if len(text) > 2000:
        return OutOfScope(reason="input_too_long")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return OutOfScope(reason="not_a_machine_reading", detail="The input is not JSON.")
    if not isinstance(data, dict):
        return OutOfScope(reason="not_a_machine_reading", detail="The input must be a JSON object.")
    try:
        return MachineInput.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
        return OutOfScope(reason="invalid_reading", detail=problems[:500])


def summary(machine: MachineInput) -> str:
    """Short, personal-data-free summary for the logs."""
    return (
        f"type={machine.type.value} air={machine.air_temperature_k:.1f}K "
        f"process={machine.process_temperature_k:.1f}K rpm={machine.rotational_speed_rpm:.0f} "
        f"torque={machine.torque_nm:.1f}Nm wear={machine.tool_wear_min:.0f}min"
    )

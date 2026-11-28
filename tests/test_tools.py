"""
Tests for the WP2 simulated plant tools. `uv run python tests/test_tools.py`
(pytest-compatible too). No API calls, no network: the tools are local and deterministic.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maintai import economics, tools  # noqa: E402
from maintai.schemas import Action, CostEstimate, FailureMode, MachineType, RiskBand  # noqa: E402

DATA = ROOT / "maintai" / "data"


def raises(fn, *args):
    try:
        fn(*args)
    except ValueError as error:
        return str(error)
    raise AssertionError(f"{fn.__name__}{args} should raise ValueError")


def test_data_is_labelled_simulated_and_ids_unique():
    for path in DATA.glob("*.json"):
        assert "SIMULATED" in json.loads(path.read_text(encoding="utf-8"))["_label"], path.name
    ids = [p["part_id"] for p in json.loads((DATA / "spare_parts.json").read_text())["parts"]]
    assert len(ids) == len(set(ids))
    assert not any(c in (DATA / "costs.json").read_text() for c in ("4000", "300.0")), "euros live in economics.py only"


def test_parts_for_each_mode_and_type():
    for mode in (FailureMode.TWF, FailureMode.OSF, FailureMode.HDF, FailureMode.PWF):
        for mtype in MachineType:
            parts = tools.check_spare_parts(mode, mtype)
            assert parts, (mode, mtype)
            assert len({p.part_id for p in parts}) == len(parts)


def test_parts_zero_stock_is_returned_but_ranked_last():
    parts = tools.check_spare_parts(FailureMode.HDF, MachineType.M)
    assert [p.part_id for p in parts] == ["P-CL-010", "P-CL-011"]
    assert parts[-1].in_stock == 0 and parts[-1].lead_time_days > 0


def test_parts_respect_machine_type_compatibility():
    low = {p.part_id for p in tools.check_spare_parts(FailureMode.TWF, MachineType.L)}
    high = {p.part_id for p in tools.check_spare_parts(FailureMode.TWF, MachineType.H)}
    assert "P-TW-002" not in low and "P-TW-002" in high


def test_parts_without_a_part_return_empty_not_error():
    assert tools.check_spare_parts(FailureMode.RNF, MachineType.L) == []
    assert tools.check_spare_parts(FailureMode.NONE, MachineType.L) == []


def test_parts_accept_strings_and_reject_unknown_values():
    assert tools.check_spare_parts("HDF", "L") == tools.check_spare_parts(FailureMode.HDF, MachineType.L)
    assert "failure_mode" in raises(tools.check_spare_parts, "XYZ", "L")
    assert "machine_type" in raises(tools.check_spare_parts, "HDF", "Q")


def test_windows_by_urgency_and_duration():
    now = tools.find_maintenance_window("now", 1.0)
    assert [w.window_id for w in now] == ["W-2026-12-01-A"]
    assert [w.window_id for w in tools.find_maintenance_window("shift", 2.0)] == ["W-2026-12-02-A"]
    planned = tools.find_maintenance_window("planned", 1.0)
    assert [w.window_id for w in planned] == ["W-2026-12-01-A", "W-2026-12-02-A", "W-2026-12-04-B"]
    assert all(w.duration_h >= 1.0 for w in planned)


def test_window_that_does_not_exist_returns_empty():
    assert tools.find_maintenance_window("now", 4.0) == []  # the only 4 h window is on 4 Dec
    assert tools.find_maintenance_window("planned", 8.0) == []


def test_window_input_validation():
    assert "urgency" in raises(tools.find_maintenance_window, "tomorrow", 1.0)
    for bad in (0, -1, 9, float("nan"), "abc", None):
        assert "duration_h" in raises(tools.find_maintenance_window, "now", bad)


def test_costs_come_from_economics_and_vary_with_action():
    a = economics.DEFAULT
    rep = tools.estimate_costs("replace_tool", FailureMode.TWF, MachineType.L)
    assert isinstance(rep, CostEstimate)
    assert rep.preventive_eur == a.intervention_eur and rep.breakdown_eur == a.failure_eur
    mon = tools.estimate_costs("monitor", FailureMode.TWF, MachineType.L)
    assert mon.preventive_eur == 0.0 and mon.downtime_h == 0.0
    assert tools.estimate_costs("stop_and_inspect", FailureMode.OSF, MachineType.H).downtime_h > rep.downtime_h


def test_costs_cover_every_action_and_reject_unknown():
    for action in Action:
        tools.estimate_costs(action.value, FailureMode.HDF, MachineType.M)
    assert "action" in raises(tools.estimate_costs, "demolish", FailureMode.HDF, MachineType.M)
    assert "failure_mode" in raises(tools.estimate_costs, "monitor", "XYZ", MachineType.M)


def test_policy_rules():
    rules = tools.get_maintenance_policy(FailureMode.HDF, "critical")
    assert rules[0].startswith("Every P1 order")
    assert any("cooling system" in r for r in rules) and any("must review" in r for r in rules)
    assert len(rules) == len(set(rules))
    assert tools.get_maintenance_policy(FailureMode.NONE, RiskBand.LOW) == tools.get_maintenance_policy("none", "low")
    assert "risk_band" in raises(tools.get_maintenance_policy, FailureMode.HDF, "extreme")
    assert "failure_mode" in raises(tools.get_maintenance_policy, "XYZ", "low")


def test_deterministic_and_json_serialisable():
    first = [p.model_dump() for p in tools.check_spare_parts("HDF", "M")]
    assert first == [p.model_dump() for p in tools.check_spare_parts("HDF", "M")]
    json.dumps(first)
    json.dumps([w.model_dump() for w in tools.find_maintenance_window("planned", 1.0)])
    json.dumps(tools.estimate_costs("monitor", "none", "L").model_dump())


def test_tools_compatible_with_agent_and_guardrails():
    """The existing callers (agent.rule_based_plan, guardrails.finalise) still work."""
    import os
    os.environ["GEMINI_API_KEY"] = ""
    from maintai import agent, component_a, guardrails
    from maintai.schemas import MachineInput
    m = MachineInput(type="L", air_temperature_k=300, process_temperature_k=310, rotational_speed_rpm=1450,
                     torque_nm=62, tool_wear_min=215)
    risk = component_a.assess(m)
    result = agent.plan(m, risk, "checklist")
    final = guardrails.finalise(m, risk, result)
    assert final.cost is None or final.cost.breakdown_eur == economics.DEFAULT.failure_eur
    assert not final.validation_errors, final.validation_errors


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

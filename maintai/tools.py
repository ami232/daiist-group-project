"""Simulated plant tools the agent can call.  OWNER: WP2.

STUB data. WP2 moves the data to maintai/data/*.json and completes it. The
function names, arguments and return types are the contract with the agent (WP3):
keep them. All data is invented and must be declared as simulated in REPORT.md.
"""

from maintai import economics
from maintai.schemas import CostEstimate, FailureMode, MachineType, MaintenanceWindow, SparePart

# Fixed date of the simulated plant calendar, so tool outputs (and the LLM cache) do not change daily.
SIMULATED_TODAY = "2026-12-01"

_PARTS = {
    FailureMode.TWF: [SparePart(part_id="P-TW-001", name="Cutting tool insert", in_stock=12, lead_time_days=0, unit_cost_eur=45.0)],
    FailureMode.OSF: [SparePart(part_id="P-TW-001", name="Cutting tool insert", in_stock=12, lead_time_days=0, unit_cost_eur=45.0)],
    FailureMode.HDF: [SparePart(part_id="P-CL-010", name="Coolant pump seal kit", in_stock=2, lead_time_days=1, unit_cost_eur=120.0)],
    FailureMode.PWF: [SparePart(part_id="P-PW-020", name="Spindle drive fuse set", in_stock=5, lead_time_days=0, unit_cost_eur=60.0)],
}

_WINDOWS = [
    MaintenanceWindow(window_id="W-2026-12-01-A", start="2026-12-01T14:00", duration_h=1.0),
    MaintenanceWindow(window_id="W-2026-12-02-A", start="2026-12-02T06:00", duration_h=2.0),
    MaintenanceWindow(window_id="W-2026-12-04-B", start="2026-12-04T22:00", duration_h=4.0),
]


def check_spare_parts(failure_mode: FailureMode, machine_type: MachineType) -> list[SparePart]:
    return list(_PARTS.get(failure_mode, []))


def find_maintenance_window(urgency: str, duration_h: float) -> list[MaintenanceWindow]:
    """urgency: 'now' | 'shift' | 'planned'."""
    windows = [w for w in _WINDOWS if w.duration_h >= duration_h]
    return windows[:1] if urgency in ("now", "shift") else windows


def estimate_costs(action: str, failure_mode: FailureMode, machine_type: MachineType) -> CostEstimate:
    a = economics.DEFAULT
    return CostEstimate(preventive_eur=a.intervention_eur, breakdown_eur=a.failure_eur, downtime_h=2.0)


def get_maintenance_policy(failure_mode: FailureMode, risk_band: str) -> list[str]:
    rules = ["Every P1 order needs approval by the maintenance supervisor."]
    if failure_mode == FailureMode.HDF:
        rules.append("Heat dissipation risk: check the cooling system before restarting.")
    if risk_band == "critical":
        rules.append("Critical risk: a person must review the order before any action.")
    return rules

"""Simulated plant tools the agent can call.  OWNER: WP2.

All data is SIMULATED (maintai/data/*.json, each file carries a `_label`). It is
declared as simulated in REPORT.md. The function names, arguments and return
types are the contract with the agent (WP3): keep them.

Rules of these tools:
- Pure and local: no API calls, no clock, no randomness. The simulated "now" is fixed so
  outputs (and the LLM cache) do not change from day to day.
- Euro amounts come from economics.py, never from the LLM and never from the JSON files.
- Invalid input raises ValueError (the agent loop turns it into {"error": ...} for the model).
  An empty list means "nothing available", never "invalid input".
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from maintai import economics
from maintai.schemas import CostEstimate, FailureMode, MachineType, MaintenanceWindow, SparePart

_DATA_DIR = Path(__file__).parent / "data"

URGENCIES = ("now", "shift", "planned")
MIN_DURATION_H, MAX_DURATION_H = 0.5, 8.0
RISK_BANDS = ("low", "medium", "high", "critical")


def _load(name: str) -> dict:
    data = json.loads((_DATA_DIR / name).read_text(encoding="utf-8"))
    if "SIMULATED" not in data.get("_label", ""):
        raise ValueError(f"{name} must be labelled as simulated data")
    return data


_PARTS = _load("spare_parts.json")["parts"]
_CALENDAR = _load("calendar.json")
_COSTS = _load("costs.json")
_POLICIES = _load("policies.json")

# Fixed date of the simulated plant calendar, so tool outputs (and the LLM cache) do not change daily.
_REFERENCE = datetime.fromisoformat(_CALENDAR["reference_datetime"])
SIMULATED_TODAY = _REFERENCE.date().isoformat()

_WINDOWS = sorted((MaintenanceWindow(**w) for w in _CALENDAR["windows"]), key=lambda w: w.start)

# Data integrity: unique IDs, checked once at import.
assert len({p["part_id"] for p in _PARTS}) == len(_PARTS), "duplicate part_id"
assert len({w.window_id for w in _WINDOWS}) == len(_WINDOWS), "duplicate window_id"


def _enum(enum_cls, value, name: str):
    """Accept the enum or its string value; reject anything else."""
    try:
        return value if isinstance(value, enum_cls) else enum_cls(value)
    except ValueError:
        allowed = ", ".join(e.value for e in enum_cls)
        raise ValueError(f"unknown {name} {value!r}; allowed: {allowed}") from None


def check_spare_parts(failure_mode: FailureMode, machine_type: MachineType) -> list[SparePart]:
    """Parts that address a failure mode and fit the machine type, best option first.

    Order: in stock before out of stock, then shorter lead time, then cheaper.
    Stock 0 is returned as is (in_stock=0, lead_time_days>0): the agent must handle it.
    """
    mode = _enum(FailureMode, failure_mode, "failure_mode")
    mtype = _enum(MachineType, machine_type, "machine_type")
    found = [p for p in _PARTS if mode.value in p["failure_modes"] and mtype.value in p["compatible_types"]]
    found.sort(key=lambda p: (p["in_stock"] == 0, p["lead_time_days"], p["unit_cost_eur"], p["part_id"]))
    return [SparePart(part_id=p["part_id"], name=p["name"], in_stock=p["in_stock"],
                      lead_time_days=p["lead_time_days"], unit_cost_eur=p["unit_cost_eur"]) for p in found]


def find_maintenance_window(urgency: str, duration_h: float) -> list[MaintenanceWindow]:
    """Windows on the simulated calendar that are long enough and start inside the horizon.

    urgency: 'now' (next 8 h) | 'shift' (next 24 h) | 'planned' (next 7 days).
    'now' and 'shift' return only the earliest fitting window; 'planned' returns all of them.
    Returns [] when no window fits.
    """
    if urgency not in URGENCIES:
        raise ValueError(f"unknown urgency {urgency!r}; allowed: {', '.join(URGENCIES)}")
    try:
        duration = float(duration_h)
    except (TypeError, ValueError):
        raise ValueError(f"duration_h must be a number, got {duration_h!r}") from None
    if not MIN_DURATION_H <= duration <= MAX_DURATION_H:  # also rejects NaN
        raise ValueError(f"duration_h must be between {MIN_DURATION_H} and {MAX_DURATION_H} hours, got {duration_h!r}")
    limit = _REFERENCE + timedelta(hours=_CALENDAR["horizon_hours"][urgency])
    fits = [w for w in _WINDOWS if w.duration_h >= duration and _REFERENCE <= datetime.fromisoformat(w.start) <= limit]
    return fits[:1] if urgency in ("now", "shift") else fits


def estimate_costs(action: str, failure_mode: FailureMode, machine_type: MachineType) -> CostEstimate:
    """Cost of acting (preventive) against the cost of a breakdown, both from economics.py.

    The preventive cost is C_I for any intervention and 0 for 'monitor'. Spare-part prices are NOT
    added: C_I already covers the intervention (no double counting). Planned downtime comes from
    the simulated data.
    """
    _enum(FailureMode, failure_mode, "failure_mode")
    _enum(MachineType, machine_type, "machine_type")
    downtime = _COSTS["downtime_h"]
    if action not in downtime:
        raise ValueError(f"unknown action {action!r}; allowed: {', '.join(downtime)}")
    a = economics.DEFAULT
    return CostEstimate(preventive_eur=0.0 if action == "monitor" else a.intervention_eur,
                        breakdown_eur=a.failure_eur, downtime_h=downtime[action])


def get_maintenance_policy(failure_mode: FailureMode, risk_band: str) -> list[str]:
    """Simulated plant rules for a failure mode and a risk band (general rule first, no duplicates)."""
    mode = _enum(FailureMode, failure_mode, "failure_mode")
    if risk_band not in RISK_BANDS:
        raise ValueError(f"unknown risk_band {risk_band!r}; allowed: {', '.join(RISK_BANDS)}")
    rules = list(_POLICIES["general"]) + _POLICIES["by_failure_mode"][mode.value] + _POLICIES["by_risk_band"][risk_band]
    return list(dict.fromkeys(rules))

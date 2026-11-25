"""Contracts between MaintAI modules.

Every module talks to the others through these models, so the team can work in
parallel. Changing a field here affects everyone: do it in its own small PR and
tell the team.

Flow of one request:

    raw text --inputs.parse--> MachineInput
    MachineInput --component_a.assess--> RiskAssessment
    (MachineInput, RiskAssessment, arm) --agent.plan--> AgentResult (contains WorkOrderDraft)
    AgentResult --guardrails.finalise--> FinalWorkOrder
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Shared vocabularies -------------------------------------------------------


class MachineType(str, Enum):
    """Product quality variant in AI4I 2020: L (low), M (medium), H (high)."""

    L = "L"
    M = "M"
    H = "H"


class FailureMode(str, Enum):
    """AI4I 2020 failure modes. RNF is random by design and cannot be predicted."""

    TWF = "TWF"  # tool wear failure
    HDF = "HDF"  # heat dissipation failure
    PWF = "PWF"  # power failure
    OSF = "OSF"  # overstrain failure
    RNF = "RNF"  # random failure
    NONE = "none"


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Priority(str, Enum):
    P1 = "P1"  # act now
    P2 = "P2"  # this shift
    P3 = "P3"  # next planned window
    P4 = "P4"  # monitor only


class Action(str, Enum):
    MONITOR = "monitor"
    SCHEDULE_INSPECTION = "schedule_inspection"
    REPLACE_TOOL = "replace_tool"
    COOLING_CHECK = "cooling_check"
    POWER_CHECK = "power_check"
    REDUCE_LOAD = "reduce_load"
    STOP_AND_INSPECT = "stop_and_inspect"


Arm = Literal["checklist", "cost", "root_cause"]
"""Bandit arms: the writing strategy of the work order. Never changes safety rules."""

ARMS: tuple[str, ...] = ("checklist", "cost", "root_cause")


# --- Input ---------------------------------------------------------------------


class MachineInput(BaseModel):
    """One machine reading, as typed in the App tab or written in eval/*.csv (as JSON).

    The limits are *physical plausibility* checks: values outside them are refused
    as out of scope. Values that are plausible but outside the training data are
    accepted and flagged later by component A (ood_flags).
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    type: MachineType
    air_temperature_k: float = Field(ge=250, le=350)
    process_temperature_k: float = Field(ge=250, le=400)
    rotational_speed_rpm: float = Field(gt=0, le=5000)
    torque_nm: float = Field(ge=0, le=200)
    tool_wear_min: float = Field(ge=0, le=500)


class OutOfScope(BaseModel):
    """Returned by inputs.parse when the input cannot be handled. No model is called."""

    reason: str
    detail: str = ""


# --- Component A -> Component B --------------------------------------------------


class ModeProbability(BaseModel):
    mode: FailureMode
    p: float = Field(ge=0, le=1)


class Indicators(BaseModel):
    """Physical indicators computed from the input (features.py)."""

    temp_diff_k: float
    power_w: float
    strain_min_nm: float
    strain_limit_min_nm: float
    tool_wear_min: float


class RiskAssessment(BaseModel):
    """What component A gives to component B."""

    p_fail: float = Field(ge=0, le=1, description="calibrated probability that this process ends in a failure")
    risk_band: RiskBand
    threshold: float = Field(ge=0, le=1, description="decision threshold in use")
    in_uncertainty_band: bool = False
    likely_modes: list[ModeProbability] = []
    indicators: Indicators
    ood_flags: list[str] = []
    model_version: str


# --- Agent tools ---------------------------------------------------------------


class SparePart(BaseModel):
    part_id: str
    name: str
    in_stock: int
    lead_time_days: int
    unit_cost_eur: float


class MaintenanceWindow(BaseModel):
    window_id: str
    start: str  # ISO date-time on the simulated plant calendar
    duration_h: float


class CostEstimate(BaseModel):
    preventive_eur: float
    breakdown_eur: float
    downtime_h: float


class ToolCall(BaseModel):
    """One tool call made by the agent, logged for the trace and the evaluation."""

    name: str
    arguments: dict
    result: dict | list


# --- Component B ---------------------------------------------------------------


class WorkOrderDraft(BaseModel):
    """What the LLM is allowed to write. It never writes euro amounts (decision D2)."""

    priority: Priority
    action: Action
    failure_mode_addressed: FailureMode
    steps: list[str] = Field(min_length=1, max_length=8)
    justification: str
    part_ids: list[str] = []
    window_id: str | None = None
    requires_human_review: bool
    review_reason: str = ""


class LLMUsage(BaseModel):
    """Token and call accounting for one request (all zero for stubs and cache hits)."""

    model: str = ""
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_thinking: int = 0
    cache_hit: bool = False
    retries: int = 0
    rate_limited: bool = False

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out + self.tokens_thinking


class AgentResult(BaseModel):
    draft: WorkOrderDraft
    tool_calls: list[ToolCall] = []
    usage: LLMUsage = LLMUsage()
    prompt_version: str
    degraded: bool = False
    degraded_reason: str = ""


class FinalWorkOrder(BaseModel):
    """The work order shown to the user and logged. Costs come from tools, not from the LLM."""

    draft: WorkOrderDraft
    cost: CostEstimate | None = None
    review: bool
    review_reasons: list[str] = []
    validation_errors: list[str] = []

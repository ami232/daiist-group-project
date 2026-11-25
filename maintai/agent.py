"""Component B: the work-order agent (agentic LLM).  OWNER: WP3.

STUB. Builds a work order with fixed rules and the tools, without calling Gemini.
This is also the shape of the degraded mode. WP3 replaces `plan()` with the real
agent loop (decision D3):

    call 1: system prompt + strategy (arm) + MachineInput + RiskAssessment, tools declared
    run the requested tools locally (parallel calls allowed)
    call 2: tool results -> WorkOrderDraft as JSON with a schema
    call 3: only if validation fails, one repair attempt
    low risk: a single call without tools (follow-up note)

At most MAX_LLM_CALLS calls per request.
"""

from maintai import tools
from maintai.schemas import (Action, AgentResult, FailureMode, MachineInput, Priority, RiskAssessment,
                             RiskBand, ToolCall, WorkOrderDraft)

MAX_LLM_CALLS = 3
PROMPT_VERSION = "stub-agent-v0"

_ACTION_FOR_MODE = {
    FailureMode.TWF: Action.REPLACE_TOOL,
    FailureMode.OSF: Action.REDUCE_LOAD,
    FailureMode.HDF: Action.COOLING_CHECK,
    FailureMode.PWF: Action.POWER_CHECK,
}
_PRIORITY_FOR_BAND = {RiskBand.LOW: Priority.P4, RiskBand.MEDIUM: Priority.P3,
                      RiskBand.HIGH: Priority.P2, RiskBand.CRITICAL: Priority.P1}


def rule_based_plan(machine: MachineInput, risk: RiskAssessment, reason: str) -> AgentResult:
    """Work order from fixed rules. Used by the stub and by the degraded mode."""
    mode = risk.likely_modes[0].mode if risk.likely_modes else FailureMode.NONE
    if risk.risk_band == RiskBand.LOW:
        draft = WorkOrderDraft(priority=Priority.P4, action=Action.MONITOR, failure_mode_addressed=FailureMode.NONE,
                               steps=["Keep normal operation and check again at the next reading."],
                               justification="Low failure risk.", requires_human_review=False)
        return AgentResult(draft=draft, prompt_version=PROMPT_VERSION, degraded=True, degraded_reason=reason)

    parts = tools.check_spare_parts(mode, machine.type)
    urgency = "now" if risk.risk_band == RiskBand.CRITICAL else "shift" if risk.risk_band == RiskBand.HIGH else "planned"
    windows = tools.find_maintenance_window(urgency, 1.0)
    calls = [
        ToolCall(name="check_spare_parts", arguments={"failure_mode": mode.value, "machine_type": machine.type.value},
                 result=[p.model_dump() for p in parts]),
        ToolCall(name="find_maintenance_window", arguments={"urgency": urgency, "duration_h": 1.0},
                 result=[w.model_dump() for w in windows]),
    ]
    draft = WorkOrderDraft(
        priority=_PRIORITY_FOR_BAND[risk.risk_band],
        action=_ACTION_FOR_MODE.get(mode, Action.SCHEDULE_INSPECTION),
        failure_mode_addressed=mode,
        steps=["Inspect the machine for the predicted failure mode.", "Record the findings in the work order."],
        justification=f"Failure risk {risk.p_fail:.0%} ({risk.risk_band.value}), most likely mode {mode.value}.",
        part_ids=[p.part_id for p in parts[:1]],
        window_id=windows[0].window_id if windows else None,
        requires_human_review=risk.risk_band == RiskBand.CRITICAL,
        review_reason="Critical risk." if risk.risk_band == RiskBand.CRITICAL else "",
    )
    return AgentResult(draft=draft, tool_calls=calls, prompt_version=PROMPT_VERSION, degraded=True, degraded_reason=reason)


def plan(machine: MachineInput, risk: RiskAssessment, arm: str) -> AgentResult:
    return rule_based_plan(machine, risk, reason="stub: agent not implemented yet")

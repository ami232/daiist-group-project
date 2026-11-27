"""Validation, safety rules and the review decision.  OWNER: WP3.

The safety rules here are deterministic and sit outside the LLM and the bandit
(go/no-go criterion 3). `finalise()` never receives the writing style (arm): a test
checks its signature. The final `review` is the OR of what the agent proposes and
what these rules require.

Checks on a work order:
1. Arm-free decision: if the agent logged the decision it took WITHOUT the arm
   (agent.AgentRun.decision), a draft whose decision fields differ from it is rejected:
   the arm-free fields are restored and the order is sent to review (failed_validation).
2. Consistency with the risk band: low risk -> monitor (P4/P3); otherwise no "monitor", a
   priority allowed for the band, an action that fits the failure mode, and a failure mode
   among the two most likely ones.
3. Grounding: part ids only from check_spare_parts results, window ids only from
   find_maintenance_window results; the text may not mention other part or window ids.
4. No money in the text: euro amounts come only from tools.estimate_costs (decision D2).
5. Mandatory review: critical risk, reading outside the training range, prediction in the
   uncertainty band, failed validation, a chosen part with zero stock, or the agent asked.
"""

import re

from maintai import tools
from maintai.schemas import (Action, AgentResult, FailureMode, FinalWorkOrder, MachineInput, Priority,
                             RiskAssessment, RiskBand, ToolCall)

DECISION_FIELDS = ("priority", "action", "failure_mode_addressed", "part_ids", "window_id",
                   "requires_human_review", "review_reason")

# Safe actions per failure mode (schedule_inspection is always acceptable for an intervention).
ACTION_FOR_MODE = {
    FailureMode.TWF: {Action.REPLACE_TOOL, Action.STOP_AND_INSPECT},
    FailureMode.OSF: {Action.REDUCE_LOAD, Action.REPLACE_TOOL, Action.STOP_AND_INSPECT},
    FailureMode.HDF: {Action.COOLING_CHECK, Action.STOP_AND_INSPECT},
    FailureMode.PWF: {Action.POWER_CHECK, Action.REDUCE_LOAD, Action.STOP_AND_INSPECT},
    FailureMode.RNF: {Action.STOP_AND_INSPECT},
    FailureMode.NONE: set(),
}
PRIORITY_FOR_BAND = {
    RiskBand.LOW: {Priority.P4, Priority.P3},
    RiskBand.MEDIUM: {Priority.P3, Priority.P2},
    RiskBand.HIGH: {Priority.P2, Priority.P1},
    RiskBand.CRITICAL: {Priority.P1},
}

_MONEY = re.compile(r"(€|\$|£)\s*\d|\d[\d.,\s]*(€|\$|£|\beur\b|\beuros?\b|\busd\b|\bdollars?\b)"
                    r"|\b(eur|euros?|usd)\s*\d", re.IGNORECASE)
_PART_ID = re.compile(r"\bP-[A-Z]{2}-\d{3}\b")
_WINDOW_ID = re.compile(r"\bW-\d{4}-\d{2}-\d{2}-[A-Z]\b")


def _value(x):
    return getattr(x, "value", x)


def grounded_ids(tool_calls: list[ToolCall]) -> tuple[dict[str, int], set[str]]:
    """({part_id: in_stock}, {window_id}) that appeared in the results of the right tools."""
    parts, windows = {}, set()
    for call in tool_calls:
        items = call.result if isinstance(call.result, list) else [call.result]
        for item in items:
            if not isinstance(item, dict):
                continue
            if call.name == "check_spare_parts" and item.get("part_id"):
                parts[str(item["part_id"])] = int(item.get("in_stock", 0) or 0)
            if call.name == "find_maintenance_window" and item.get("window_id"):
                windows.add(str(item["window_id"]))
    return parts, windows


def decision_errors(risk: RiskAssessment, decision, tool_calls: list[ToolCall]) -> list[str]:
    """What is wrong with a decision (a WorkOrderDraft or agent.Decision), as a list. Empty if valid."""
    errors = []
    band = risk.risk_band
    priority, action = Priority(_value(decision.priority)), Action(_value(decision.action))
    mode = FailureMode(_value(decision.failure_mode_addressed))
    if priority not in PRIORITY_FOR_BAND[band]:
        errors.append(f"priority {priority.value} not allowed for {band.value} risk")
    if band == RiskBand.LOW:
        if action != Action.MONITOR:
            errors.append(f"action {action.value} at low risk (expected monitor)")
    else:
        if action == Action.MONITOR:
            errors.append(f"action monitor not allowed at {band.value} risk")
        elif action != Action.SCHEDULE_INSPECTION and action not in ACTION_FOR_MODE.get(mode, set()):
            errors.append(f"action {action.value} does not fit failure mode {mode.value}")
        top = [m.mode for m in sorted(risk.likely_modes, key=lambda m: m.p, reverse=True)[:2]]
        if top and mode not in top:
            errors.append(f"failure mode {mode.value} is not among the most likely ({', '.join(m.value for m in top)})")
    parts, windows = grounded_ids(tool_calls)
    errors += [f"unknown part_id {p}" for p in decision.part_ids if p not in parts]
    if decision.window_id and decision.window_id not in windows:
        errors.append(f"unknown window_id {decision.window_id}")
    return errors


def text_errors(draft, tool_calls: list[ToolCall]) -> list[str]:
    """Problems in the free text: money amounts, or part/window ids that no tool returned."""
    text = " ".join(list(draft.steps) + [draft.justification])
    parts, windows = grounded_ids(tool_calls)
    errors = []
    if _MONEY.search(text):
        errors.append("money amount in the text (costs come only from the cost tool)")
    errors += [f"text mentions unknown part_id {p}" for p in sorted(set(_PART_ID.findall(text)) - set(parts))]
    errors += [f"text mentions unknown window_id {w}" for w in sorted(set(_WINDOW_ID.findall(text)) - windows)]
    return errors


def _decision_mismatch(result: AgentResult) -> tuple[list[str], dict]:
    """Fields where the draft differs from the arm-free decision, and the arm-free values."""
    decision = getattr(result, "decision", None)
    if decision is None:
        return [], {}
    reference = {f: _value(getattr(decision, f)) for f in DECISION_FIELDS}
    differ = [f for f in DECISION_FIELDS if _value(getattr(result.draft, f)) != reference[f]]
    return differ, reference


def finalise(machine: MachineInput, risk: RiskAssessment, result: AgentResult) -> FinalWorkOrder:
    draft = result.draft
    errors = []
    differ, reference = _decision_mismatch(result)
    if differ:  # reject the draft's decision: restore the arm-free one and send the order to a person
        errors.append(f"decision differs from the arm-free decision: {', '.join(differ)}")
        draft = draft.model_copy(update={f: reference[f] for f in differ})
        draft = type(draft).model_validate(draft.model_dump())
    errors += decision_errors(risk, draft, result.tool_calls)
    errors += text_errors(draft, result.tool_calls)

    reasons = []
    if risk.risk_band == RiskBand.CRITICAL:
        reasons.append("critical_risk")
    if risk.ood_flags:
        reasons.append("out_of_training_range")
    if risk.in_uncertainty_band:
        reasons.append("uncertain_prediction")
    if errors:
        reasons.append("failed_validation")
    stock, _ = grounded_ids(result.tool_calls)
    if any(stock.get(p, 1) == 0 for p in draft.part_ids):
        reasons.append("part_out_of_stock")
    if draft.requires_human_review:
        reasons.append("agent_requested")

    cost = None  # euros only from the cost tool, never from the LLM
    if draft.failure_mode_addressed != FailureMode.NONE:
        cost = tools.estimate_costs(draft.action.value, draft.failure_mode_addressed, machine.type)
    return FinalWorkOrder(draft=draft, cost=cost, review=bool(reasons), review_reasons=reasons,
                          validation_errors=errors)

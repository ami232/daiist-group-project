"""Validation, safety rules and the review decision.  OWNER: WP3.

The safety rules here are deterministic and sit outside the LLM and the bandit
(go/no-go criterion 3). The final `review` is the OR of what the agent proposes
and what these rules require.

Partly STUB: the grounding checks are in place; WP3 adds action/risk consistency
checks and the costs from tools.estimate_costs.
"""

from maintai import tools
from maintai.schemas import AgentResult, FinalWorkOrder, MachineInput, RiskAssessment, RiskBand


def _grounding_errors(result: AgentResult) -> list[str]:
    """Every part and window id must come from a tool result (no invented ids)."""
    seen_parts, seen_windows = set(), set()
    for call in result.tool_calls:
        items = call.result if isinstance(call.result, list) else [call.result]
        for item in items:
            if isinstance(item, dict):
                seen_parts.add(item.get("part_id"))
                seen_windows.add(item.get("window_id"))
    errors = [f"unknown part_id {p}" for p in result.draft.part_ids if p not in seen_parts]
    if result.draft.window_id and result.draft.window_id not in seen_windows:
        errors.append(f"unknown window_id {result.draft.window_id}")
    return errors


def finalise(machine: MachineInput, risk: RiskAssessment, result: AgentResult) -> FinalWorkOrder:
    reasons = []
    if risk.risk_band == RiskBand.CRITICAL:
        reasons.append("critical_risk")
    if risk.ood_flags:
        reasons.append("out_of_training_range")
    if risk.in_uncertainty_band:
        reasons.append("uncertain_prediction")
    errors = _grounding_errors(result)
    if errors:
        reasons.append("failed_validation")
    if result.draft.requires_human_review:
        reasons.append("agent_requested")

    cost = None
    if result.draft.failure_mode_addressed.value != "none":
        cost = tools.estimate_costs(result.draft.action.value, result.draft.failure_mode_addressed, machine.type)
    return FinalWorkOrder(draft=result.draft, cost=cost, review=bool(reasons), review_reasons=reasons,
                          validation_errors=errors)

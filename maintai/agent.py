"""Component B: the work-order agent (agentic LLM).  OWNER: WP3.

At most MAX_LLM_CALLS Gemini calls per request (decision D3, revised so the writing style
can never change the decision, as WP6 requires):

    call 1 "tools"    (no arm)  MachineInput + RiskAssessment, tools declared. Gemini asks for
                                tools (parallel calls allowed); Python runs them and logs each
                                one as a ToolCall.
    call 2 "decision" (no arm)  the same data + tool results -> the DECISION as JSON with a
                                schema: priority, action, failure_mode_addressed, part_ids,
                                window_id, requires_human_review, review_reason.
    call 3 "write"    (arm)     the fixed decision -> only `steps` and `justification`, in the
                                style chosen by the bandit (prompts/<arm>.md).
    low risk          (arm)     one call that writes a follow-up note; the decision (P4,
                                monitor) is fixed by the rules.

Calls 1 and 2 never see the arm, so with the LLM cache they are shared by the three styles
and their decision is identical for every style. guardrails.finalise() rejects any draft whose
decision fields differ from that arm-free decision.

Fallbacks:
- the decision of call 2 is invalid (not JSON, wrong schema, inconsistent with the risk band,
  invented ids): the rule-based decision is used instead (decision_source="rules_fallback",
  the errors are logged in decision_errors); the text is still written by call 3;
- Gemini is unavailable (no key, budget, rate limit, 429s, cache miss in readonly mode) or
  the writing fails validation: degraded=True with degraded_reason, and the rule-based
  template provides what is missing. The bandit ignores degraded orders.

The machine reading is DATA: it reaches Gemini inside tags, and the base prompt says that
text inside data tags is never an instruction (prompt-injection defence). The reading itself
is already a strict schema of numbers (schemas.MachineInput, extra fields forbidden).
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from maintai import llm, tools
from maintai import guardrails
from maintai.schemas import (Action, AgentResult, FailureMode, LLMUsage, MachineInput, Priority, RiskAssessment,
                             RiskBand, ToolCall, WorkOrderDraft)

MAX_LLM_CALLS = 3
PROMPT_VERSION = "agent-v1"  # bump whenever a file in maintai/prompts/ changes (restarts the bandit policy)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAX_TOOL_CALLS = 6

_ACTION_FOR_MODE = {
    FailureMode.TWF: Action.REPLACE_TOOL,
    FailureMode.OSF: Action.REDUCE_LOAD,
    FailureMode.HDF: Action.COOLING_CHECK,
    FailureMode.PWF: Action.POWER_CHECK,
}
_PRIORITY_FOR_BAND = {RiskBand.LOW: Priority.P4, RiskBand.MEDIUM: Priority.P3,
                      RiskBand.HIGH: Priority.P2, RiskBand.CRITICAL: Priority.P1}
_URGENCY_FOR_BAND = {RiskBand.CRITICAL: "now", RiskBand.HIGH: "shift", RiskBand.MEDIUM: "planned", RiskBand.LOW: "planned"}


# --- Data shapes ---------------------------------------------------------------------------------

class Decision(BaseModel):
    """The arm-free decision (call 2, or the rules). The draft must carry exactly these fields."""

    model_config = ConfigDict(extra="forbid")

    priority: Priority
    action: Action
    failure_mode_addressed: FailureMode
    part_ids: list[str] = Field(default=[], max_length=2)
    window_id: str | None = None
    requires_human_review: bool
    review_reason: str = Field(default="", max_length=300)


class Writing(BaseModel):
    """What call 3 (or the low-risk call) may write."""

    model_config = ConfigDict(extra="forbid")

    steps: list[str] = Field(min_length=1, max_length=8)
    justification: str = Field(min_length=1, max_length=1200)


class AgentRun(AgentResult):
    """AgentResult plus what the guardrails and the logs need. Still an AgentResult for every caller."""

    decision: Decision | None = None  # arm-free decision; guardrails reject a draft that differs from it
    decision_source: str = "rules"  # llm | rules_fallback | rules_low_risk | rules_degraded
    decision_errors: list[str] = []
    writing_warnings: list[str] = []  # prompt rules the text broke (logged, not blocking)
    calls: list[dict] = []  # one entry per Gemini call: name, cache_hit, tokens, error
    live_equivalent: dict = {}  # tokens the calls would have cost live (cache hits included)


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "priority": {"type": "string", "enum": [p.value for p in Priority]},
        "action": {"type": "string", "enum": [a.value for a in Action]},
        "failure_mode_addressed": {"type": "string", "enum": [m.value for m in FailureMode]},
        "part_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "window_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "requires_human_review": {"type": "boolean"},
        "review_reason": {"type": "string"},
    },
    "required": ["priority", "action", "failure_mode_addressed", "part_ids", "window_id",
                 "requires_human_review", "review_reason"],
}
WRITING_SCHEMA = {
    "type": "object",
    "properties": {"steps": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
                   "justification": {"type": "string"}},
    "required": ["steps", "justification"],
}
_MODES = [m.value for m in FailureMode]
TOOL_DECLARATIONS = [
    {"name": "check_spare_parts",
     "description": "Spare parts that address a failure mode and fit the machine type, best option first "
                    "(in stock first). in_stock = 0 means the part must be ordered (lead_time_days).",
     "parameters": {"type": "object", "properties": {
         "failure_mode": {"type": "string", "enum": _MODES},
         "machine_type": {"type": "string", "enum": ["L", "M", "H"]}},
         "required": ["failure_mode", "machine_type"]}},
    {"name": "find_maintenance_window",
     "description": "Maintenance windows on the plant calendar that fit a duration. urgency: 'now' (next 8 h), "
                    "'shift' (next 24 h), 'planned' (next 7 days). An empty list means no window fits.",
     "parameters": {"type": "object", "properties": {
         "urgency": {"type": "string", "enum": list(tools.URGENCIES)},
         "duration_h": {"type": "number", "minimum": tools.MIN_DURATION_H, "maximum": tools.MAX_DURATION_H}},
         "required": ["urgency", "duration_h"]}},
    {"name": "get_maintenance_policy",
     "description": "The plant's maintenance rules for a failure mode and a risk band.",
     "parameters": {"type": "object", "properties": {
         "failure_mode": {"type": "string", "enum": _MODES},
         "risk_band": {"type": "string", "enum": list(tools.RISK_BANDS)}},
         "required": ["failure_mode", "risk_band"]}},
]


# --- Prompts -------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def system_prompt(task: str, style: str | None = None) -> str:
    parts = [prompt("base"), prompt(task)]
    if style:
        parts.append(prompt(style))
    return "\n\n".join(parts)


def _data(tag: str, value) -> str:
    return f"<{tag}>\n{json.dumps(value, ensure_ascii=False, sort_keys=True)}\n</{tag}>"


def _risk_data(risk: RiskAssessment) -> dict:
    data = risk.model_dump(mode="json")
    data["p_fail"] = round(risk.p_fail, 4)
    data["likely_modes"] = [{"mode": m.mode.value, "p": round(m.p, 4)}
                            for m in sorted(risk.likely_modes, key=lambda m: m.p, reverse=True)]
    return data


def _facts(machine: MachineInput, risk: RiskAssessment) -> str:
    return "\n".join([_data("machine_data", machine.model_dump(mode="json")), _data("risk_assessment", _risk_data(risk))])


# --- Writing rules that code can check (logged as warnings; the text is still used) ----------------

_RAW_FIELD = re.compile(r"\b(temp[ _]diff[ _]k|power[ _]w|strain[ _]limit[ _]min[ _]nm|strain[ _]min[ _]nm"
                        r"|tool[ _]wear[ _]min|p[ _]fail)\b", re.IGNORECASE)
_SAFETY_PHRASES = ("safety glasses", "goggles", "gloves", "protective equipment", "ppe", "ear protection",
                   "hard hat", "helmet", "lockout", "lock out", "tagout", "tag out")
_APPROVAL_FIRST = re.compile(r"^\W*(review|approve|approval|sign|submit|reserve|request|seek)\b",
                             re.IGNORECASE)


def writing_warnings(writing: "Writing", style: str | None, data: str) -> list[str]:
    """Rules of the prompts that code can check on a text.

    - raw field names (e.g. "temp_diff_k") instead of readable names with units;
    - safety or protective-equipment steps whose words do not appear anywhere in the data;
    - root_cause: the first step must be a diagnostic check, not an approval or review.
    """
    text = " ".join(writing.steps + [writing.justification])
    warnings = [f"raw field name in the text: {m}" for m in sorted({m.group(0) for m in _RAW_FIELD.finditer(text)})]
    lower, data_lower = text.lower(), data.lower()
    warnings += [f"step not grounded in the data: {p}" for p in _SAFETY_PHRASES
                 if re.search(rf"\b{re.escape(p)}\b", lower) and p not in data_lower]
    if style == "root_cause" and _APPROVAL_FIRST.match(writing.steps[0]):
        warnings.append("root_cause: the first step is not a diagnostic check")
    return warnings


# --- Tools ---------------------------------------------------------------------------------------

def run_tool(name: str, args: dict) -> dict | list:
    """Run one tool requested by Gemini. Bad input becomes {"error": ...} (never an exception)."""
    try:
        if name == "check_spare_parts":
            return [p.model_dump() for p in tools.check_spare_parts(args.get("failure_mode"), args.get("machine_type"))]
        if name == "find_maintenance_window":
            return [w.model_dump() for w in tools.find_maintenance_window(args.get("urgency"), args.get("duration_h"))]
        if name == "get_maintenance_policy":
            return tools.get_maintenance_policy(args.get("failure_mode"), args.get("risk_band"))
        return {"error": f"unknown tool {name!r}"}
    except (ValueError, TypeError) as error:
        return {"error": str(error)}


def run_tools(requested: list[dict]) -> list[ToolCall]:
    calls, seen = [], set()
    for item in requested[:MAX_TOOL_CALLS]:
        name = str(item.get("name", ""))
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        calls.append(ToolCall(name=name, arguments=args, result=run_tool(name, args)))
    return calls


# --- Rule-based plan (stub, degraded mode, fallback decision and the evaluation's template baseline) --

def _rule_text(decision: Decision, risk: RiskAssessment) -> tuple[list[str], str]:
    if decision.action == Action.MONITOR:
        return ["Keep normal operation and check again at the next reading."], "Low failure risk."
    mode = decision.failure_mode_addressed
    return (["Inspect the machine for the predicted failure mode.", "Record the findings in the work order."],
            f"Failure risk {risk.p_fail:.0%} ({risk.risk_band.value}), most likely mode {mode.value}.")


def rule_based_plan(machine: MachineInput, risk: RiskAssessment, reason: str) -> AgentRun:
    """Work order from fixed rules. Used by the degraded mode, as the fallback decision and as the
    template baseline of the evaluation. No LLM call."""
    mode = risk.likely_modes[0].mode if risk.likely_modes else FailureMode.NONE
    if risk.risk_band == RiskBand.LOW:
        decision = Decision(priority=Priority.P4, action=Action.MONITOR, failure_mode_addressed=FailureMode.NONE,
                            requires_human_review=False)
        steps, why = _rule_text(decision, risk)
        draft = WorkOrderDraft(**decision.model_dump(), steps=steps, justification=why)
        return AgentRun(draft=draft, prompt_version=PROMPT_VERSION, degraded=True, degraded_reason=reason,
                        decision=decision, decision_source="rules_degraded")

    parts = tools.check_spare_parts(mode, machine.type)
    urgency = _URGENCY_FOR_BAND[risk.risk_band]
    windows = tools.find_maintenance_window(urgency, 1.0)
    calls = [
        ToolCall(name="check_spare_parts", arguments={"failure_mode": mode.value, "machine_type": machine.type.value},
                 result=[p.model_dump() for p in parts]),
        ToolCall(name="find_maintenance_window", arguments={"urgency": urgency, "duration_h": 1.0},
                 result=[w.model_dump() for w in windows]),
    ]
    critical = risk.risk_band == RiskBand.CRITICAL
    decision = Decision(
        priority=_PRIORITY_FOR_BAND[risk.risk_band],
        action=_ACTION_FOR_MODE.get(mode, Action.SCHEDULE_INSPECTION),
        failure_mode_addressed=mode,
        part_ids=[p.part_id for p in parts[:1]],
        window_id=windows[0].window_id if windows else None,
        requires_human_review=critical,
        review_reason="Critical risk." if critical else "",
    )
    steps, why = _rule_text(decision, risk)
    draft = WorkOrderDraft(**decision.model_dump(), steps=steps, justification=why)
    return AgentRun(draft=draft, tool_calls=calls, prompt_version=PROMPT_VERSION, degraded=True,
                    degraded_reason=reason, decision=decision, decision_source="rules_degraded")


# --- The agent -----------------------------------------------------------------------------------

class _Calls:
    """Accumulates the usage of the Gemini calls of one request."""

    def __init__(self):
        self.usage = LLMUsage()
        self.trace: list[dict] = []
        self.live = {"tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0}

    def add(self, name: str, usage: LLMUsage, live: dict | None = None, error: str = ""):
        u = self.usage
        u.model = usage.model or u.model
        u.llm_calls += usage.llm_calls
        u.tokens_in += usage.tokens_in
        u.tokens_out += usage.tokens_out
        u.tokens_thinking += usage.tokens_thinking
        u.retries += usage.retries
        u.cache_hit = u.cache_hit or usage.cache_hit
        u.rate_limited = u.rate_limited or usage.rate_limited
        for k in self.live:
            self.live[k] += int((live or {}).get(k, 0))
        self.trace.append({"call": name, "cache_hit": usage.cache_hit, "llm_calls": usage.llm_calls,
                           "retries": usage.retries, "tokens_in": usage.tokens_in, "tokens_out": usage.tokens_out,
                           "tokens_thinking": usage.tokens_thinking, "error": error})

    def call(self, name: str, **kwargs) -> llm.LLMReply:
        try:
            reply = llm.generate(call=name, prompt_version=PROMPT_VERSION, **kwargs)
        except llm.LLMUnavailable as error:
            self.add(name, error.usage, error=str(error))
            raise
        self.add(name, reply.usage, reply.live_equivalent)
        return reply


def _finish(run: AgentRun, calls: _Calls) -> AgentRun:
    run.usage = calls.usage
    run.calls = calls.trace
    run.live_equivalent = calls.live
    return run


def _degraded(machine, risk, calls: _Calls, reason: str) -> AgentRun:
    return _finish(rule_based_plan(machine, risk, reason=reason), calls)


def _parse_writing(reply: llm.LLMReply, decision: Decision, tool_calls: list[ToolCall]) -> tuple["Writing | None", str]:
    data = reply.json()
    if not isinstance(data, dict):
        return None, "writing is not a JSON object"
    try:
        writing = Writing.model_validate({"steps": [str(s).strip() for s in data.get("steps", []) if str(s).strip()],
                                          "justification": str(data.get("justification", "")).strip()})
    except ValidationError as error:
        return None, f"writing does not match the schema ({error.error_count()} error(s))"
    draft = WorkOrderDraft(**decision.model_dump(), steps=writing.steps, justification=writing.justification)
    problems = guardrails.text_errors(draft, tool_calls)
    return (None, "; ".join(problems)) if problems else (writing, "")


def _cost_facts(machine: MachineInput, decision: Decision) -> dict:
    """Non-monetary cost facts for the writing call (the LLM never sees or writes euro amounts)."""
    if decision.failure_mode_addressed == FailureMode.NONE:
        return {}
    cost = tools.estimate_costs(decision.action.value, decision.failure_mode_addressed, machine.type)
    ratio = round(cost.breakdown_eur / cost.preventive_eur, 1) if cost.preventive_eur else None
    return {"planned_downtime_h": cost.downtime_h, "breakdown_to_preventive_cost_ratio": ratio}


def _low_risk(machine: MachineInput, risk: RiskAssessment, arm: str, calls: _Calls) -> AgentRun:
    decision = Decision(priority=Priority.P4, action=Action.MONITOR, failure_mode_addressed=FailureMode.NONE,
                        requires_human_review=False)
    try:
        reply = calls.call("low_risk", system=system_prompt("low_risk", arm), contents=_facts(machine, risk),
                           style=arm, response_schema=WRITING_SCHEMA, max_output_tokens=1024)
    except llm.LLMUnavailable as error:
        return _degraded(machine, risk, calls, f"llm_unavailable: {error}")
    writing, problem = _parse_writing(reply, decision, [])
    if writing is None:
        return _degraded(machine, risk, calls, f"writing_failed: {problem}")
    draft = WorkOrderDraft(**decision.model_dump(), steps=writing.steps, justification=writing.justification)
    return _finish(AgentRun(draft=draft, prompt_version=PROMPT_VERSION, decision=decision,
                            decision_source="rules_low_risk",
                            writing_warnings=writing_warnings(writing, arm, _facts(machine, risk))), calls)


def decide(machine: MachineInput, risk: RiskAssessment, calls: _Calls) -> tuple[Decision, list[ToolCall], str, list[str]]:
    """Calls 1 and 2, WITHOUT the arm. Returns (decision, tool calls, source, errors).

    Raises llm.LLMUnavailable if Gemini cannot be used (the caller degrades).
    """
    facts = _facts(machine, risk)
    reply = calls.call("tools", system=system_prompt("tools"), contents=facts, tools=TOOL_DECLARATIONS,
                       max_output_tokens=1024)
    tool_calls = run_tools(reply.function_calls)
    results = [c.model_dump(mode="json") for c in tool_calls]
    reply = calls.call("decision", system=system_prompt("decision"), contents=facts + "\n" + _data("tool_results", results),
                       response_schema=DECISION_SCHEMA, max_output_tokens=1024)
    data = reply.json()
    errors: list[str] = []
    decision = None
    if not isinstance(data, dict):
        errors.append("decision is not a JSON object")
    else:
        try:
            decision = Decision.model_validate(data)
        except ValidationError as error:
            errors += [f"schema: {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()][:5]
    if decision is not None:
        errors += guardrails.decision_errors(risk, decision, tool_calls)
    if not errors:
        return decision, tool_calls, "llm", []
    rules = rule_based_plan(machine, risk, reason="")
    merged = tool_calls + [c for c in rules.tool_calls if c not in tool_calls]
    return rules.decision, merged, "rules_fallback", errors


def plan(machine: MachineInput, risk: RiskAssessment, arm: str) -> AgentRun:
    calls = _Calls()
    if risk.risk_band == RiskBand.LOW:
        return _low_risk(machine, risk, arm, calls)
    try:
        decision, tool_calls, source, errors = decide(machine, risk, calls)  # no arm here
    except llm.LLMUnavailable as error:
        return _degraded(machine, risk, calls, f"llm_unavailable: {error}")

    def result(steps, justification, degraded=False, reason="", warnings=()):
        draft = WorkOrderDraft(**decision.model_dump(), steps=steps, justification=justification)
        return _finish(AgentRun(draft=draft, tool_calls=tool_calls, prompt_version=PROMPT_VERSION, degraded=degraded,
                                degraded_reason=reason, decision=decision, decision_source=source,
                                decision_errors=errors, writing_warnings=list(warnings)), calls)

    contents = "\n".join([_facts(machine, risk), _data("tool_results", [c.model_dump(mode="json") for c in tool_calls]),
                          _data("decision", decision.model_dump(mode="json")),
                          _data("cost_facts", _cost_facts(machine, decision))])
    try:
        reply = calls.call("write", system=system_prompt("write", arm), contents=contents, style=arm,
                           response_schema=WRITING_SCHEMA)
    except llm.LLMUnavailable as error:
        return result(*_rule_text(decision, risk), degraded=True, reason=f"llm_unavailable at writing: {error}")
    writing, problem = _parse_writing(reply, decision, tool_calls)
    if writing is None:
        return result(*_rule_text(decision, risk), degraded=True, reason=f"writing_failed: {problem}")
    return result(writing.steps, writing.justification, warnings=writing_warnings(writing, arm, contents))

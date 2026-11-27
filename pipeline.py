"""
MaintAI's flow: component A (tabular ML) feeds component B (agentic LLM).  OWNER: WP3.

The modules live in maintai/. This file only chains them and builds the record
that app.py logs. It keeps the template's contract, which app.py, evaluate.py,
main.py and check.py read:

    run(raw_input) -> {
        "input_summary": str,          # short, no personal data: it is logged
        "features": {name: number},    # what the drift check watches
        "outputs": {component: ...},   # each component's output, JSON-serialisable
        "versions": {component: str},  # model and prompt versions
        "review": bool,                # does this case go to a person?
        "tokens": int,                 # tokens spent on this request (0 if none)
        "cost_eur": float,             # production-equivalent cost of this request in euros
    }

Extra fields (arm, degraded, llm usage...) are added after these and are logged too:

    arm, risk_band, status       the writing style used, component A's band, ok | out_of_scope
    degraded, degraded_reason    True when the rule-based template replaced (part of) the agent
    decision_source              llm | rules_fallback | rules_low_risk | rules_degraded
    decision_errors              why call 2's decision was rejected (rules_fallback)
    writing_warnings             prompt rules the text broke (raw field names, steps not in the data...)
    llm                          schemas.LLMUsage of the request (all Gemini calls together)
    llm_calls, cache_hit, retries, rate_limited   copies of the main llm counters, for the dashboards
    llm_trace                    one entry per Gemini call (name, cache hit, tokens, error)
    llm_live_equivalent          tokens and euros the request would cost live (cache hits included),
                                 so the evaluation can report cost even when it replays the cache
    llm_budget                   Gemini requests used today / daily budget, when served

`arm` (decision D1): app.serve() passes the arm chosen by the bandit. When it is
None (evaluate.py, main.py, check.py), a deterministic arm is used so the LLM
cache gives the same answer every time.
"""

from maintai import agent, bandit, component_a, economics, features, guardrails, inputs, llm
from maintai.schemas import OutOfScope


def run(raw_input: str, arm: str | None = None) -> dict:
    parsed = inputs.parse(raw_input)
    if isinstance(parsed, OutOfScope):
        return _out_of_scope(parsed)

    machine = parsed
    risk = component_a.assess(machine)
    arm = arm or bandit.offline_arm(raw_input)
    result = agent.plan(machine, risk, arm)
    order = guardrails.finalise(machine, risk, result)
    usage = result.usage
    decision = getattr(result, "decision", None)
    source = getattr(result, "decision_source", "unknown")
    live = dict(getattr(result, "live_equivalent", {}) or {})
    live["cost_eur"] = economics.llm_cost_eur(usage.model, live.get("tokens_in", 0),
                                              live.get("tokens_out", 0) + live.get("tokens_thinking", 0))

    return {
        "input_summary": inputs.summary(machine),
        "features": features.drift_features(machine, risk),
        "outputs": {
            "component_a": risk.model_dump(mode="json"),
            "component_b": {
                "work_order": order.model_dump(mode="json"),
                "tool_calls": [c.model_dump(mode="json") for c in result.tool_calls],
                "decision": decision.model_dump(mode="json") if decision else None,  # arm-free
                "decision_source": source,
            },
        },
        "versions": {
            "component_a": risk.model_version,
            "component_b": f"{usage.model or 'none'}|{result.prompt_version}",
            "strategy": arm,
            "bandit": bandit.POLICY_VERSION,
        },
        "review": order.review,
        "tokens": usage.tokens,
        "cost_eur": economics.llm_cost_eur(usage.model, usage.tokens_in, usage.tokens_out + usage.tokens_thinking),
        # --- extra fields (logged as well) ---
        "status": "ok",
        "arm": arm,
        "risk_band": risk.risk_band.value,
        "degraded": result.degraded,
        "degraded_reason": result.degraded_reason,
        "llm": usage.model_dump(),
        "llm_calls": usage.llm_calls,
        "cache_hit": usage.cache_hit,
        "retries": usage.retries,
        "rate_limited": usage.rate_limited,
        "decision_source": source,
        "decision_errors": list(getattr(result, "decision_errors", []) or []),
        "writing_warnings": list(getattr(result, "writing_warnings", []) or []),
        "llm_trace": list(getattr(result, "calls", []) or []),
        "llm_live_equivalent": live,
        "llm_budget": _budget(),
    }


def _budget() -> dict:
    status = llm.budget_status()
    return {"day": status["day"], "used": status["used"], "limit": status["limit"]}


def _out_of_scope(parsed: OutOfScope) -> dict:
    """Refused before any model call (go/no-go criterion 6). No work order is produced."""
    refusal = {"status": "out_of_scope", "reason": parsed.reason, "detail": parsed.detail}
    return {
        "input_summary": f"out of scope: {parsed.reason}",
        "features": {},
        "outputs": {"component_a": refusal, "component_b": refusal},
        "versions": {"component_a": component_a.MODEL_VERSION, "component_b": agent.PROMPT_VERSION},
        "review": False,
        "tokens": 0,
        "cost_eur": 0.0,
        "status": "out_of_scope",
        "degraded": False,
        "degraded_reason": "",
        "decision_source": "refused",
        "llm_calls": 0,
        "cache_hit": False,
        "retries": 0,
    }

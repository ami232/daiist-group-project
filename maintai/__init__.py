"""MaintAI – Intelligent Maintenance Advisor.

Package layout (who owns what is in the master plan, section 09):

    schemas.py      CONTRACTS between modules. Change only through a reviewed PR.
    config.py       settings read from the environment
    inputs.py       parse and validate the raw input, refuse out-of-scope inputs
    features.py     physical indicators and the drift features
    component_a.py  failure-risk model (tabular ML)            -> WP1
    tools.py        simulated plant tools used by the agent     -> WP2
    economics.py    the single cost model (REPORT.md)           -> WP2
    agent.py        Gemini work-order agent (agentic LLM)       -> WP3
    llm.py          Gemini client: rate limit, budget, cache    -> WP3
    guardrails.py   validation, safety rules, review decision   -> WP3
    bandit.py       Thompson-sampling bandit over writing styles -> WP6

Modules marked STUB return fixed, valid outputs so that the whole app runs end
to end before the real components exist. Replace a stub without changing its
function signature or the schema it returns.
"""

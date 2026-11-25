"""The single cost model of REPORT.md ("Economic model").  OWNER: WP2.

Used by the agent's tools, the evaluation and the report, so there is only one
version of every number. All values are ASSUMPTIONS (approved by Manuela,
pending confirmation by the rest of the team).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostAssumptions:
    intervention_eur: float = 300.0  # C_I
    failure_eur: float = 4_000.0  # C_F
    prevention_effectiveness: float = 0.8  # e
    review_eur: float = 15.0  # C_R
    processes_per_day: int = 1_000  # V (machine cycles assessed, not machines)


DEFAULT = CostAssumptions()


def outcome_cost(intervened: bool, failed: bool, reviewed: bool, llm_eur: float = 0.0,
                 a: CostAssumptions = DEFAULT) -> float:
    """Cost of one process for one decision, as in the outcome table of REPORT.md."""
    if intervened and failed:
        cost = a.intervention_eur + (1 - a.prevention_effectiveness) * a.failure_eur
    elif intervened:
        cost = a.intervention_eur
    elif failed:
        cost = a.failure_eur
    else:
        cost = 0.0
    return cost + (a.review_eur if reviewed else 0.0) + llm_eur


# Paid Gemini prices per 1M tokens, Standard tier, from the Gemini pricing page (checked 25 Nov 2026).
# We use the free tier, so the money really paid is 0; these give the production-equivalent cost.
PRICE_USD_PER_M = {"gemini-3.5-flash-lite": {"in": 0.30, "out": 2.50}}  # output price includes thinking
USD_TO_EUR = 0.90  # TODO(WP3): placeholder ASSUMPTION; replace with a dated exchange rate and cite it


def llm_cost_eur(model: str, tokens_in: int, tokens_out_and_thinking: int) -> float:
    """Production-equivalent cost of the tokens of one request, in euros."""
    price = PRICE_USD_PER_M.get(model)
    if price is None:
        return 0.0
    usd = tokens_in / 1e6 * price["in"] + tokens_out_and_thinking / 1e6 * price["out"]
    return round(usd * USD_TO_EUR, 6)


def break_even_probability(a: CostAssumptions = DEFAULT) -> float:
    """Simplified break-even p* = C_I / (e * C_F), without review or AI costs."""
    return a.intervention_eur / (a.prevention_effectiveness * a.failure_eur)

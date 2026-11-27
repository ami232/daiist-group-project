"""The single cost model of REPORT.md ("Economic model").  OWNER: WP2.

Used by the agent's tools, the evaluation and the report, so there is only one
version of every number. All values are ASSUMPTIONS (approved by Manuela,
pending confirmation by the rest of the team).
"""

from dataclasses import dataclass

import numpy as np


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


# Paid Gemini prices per 1M tokens, Standard tier. Source: Google AI for Developers pricing page
# (https://ai.google.dev/pricing), read on 27 Nov 2026; the page shows no update date.
# We use the free tier, so the money really paid is 0; these give the production-equivalent cost.
PRICE_USD_PER_M = {"gemini-3.5-flash-lite": {"in": 0.30, "out": 2.50}}  # output price includes thinking
# Exchange rate: ECB euro reference rate, 1.0989 USD per EUR on 17 Nov 2026, as published by the Banca d'Italia
# (https://www.bancaditalia.it/compiti/operazioni-cambi/cambio/cambi_rif_20261117). A fixed, dated rate keeps
# results reproducible; the cost of the AI is tiny, so a few cents of FX change nothing.
USD_PER_EUR = 1.0989
USD_TO_EUR = round(1 / USD_PER_EUR, 4)
FX_SOURCE = "ECB reference rate 1.0989 USD/EUR, 17 Nov 2026 (via Banca d'Italia)"
PRICE_SOURCE = "Google AI for Developers pricing page, Standard tier, read 27 Nov 2026"


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


# --- Evaluation of policies on a labelled split (WP2) ------------------------------------------------
# A "policy" is just two boolean arrays per process: did we intervene, did a person review. The same
# functions score MaintAI and every baseline (B0, B1, B2), so all of them are paid with one cost model.
# Everything below is computed by Python. Savings are model outputs under ASSUMPTIONS, never observed savings.

OPERATING_DAYS_PER_MONTH = 30  # [Assumption] days per month the plant assesses processes
REVIEW_HOURS = 0.25  # [Assumption] about 15 minutes of a supervisor per review (REPORT.md)


def cost_vector(y_true, intervene, review=None, llm_eur=0.0, a: CostAssumptions = DEFAULT) -> np.ndarray:
    """Cost of every process (vectorised version of outcome_cost). `llm_eur` is a scalar or an array."""
    y = np.asarray(y_true, dtype=bool)
    d = np.asarray(intervene, dtype=bool)
    cost = np.where(d & y, a.intervention_eur + (1 - a.prevention_effectiveness) * a.failure_eur,
                    np.where(d, a.intervention_eur, np.where(y, a.failure_eur, 0.0)))
    if review is not None:
        cost = cost + a.review_eur * np.asarray(review, dtype=bool)
    return cost + np.asarray(llm_eur, dtype=float)


def policy_summary(y_true, intervene, review=None, llm_eur=0.0, a: CostAssumptions = DEFAULT) -> dict:
    """Outcome counts and expected cost per 1,000 processes of one policy."""
    y = np.asarray(y_true, dtype=bool)
    d = np.asarray(intervene, dtype=bool)
    n = len(y)
    r = np.zeros(n, dtype=bool) if review is None else np.asarray(review, dtype=bool)
    tp, fp, fn, tn = int((d & y).sum()), int((d & ~y).sum()), int((~d & y).sum()), int((~d & ~y).sum())
    return {
        "n": n, "failures": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "cost_per_1000": float(cost_vector(y, d, r, llm_eur, a).mean() * 1000),
        "interventions_per_1000": d.mean() * 1000,
        "reviews_per_1000": r.mean() * 1000,
        "missed_failures_per_1000": fn / n * 1000,
        "unnecessary_interventions_per_1000": fp / n * 1000,
    }


def paired_saving(cost_baseline, cost_policy, n_boot: int = 2000, seed: int = 0) -> dict:
    """Saving of a policy against a baseline per 1,000 processes, with a paired bootstrap 95% interval.

    Both cost vectors must refer to the same processes in the same order. Each bootstrap sample resamples
    processes once and applies the same resample to both policies (paired), so the interval reflects the
    difference, not the two costs separately. Assumes independent processes (the dataset has some
    autocorrelation between neighbouring rows: the interval is a guide, not a guarantee).
    """
    b = np.asarray(cost_baseline, dtype=float)
    m = np.asarray(cost_policy, dtype=float)
    if b.shape != m.shape or b.ndim != 1 or len(b) == 0:
        raise ValueError("cost vectors must be 1-D, non-empty and of equal length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(n_boot, len(b)))
    base_s, pol_s = b[idx].mean(axis=1), m[idx].mean(axis=1)
    save = (base_s - pol_s) * 1000
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(base_s > 0, (base_s - pol_s) / base_s, np.nan)
    lo, hi = np.percentile(save, [2.5, 97.5])
    rlo, rhi = np.nanpercentile(rel, [2.5, 97.5])
    base_mean = b.mean()
    return {"saving_per_1000": float((b.mean() - m.mean()) * 1000), "ci_low": float(lo), "ci_high": float(hi),
            "relative_saving": float((base_mean - m.mean()) / base_mean) if base_mean > 0 else None,
            "relative_ci_low": float(rlo), "relative_ci_high": float(rhi), "n_boot": n_boot, "seed": seed}


def default_review_rule(p, threshold: float, critical: float = 0.8):
    """Which processes MaintAI sends to a person: critical risk or the uncertainty band around the threshold.

    Mirrors the band logic of component A (threshold / 2 <= p < 2 * threshold). WP1 owns the real bands:
    pass the real rule to the functions below when it exists.
    """
    p = np.asarray(p, dtype=float)
    return (p >= critical) | ((p >= threshold / 2) & (p < threshold * 2))


def tune_threshold(p, y_true, a: CostAssumptions = DEFAULT, review_rule=default_review_rule, llm_eur: float = 0.0,
                   grid=None) -> dict:
    """Decision threshold with the lowest expected cost, using the FULL cost model (review and AI included).

    Call it ONLY with training or validation data, never with the test split.
    Ties go to the lowest threshold (more recall). Returns the threshold and its cost per 1,000.
    """
    p = np.asarray(p, dtype=float)
    grid = np.round(np.arange(0.01, 0.99, 0.01), 2) if grid is None else np.asarray(grid, dtype=float)
    best = None
    for t in grid:
        c = cost_vector(y_true, p >= t, review_rule(p, t), llm_eur, a).mean() * 1000
        if best is None or c < best[1] - 1e-9:
            best = (float(t), float(c))
    return {"threshold": best[0], "cost_per_1000": best[1]}


def tune_cutoff(score, y_true, a: CostAssumptions = DEFAULT, candidates=None, higher_is_riskier: bool = True) -> dict:
    """Best cutoff of a single rule (for example B1: intervene when tool wear >= T). No review, no AI cost.

    Tuning data only (training or validation). Use `higher_is_riskier=False` for 'score <= cutoff' rules.
    """
    s = np.asarray(score, dtype=float)
    cand = np.unique(s) if candidates is None else np.asarray(candidates, dtype=float)
    best = None
    for c in cand:
        d = s >= c if higher_is_riskier else s <= c
        cost = cost_vector(y_true, d, None, 0.0, a).mean() * 1000
        if best is None or cost < best[1] - 1e-9:
            best = (float(c), float(cost))
    return {"cutoff": best[0], "cost_per_1000": best[1]}


def daily_load(review_rate: float, intervention_rate: float, processes_per_day: int | None = None,
               a: CostAssumptions = DEFAULT) -> dict:
    """Reviews, review hours and interventions per day = representative rate x daily volume V."""
    v = a.processes_per_day if processes_per_day is None else processes_per_day
    reviews = review_rate * v
    return {"processes_per_day": v, "reviews_per_day": reviews, "review_hours_per_day": reviews * REVIEW_HOURS,
            "interventions_per_day": intervention_rate * v}


def monthly_value(saving_per_1000: float, processes_per_day: int | None = None,
                  days: int = OPERATING_DAYS_PER_MONTH, a: CostAssumptions = DEFAULT) -> float:
    """Net simulated saving per month in euros (review and AI costs are already inside the saving)."""
    v = a.processes_per_day if processes_per_day is None else processes_per_day
    return saving_per_1000 * v / 1000 * days
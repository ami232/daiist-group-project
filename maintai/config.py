"""Settings read from the environment (.env locally, the Environment page on Render).

Never write a key here. Read it with gemini_api_key() when you need it.
"""

import os

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # TODO(WP3): confirm the exact model id in the Gemini docs
DEFAULT_JUDGE_MODEL = "gemini-3.8-flash"  # TODO(WP4): confirm the exact model id in the Gemini docs
DEFAULT_DAILY_BUDGET = 400  # of the 500 requests a day of the free tier
CACHE_MODES = ("readwrite", "readonly", "off")


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def gemini_api_key() -> str:
    """The key, or "" if it is not set (the app then runs in degraded mode)."""
    return _get("GEMINI_API_KEY")


def agent_model() -> str:
    return _get("GEMINI_MODEL", DEFAULT_MODEL)


def judge_model() -> str:
    return _get("GEMINI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)


def daily_budget() -> int:
    try:
        return int(_get("DAILY_LLM_BUDGET", str(DEFAULT_DAILY_BUDGET)))
    except ValueError:
        return DEFAULT_DAILY_BUDGET


def cache_mode() -> str:
    mode = _get("LLM_CACHE", "readonly").lower()
    return mode if mode in CACHE_MODES else "readonly"

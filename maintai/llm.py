"""Gemini client.  OWNER: WP3.

STUB. No call to Gemini is made yet. WP3 implements, behind `generate()`:

- rate limit of 12 requests per minute (the free tier allows 15);
- daily budget (config.daily_budget()), counted from the logs so it survives restarts;
- up to 2 retries on 429 errors, then raise LLMUnavailable;
- token accounting from the response's usage metadata -> LLMUsage;
- the versioned cache in eval/llm_cache/ (config.cache_mode()). On Render the mode is
  "off", so live traffic is never served from the cache.
"""

from maintai import config
from maintai.schemas import LLMUsage


class LLMUnavailable(Exception):
    """No key, budget used up, or repeated 429s. The caller switches to degraded mode."""


def available() -> bool:
    return bool(config.gemini_api_key())


def generate(*args, **kwargs) -> tuple[dict, LLMUsage]:
    raise LLMUnavailable("STUB: the Gemini client is not implemented yet (WP3).")

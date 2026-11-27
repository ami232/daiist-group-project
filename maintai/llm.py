"""Gemini client.  OWNER: WP3.

Every Gemini call of the agent goes through `generate()`, which applies, in this order:

1. the versioned cache in eval/llm_cache/ (config.cache_mode()):
   - readwrite: a hit is returned without calling Gemini; a miss calls Gemini and is saved;
   - readonly:  a hit is returned; a miss raises LLMCacheMiss (explicit, no call is made);
   - off:       always live (Render sets LLM_CACHE=off, so live traffic never comes from the cache).
   The key is a SHA-256 of the model, prompt version, writing style, call name, system prompt,
   input, generation parameters, tool declarations and response schema. Entries hold only the
   model's output and token counts: never the API key, never the request headers.
2. no key -> LLMUnavailable (the app then runs in degraded mode);
3. the daily budget (config.daily_budget()), counted in Gemini API requests per Pacific-time day
   (the day the free-tier quota uses). The count is rebuilt from the logs once per process and
   day (requests logged earlier today: llm_calls + retries), plus the calls this process made,
   so it survives restarts;
4. the rate limit, RATE_LIMIT_RPM requests in any 60 s (the free tier allows more). A call that
   would have to wait longer than MAX_RATE_WAIT_S is refused with LLMUnavailable;
5. the call itself, with a timeout, temperature 0, and at most MAX_RETRIES retries on 429 / 503,
   waiting at most MAX_RETRY_WAIT_S each time; then LLMUnavailable.

Token accounting comes from the response's usage metadata. Cache hits cost nothing, so their
LLMUsage has zero tokens and llm_calls = 0 (schemas.LLMUsage); the tokens of the original call are
kept in the cache entry and returned in `live_equivalent`, so the evaluation can still report the
cost a live request would have had.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from maintai import config
from maintai.schemas import LLMUsage

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "eval" / "llm_cache"
CACHE_FORMAT = "llm-cache-v1"  # bump if the shape of a cache entry changes

RATE_LIMIT_RPM = 12
MAX_RATE_WAIT_S = 20.0
MAX_RETRIES = 2
MAX_RETRY_WAIT_S = 8.0
TIMEOUT_S = 30.0
TEMPERATURE = 0.0
THINKING_LEVEL = "low"  # supported by gemini-3.5-flash-lite and gemini-3.8-flash ("minimal" is not by 3.8)
MAX_OUTPUT_TOKENS = 2048
RETRYABLE = {429, 503}


class LLMUnavailable(Exception):
    """No key, budget used up, rate limit, repeated 429s or an API error. The caller degrades.

    `usage` holds what was spent before giving up (retries, rate_limited), so it is logged too.
    """

    def __init__(self, message: str, usage: LLMUsage | None = None):
        super().__init__(message)
        self.usage = usage or LLMUsage()


class LLMCacheMiss(LLMUnavailable):
    """LLM_CACHE=readonly and this exact call is not in the cache. No call was made."""


@dataclass
class LLMReply:
    """What one generate() call returns."""

    text: str = ""
    function_calls: list[dict] = field(default_factory=list)  # [{"name": str, "args": dict}]
    usage: LLMUsage = field(default_factory=LLMUsage)
    live_equivalent: dict = field(default_factory=dict)  # tokens of the original call (cache hits too)
    cache_key: str = ""

    def json(self) -> dict | list | None:
        """The text parsed as JSON, or None if it is not valid JSON."""
        text = self.text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None


def available() -> bool:
    return bool(config.gemini_api_key())


# --- Cache -----------------------------------------------------------------------------------------

def cache_key(*, model: str, prompt_version: str, style: str | None, call: str, system: str, contents: str,
              params: dict, tools: list[dict] | None, response_schema: dict | None) -> str:
    material = {"format": CACHE_FORMAT, "model": model, "prompt_version": prompt_version, "style": style,
                "call": call, "system": system, "contents": contents, "params": params,
                "tools": tools or [], "response_schema": response_schema}
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / key[:2] / f"{key}.json"


def _cache_read(key: str) -> dict | None:
    path = _cache_path(key)
    if not path.is_file():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return entry if entry.get("format") == CACHE_FORMAT else None


def _cache_write(key: str, entry: dict) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)  # atomic: a reader never sees half an entry


# --- Daily budget (Gemini requests per Pacific-time day) -------------------------------------------

def _quota_day(now: datetime | None = None) -> str:
    """The free-tier quota resets at midnight Pacific time."""
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        return now.astimezone(ZoneInfo("America/Los_Angeles")).date().isoformat()
    except Exception:  # noqa: BLE001 - no tz database (e.g. Windows without tzdata): UTC-8 is close enough
        return (now - timedelta(hours=8)).date().isoformat()


_PROCESS_START = datetime.now(timezone.utc)
_lock = threading.Lock()
_budget = {"day": None, "from_logs": 0, "this_process": 0, "logs_error": ""}


def _requests_in_logs_today(day: str) -> int:
    """Gemini requests recorded in the logs for `day`, before this process started.

    Requests logged by this process are counted in memory instead, so nothing is counted twice.
    """
    import logbook  # repo-root module; imported here so maintai stays importable on its own

    total = 0
    for record in logbook._read("requests"):
        llm = record.get("llm") if isinstance(record, dict) else None
        if not isinstance(llm, dict):
            continue
        try:
            ts = datetime.fromisoformat(str(record.get("timestamp")))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < _PROCESS_START and _quota_day(ts) == day:
            total += int(llm.get("llm_calls") or 0) + int(llm.get("retries") or 0)
    return total


def budget_status() -> dict:
    """{"day", "used", "limit", "remaining", "logs_error"}. Reads the logs once per process and day."""
    day = _quota_day()
    with _lock:
        if _budget["day"] != day:
            try:
                from_logs, error = _requests_in_logs_today(day), ""
            except Exception as exc:  # noqa: BLE001 - logs unreadable: count this process only, and say so
                from_logs, error = 0, f"logs unreadable ({type(exc).__name__})"
            _budget.update(day=day, from_logs=from_logs, this_process=0, logs_error=error)
        used = _budget["from_logs"] + _budget["this_process"]
    limit = config.daily_budget()
    return {"day": day, "used": used, "limit": limit, "remaining": max(limit - used, 0),
            "logs_error": _budget["logs_error"]}


def _spend_one() -> None:
    """Reserve one request of today's budget, or raise LLMUnavailable."""
    status = budget_status()
    with _lock:
        if _budget["from_logs"] + _budget["this_process"] >= status["limit"]:
            raise LLMUnavailable(f"daily LLM budget used up ({status['limit']} requests on {status['day']})")
        _budget["this_process"] += 1


def reset_state() -> None:
    """Forget the in-memory budget and rate-limit state (tests only)."""
    with _lock:
        _budget.update(day=None, from_logs=0, this_process=0, logs_error="")
        _recent.clear()


# --- Rate limit (sliding 60 s window) --------------------------------------------------------------

_recent: deque[float] = deque()


def _wait_for_slot() -> bool:
    """Block until a request fits in the RPM window. Returns True if it had to wait."""
    waited = False
    while True:
        with _lock:
            now = time.monotonic()
            while _recent and now - _recent[0] >= 60:
                _recent.popleft()
            if len(_recent) < RATE_LIMIT_RPM:
                _recent.append(now)
                return waited
            wait = 60 - (now - _recent[0]) + 0.05
        if wait > MAX_RATE_WAIT_S:
            raise LLMUnavailable(f"rate limit: {RATE_LIMIT_RPM} requests in the last minute")
        waited = True
        _sleep(wait)


def _sleep(seconds: float) -> None:  # patched in tests
    time.sleep(seconds)


# --- The API call ----------------------------------------------------------------------------------

_client = None
_client_key = None


def _get_client():
    global _client, _client_key
    key = config.gemini_api_key()
    if _client is None or _client_key != key:
        from google import genai
        from google.genai import types

        _client = genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=int(TIMEOUT_S * 1000), retry_options=types.HttpRetryOptions(attempts=1)))
        _client_key = key
    return _client


def _call_api(model: str, system: str, contents: str, params: dict, tools: list[dict] | None,
              response_schema: dict | None) -> dict:
    """One request to Gemini. Returns {"text", "function_calls", "usage": {...raw counts}}.

    Tests replace this function with a fake Gemini, so no quota is spent offline.
    """
    from google.genai import types

    cfg = {"system_instruction": system, "temperature": params["temperature"],
           "max_output_tokens": params["max_output_tokens"],
           "thinking_config": types.ThinkingConfig(thinking_level=params["thinking_level"]),
           "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True)}
    if tools:
        cfg["tools"] = [types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t["name"], description=t["description"],
                                      parameters_json_schema=t["parameters"]) for t in tools])]
        cfg["tool_config"] = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY"))
    if response_schema:
        cfg["response_mime_type"] = "application/json"
        cfg["response_json_schema"] = response_schema
    response = _get_client().models.generate_content(model=model, contents=contents,
                                                     config=types.GenerateContentConfig(**cfg))
    texts, calls = [], []
    for candidate in (response.candidates or [])[:1]:
        for part in (candidate.content.parts if candidate.content and candidate.content.parts else []):
            if part.function_call:
                calls.append({"name": part.function_call.name, "args": dict(part.function_call.args or {})})
            elif part.text and not part.thought:
                texts.append(part.text)
    meta = response.usage_metadata
    usage = {k: int(getattr(meta, k, 0) or 0) for k in ("prompt_token_count", "candidates_token_count",
                                                       "thoughts_token_count", "tool_use_prompt_token_count",
                                                       "total_token_count")} if meta else {}
    return {"text": "".join(texts), "function_calls": calls, "usage": usage}


def split_tokens(raw: dict) -> tuple[int, int, int]:
    """(tokens_in, tokens_out, tokens_thinking) from raw usage counts, never counting thinking twice.

    Gemini reports thoughts_token_count separately from candidates_token_count; if a response's
    total shows that the candidates count already included the thoughts, they are subtracted.
    """
    p = raw.get("prompt_token_count", 0) + raw.get("tool_use_prompt_token_count", 0)
    c, t, total = raw.get("candidates_token_count", 0), raw.get("thoughts_token_count", 0), raw.get("total_token_count", 0)
    if t and total and total == p + c and c >= t:  # thinking already inside the candidates count
        c -= t
    return p, c, t


def _retry_delay(error: Exception, attempt: int) -> float:
    """Server-suggested delay (RetryInfo) or exponential backoff, capped at MAX_RETRY_WAIT_S."""
    suggested = None
    details = getattr(error, "details", None) or {}
    try:
        for item in (details.get("error", details) or {}).get("details", []) or []:
            if "retryDelay" in item:
                suggested = float(str(item["retryDelay"]).rstrip("s"))
    except (AttributeError, TypeError, ValueError):
        suggested = None
    delay = suggested if suggested is not None else 2.0 * (2 ** attempt) + random.uniform(0, 0.5)
    return min(max(delay, 0.5), MAX_RETRY_WAIT_S)


def generate(*, call: str, system: str, contents: str, prompt_version: str, style: str | None = None,
             tools: list[dict] | None = None, response_schema: dict | None = None,
             max_output_tokens: int = MAX_OUTPUT_TOKENS) -> LLMReply:
    """One logical Gemini call (cache, key, budget, rate limit, retries). Raises LLMUnavailable."""
    model = config.agent_model()
    params = {"temperature": TEMPERATURE, "max_output_tokens": max_output_tokens, "thinking_level": THINKING_LEVEL}
    key = cache_key(model=model, prompt_version=prompt_version, style=style, call=call, system=system,
                    contents=contents, params=params, tools=tools, response_schema=response_schema)
    mode = config.cache_mode()

    if mode in ("readwrite", "readonly"):
        entry = _cache_read(key)
        if entry is not None:
            out = entry["output"]
            return LLMReply(text=out.get("text", ""), function_calls=out.get("function_calls", []),
                            usage=LLMUsage(model=model, cache_hit=True),
                            live_equivalent=entry.get("usage", {}), cache_key=key)
        if mode == "readonly":
            raise LLMCacheMiss(f"LLM_CACHE=readonly and call '{call}' is not cached ({key[:12]})")

    if not available():
        raise LLMUnavailable("GEMINI_API_KEY is not set")

    from google.genai import errors

    usage = LLMUsage(model=model)
    attempt = 0
    while True:
        try:
            if _wait_for_slot():
                usage.rate_limited = True
            _spend_one()
        except LLMUnavailable as error:
            if attempt:  # this retry was never sent: count only the requests really made
                usage.retries -= 1
                usage.llm_calls = 1
            raise LLMUnavailable(str(error), usage) from None
        try:
            raw = _call_api(model, system, contents, params, tools, response_schema)
            break
        except errors.APIError as error:
            code = getattr(error, "code", None)
            if code == 429:
                usage.rate_limited = True
            if code in RETRYABLE and attempt < MAX_RETRIES:
                usage.retries += 1
                _sleep(_retry_delay(error, attempt))
                attempt += 1
                continue
            usage.llm_calls = 1  # requests sent = llm_calls + retries (what the daily budget counts)
            raise LLMUnavailable(f"Gemini API error {code} after {usage.retries} retries", usage) from error
        except Exception as error:  # noqa: BLE001 - timeouts, network: degrade instead of failing the request
            usage.llm_calls = 1
            raise LLMUnavailable(f"Gemini call failed ({type(error).__name__})", usage) from error

    tokens_in, tokens_out, tokens_thinking = split_tokens(raw.get("usage", {}))
    usage.llm_calls = 1
    usage.tokens_in, usage.tokens_out, usage.tokens_thinking = tokens_in, tokens_out, tokens_thinking
    counts = {"tokens_in": tokens_in, "tokens_out": tokens_out, "tokens_thinking": tokens_thinking}
    reply = LLMReply(text=raw.get("text", ""), function_calls=raw.get("function_calls", []), usage=usage,
                     live_equivalent=counts, cache_key=key)
    if mode == "readwrite":
        _cache_write(key, {"format": CACHE_FORMAT, "model": model, "prompt_version": prompt_version,
                           "style": style, "call": call, "created": datetime.now(timezone.utc).date().isoformat(),
                           "output": {"text": reply.text, "function_calls": reply.function_calls}, "usage": counts})
    return reply

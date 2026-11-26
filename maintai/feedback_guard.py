"""Session tags and feedback limits (WP6/WP7, abuse protection). No personal data.

What a session tag is
    Gradio gives every page load a random `session_hash`. We log only
    sha256(per-process random salt + session_hash), cut to 16 hex characters. No
    name, e-mail, IP address, user agent, cookie or device fingerprint is read or
    stored. The salt is drawn when the server process starts and is never written
    anywhere, so a tag cannot be linked to the browser session after a restart,
    and tags from different restarts cannot be linked to each other.

What it is NOT
    It is pseudonymous, not anonymous: within one server process, all requests and
    ratings from the same page load share a tag and can be linked to each other
    (that is the point of the cap). It is not an identity. Reloading the page, opening a private window or using a
    script gives a new session. The limits below bound casual or accidental abuse
    (a user clicking many times, one session flooding the policy); they cannot stop
    a determined person with many sessions. Real identity would need authentication,
    which the project does not add: check.py and the grader must reach /run freely.

Limits (assumptions, decided by Manuela on 2026-11-26; see docs/WP6_BANDIT.md)
    - MAX_EVENTS_PER_REQUEST rating clicks per request (a user may correct a rating;
      only the latest counts, as before).
    - MAX_EVENTS_PER_SESSION rating clicks per session are written to the logs at
      all. Beyond that, clicks are refused, which also protects the free database.
    - bandit.MAX_RATINGS_PER_SESSION rated requests per session train the policy
      (applied when the policy is rebuilt from the logs, so it survives restarts).

The in-memory counters live in this process only (bounded, expire after
SESSION_TTL_S). A restart resets them; the bandit cap does not depend on them.
Quota protection for /run stays where it was: the daily LLM budget (WP3), the
queue size and the concurrency limit (app.py).
"""

import hashlib
import secrets
import threading
import time
from collections import OrderedDict

MAX_EVENTS_PER_REQUEST = 3
MAX_EVENTS_PER_SESSION = 60
SESSION_TTL_S = 6 * 3600
MAX_TRACKED_SESSIONS = 5000

_SALT = secrets.token_hex(16)  # per process, never logged


def session_tag(session_hash: str | None) -> str | None:
    """Random, temporary tag for a Gradio session, or None outside the UI (check.py, tests)."""
    if not session_hash:
        return None
    return hashlib.sha256(f"{_SALT}:{session_hash}".encode("utf-8")).hexdigest()[:16]


def tag_from_request(request) -> str | None:
    """Session tag from a gr.Request. Reads only `session_hash`, never headers or the client address."""
    return session_tag(getattr(request, "session_hash", None)) if request is not None else None


class FeedbackGuard:
    """Per-process counters of rating clicks per session and per request."""

    def __init__(self, per_request: int = MAX_EVENTS_PER_REQUEST, per_session: int = MAX_EVENTS_PER_SESSION,
                 ttl_s: float = SESSION_TTL_S, max_sessions: int = MAX_TRACKED_SESSIONS, clock=time.monotonic):
        self.per_request, self.per_session = per_request, per_session
        self.ttl_s, self.max_sessions, self.clock = ttl_s, max_sessions, clock
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        while self._sessions:
            tag, state = next(iter(self._sessions.items()))
            if now - state["seen"] <= self.ttl_s and len(self._sessions) <= self.max_sessions:
                break
            self._sessions.pop(tag)

    def allow(self, tag: str | None, request_id: str) -> tuple[bool, str]:
        """Record one rating click if it is within the limits. Returns (allowed, reason)."""
        if not tag:  # outside the UI there is no session; the UI always has one
            return True, "no session"
        with self._lock:
            now = self.clock()
            state = self._sessions.pop(tag, None)
            if state is None or now - state["seen"] > self.ttl_s:  # new or expired session: fresh counters
                state = {"total": 0, "per_request": {}, "seen": now}
            state["seen"] = now
            self._sessions[tag] = state  # most recently used last
            self._expire(now)
            if state["total"] >= self.per_session:
                return False, "session_limit"
            if state["per_request"].get(request_id, 0) >= self.per_request:
                return False, "request_limit"
            state["total"] += 1
            state["per_request"][request_id] = state["per_request"].get(request_id, 0) + 1
            return True, "ok"

    def tracked_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

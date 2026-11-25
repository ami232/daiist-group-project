"""Thompson-sampling bandit over the three work-order writing strategies.  OWNER: WP6.

STUB. Always returns the default arm. WP6 implements (decision D6):

- posterior Beta(alpha, beta) per arm = simulated prior (at most 10 pseudo-observations
  per arm) + real 👍/👎 rebuilt from the logs (logbook.load()), so it survives restarts;
- only source == "real", same prompt version, not degraded; the latest feedback per
  request counts once (logbook.load() already keeps the latest);
- choose_arm samples once from each posterior and returns the best arm and the
  probabilities, which are logged.

The bandit only picks the writing strategy. It never changes the action, the
review decision or any safety rule.
"""

import hashlib

from maintai.schemas import ARMS

POLICY_VERSION = "stub-bandit-v0"
DEFAULT_ARM = "checklist"


def offline_arm(raw_input: str) -> str:
    """Deterministic arm for evaluation, main.py and check.py (decision D1), so the cache works."""
    digest = hashlib.sha256(str(raw_input).encode("utf-8")).digest()
    return ARMS[digest[0] % len(ARMS)]


def choose_arm(logs=None) -> tuple[str, dict[str, float]]:
    """Live choice for app.serve(). Returns (arm, probability of each arm being chosen)."""
    return DEFAULT_ARM, {arm: float(arm == DEFAULT_ARM) for arm in ARMS}

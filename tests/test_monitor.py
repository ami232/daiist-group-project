"""WP5 tests for the drift check (no Gemini, no network). Run: uv run python tests/test_monitor.py"""
import sys
from pathlib import Path

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor  # noqa: E402

rng = np.random.default_rng(0)
ref = pd.DataFrame({"x": rng.normal(0, 1, 300), "t": rng.integers(0, 3, 300).astype(float)})


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    return ok


def make_logs(n, *, review=0.0, degraded=0.0, source="real", stub=False, used=10, limit=400, day=None):
    """Synthetic request logs shaped like logbook.load() output (no Gemini involved)."""
    day = day or datetime.now(monitor.BUDGET_TZ).date().isoformat()
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(n):
        rows.append({
            "timestamp": now - timedelta(minutes=n - i), "review": i < review * n, "feedback": pd.NA,
            "source": source, "versions": {"component_a": "stub-risk" if stub else "maintai-gb-sigmoid-v1", "component_b": "gemini|agent-v1"},
            "status": "ok", "llm_calls": 3, "cache_hit": False, "degraded": i < degraded * n, "retries": 0, "rate_limited": False,
            "llm": {"tokens_in": 1000, "tokens_out": 200, "tokens_thinking": 50}, "latency_s": 1.0 + i, "cost_eur": 0.002,
            "llm_budget": {"day": day, "used": used, "limit": limit},
        })
    return pd.DataFrame(rows)


def fired(logs):
    return " | ".join(monitor.alerts(logs, None))


results = [
    check("empty bin in a small window does not explode the PSI", monitor.psi(ref["x"], rng.normal(0, 1, 16)) < 1.5),
    check("same distribution: no alert", not monitor.drift(ref, pd.DataFrame({"x": rng.normal(0, 1, 100), "t": rng.integers(0, 3, 100).astype(float)}))["alert"].any()),
    check("shifted distribution: alert", monitor.drift(ref, pd.DataFrame({"x": rng.normal(3, 1, 20), "t": rng.integers(0, 3, 20).astype(float)}))["alert"].any()),
    check("alerts() silent below MIN_WINDOW", monitor.alerts(pd.DataFrame({"f_x": rng.normal(5, 1, 5), "feedback": [None] * 5}), ref) == []),
    check("reference_features.csv has 9 features and 100+ rows", (lambda r: r is not None and len(r) >= 100 and r.shape[1] == 9)(monitor.load_reference())),
    check("real_traffic drops tests and stub versions", len(monitor.real_traffic(pd.concat([make_logs(5), make_logs(3, source="test"), make_logs(2, stub=True)], ignore_index=True))) == 5),
    check("quota alert above 80% of the budget", "Gemini quota" in fired(make_logs(25, used=330))),
    check("no quota alert at 79%", "Gemini quota" not in fired(make_logs(25, used=316))),
    check("quota alert ignores another day's counter", "Gemini quota" not in fired(make_logs(25, used=399, day="2020-01-01"))),
    check("degraded alert above 20% of the last 50", "Degraded mode" in fired(make_logs(40, degraded=0.3))),
    check("no degraded alert at 5%", "Degraded mode" not in fired(make_logs(40, degraded=0.05))),
    check("no degraded alert with fewer than MIN_DEGRADED requests", "Degraded mode" not in fired(make_logs(5, degraded=1.0))),
    check("degraded stubs are not counted", "Degraded mode" not in fired(pd.concat([make_logs(30, stub=True, degraded=1.0), make_logs(15)], ignore_index=True))),
    check("review alert above 50%", "human review" in fired(make_logs(30, review=0.6))),
    check("no review alert at the campaign mix (30%)", "human review" not in fired(make_logs(30, review=0.3))),
    check("alerts() on empty logs is quiet", monitor.alerts(pd.DataFrame(), None) == []),
    check("operations: p50 and p95 of 20 requests", (lambda t: t["latency p50 / p95 (s), Gemini requests"] == "10.50 / 19.05")(dict(monitor.operations(make_logs(20)).values))),
    check("operations: tokens summed", (lambda t: t["tokens in / out / thinking"] == "20000 / 4000 / 1000")(dict(monitor.operations(make_logs(20)).values))),
    check("operations: quota line", (lambda t: t["Gemini requests today / budget"] == "10 / 400 (2%)")(dict(monitor.operations(make_logs(20)).values))),
    check("quality_counts denominators add up", int(monitor.quality_counts(make_logs(20, review=0.25))["requests"].sum()) == 20 and int(monitor.quality_counts(make_logs(20, review=0.25))["sent to review"].sum()) == 5),
]
print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)

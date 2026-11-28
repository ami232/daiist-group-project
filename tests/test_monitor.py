"""WP5 tests for the drift check (no Gemini, no network). Run: uv run python tests/test_monitor.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor  # noqa: E402

rng = np.random.default_rng(0)
ref = pd.DataFrame({"x": rng.normal(0, 1, 300), "t": rng.integers(0, 3, 300).astype(float)})


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    return ok


results = [
    check("empty bin in a small window does not explode the PSI", monitor.psi(ref["x"], rng.normal(0, 1, 16)) < 1.5),
    check("same distribution: no alert", not monitor.drift(ref, pd.DataFrame({"x": rng.normal(0, 1, 100), "t": rng.integers(0, 3, 100).astype(float)}))["alert"].any()),
    check("shifted distribution: alert", monitor.drift(ref, pd.DataFrame({"x": rng.normal(3, 1, 20), "t": rng.integers(0, 3, 20).astype(float)}))["alert"].any()),
    check("alerts() silent below MIN_WINDOW", monitor.alerts(pd.DataFrame({"f_x": rng.normal(5, 1, 5), "feedback": [None] * 5}), ref) == []),
    check("reference_features.csv has 9 features and 100+ rows", (lambda r: r is not None and len(r) >= 100 and r.shape[1] == 9)(monitor.load_reference())),
]
print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)

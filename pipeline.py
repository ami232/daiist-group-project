"""
Your product's logic: the two components and how one feeds the other.

Everything here is a PLACEHOLDER so the template runs end to end without any
key: component A is a keyword "classifier" and component B a canned "writer".
Replace both with your real components (or import them from your own modules)
but keep what `run()` returns, because app.py, evaluate.py, main.py and
check.py all read it:

    run(raw_input) -> {
        "input_summary": str,          # short, no personal data: it is logged
        "features": {name: number},    # what the drift check watches
        "outputs": {component: ...},   # each component's output, JSON-serialisable
        "versions": {component: str},  # model and prompt versions
        "review": bool,                # does this case go to a person?
        "tokens": int,                 # tokens spent on this request (0 if none)
        "cost_eur": float,             # cost of this request in euros
    }

`raw_input` is what the App tab's input gives you: text, or a file path if you
switch the input to gr.Image / gr.File with type="filepath". The `input` column
of eval/cases.csv uses the same convention.
"""

from pathlib import Path

VERSIONS = {
    "component_a": "placeholder-keywords-v0",
    "component_b": "placeholder-template-v0",
}

# Placeholder thresholds. Yours must be justified in REPORT.md.
REVIEW_BELOW_CONFIDENCE = 0.6

_NEGATIVE = {"broken", "late", "missing", "wrong", "refund", "bad", "problem", "error"}
_POSITIVE = {"thanks", "great", "perfect", "fine", "good", "works"}


def read_input(raw_input: str) -> str:
    """Text as it is; for a file path, the file's text. Replace for images."""
    path = Path(str(raw_input))
    if len(str(raw_input)) < 300 and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return str(raw_input)


def input_features(text: str) -> dict[str, float]:
    """Numbers that describe the input, for the drift check (lengths, brightness, language...)."""
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    return {
        "n_words": float(len(words)),
        "share_uppercase": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
    }


def component_a(text: str) -> dict:
    """PLACEHOLDER first component: label the input and say how sure it is."""
    words = {w.strip(".,!?;:").lower() for w in text.split()}
    negative, positive = len(words & _NEGATIVE), len(words & _POSITIVE)
    if negative == positive:
        return {"label": "unclear", "confidence": 0.5}
    label = "complaint" if negative > positive else "no_issue"
    confidence = 0.5 + 0.5 * abs(negative - positive) / (negative + positive)
    return {"label": label, "confidence": round(confidence, 3)}


def component_b(text: str, a: dict) -> dict:
    """PLACEHOLDER second component: uses component A's output to write a reply."""
    replies = {
        "complaint": "We are sorry about the problem. We will contact you within 48 hours.",
        "no_issue": "Thank you for your message. No action is needed.",
        "unclear": "Thank you. A member of the team will review your message.",
    }
    return {"reply": replies[a["label"]], "tokens": 0}


def run(raw_input: str) -> dict:
    text = read_input(raw_input)
    a = component_a(text)
    b = component_b(text, a)
    features = input_features(text) | {"confidence": a["confidence"]}
    return {
        "input_summary": text[:120],
        "features": features,
        "outputs": {"component_a": a, "component_b": b},
        "versions": VERSIONS,
        "review": a["confidence"] < REVIEW_BELOW_CONFIDENCE,
        "tokens": b["tokens"],
        "cost_eur": 0.0,
    }

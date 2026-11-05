"""
Environment variables: load .env, and know which variables the app needs.

`.env.example` is the list. Every `NAME=` line that is not commented out is a
variable the app needs, unless its line says "(optional)". check.py and the
app's warning banner both read it from here, so keep it up to date.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
EXAMPLE_FILE = ROOT / ".env.example"


def _assignments(path: Path) -> list[tuple[str, str, str]]:
    """(name, value, comment) for every NAME=value line, skipping comments and blanks."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assignment, _, comment = line.partition("#")
        name, separator, value = assignment.partition("=")
        if separator and name.strip():
            rows.append((name.strip(), value.strip().strip('"').strip("'"), comment.strip()))
    return rows


def load_env() -> None:
    """Put the variables in .env into os.environ. Variables already set win."""
    for name, value, _ in _assignments(ENV_FILE):
        if value and not os.environ.get(name):
            os.environ[name] = value


def declared() -> list[tuple[str, bool]]:
    """(name, optional) for every variable listed in .env.example."""
    return [(name, "(optional)" in comment) for name, _, comment in _assignments(EXAMPLE_FILE)]


def missing(include_optional: bool = False) -> list[str]:
    """The variables listed in .env.example that are not set."""
    return [
        name for name, optional in declared()
        if (include_optional or not optional) and not os.environ.get(name, "").strip()
    ]


if __name__ == "__main__":
    load_env()
    for name, optional in declared():
        state = "set" if os.environ.get(name, "").strip() else "MISSING"
        print(f"{name:<20} {state}{'  (optional)' if optional else ''}")

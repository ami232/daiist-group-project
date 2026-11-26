"""Validate the team's evaluation photos and preview brightness drift without Gemini."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from math import isfinite
from pathlib import Path

import pandas as pd

import check
import monitor
import pipeline
import store

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "eval" / "inputs"
SETS = ("cases", "reference", "shifted")


def read_rows(filename, required):
    with (INPUTS.parent / filename).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("missing columns: " + ", ".join(sorted(missing)))
        rows = []
        for row in reader:
            if None in row or any(row.get(column) is None for column in required):
                raise ValueError(f"line {reader.line_num}: missing or extra CSV fields")
            if any(row.values()):
                rows.append(row)
        return rows


def validate_case(row):
    if not row["id"].strip():
        raise ValueError("case ID is empty")
    kind = row["kind"].strip().lower()
    if kind not in check.KINDS:
        raise ValueError("unknown case kind")
    row["kind"] = kind
    expected = json.loads(row["expected"])
    baseline = json.loads(row["baseline"])
    if not isinstance(expected, dict) or type(expected.get("review")) is not bool:
        raise ValueError("expected.review must be true or false")
    if expected["review"] != (kind in {"review", "out_of_scope"}):
        raise ValueError("expected.review disagrees with the case kind")
    if not isinstance(baseline, dict) or type(baseline.get("correct")) is not bool:
        raise ValueError("baseline.correct must be true or false")
    meters = baseline.get("meters")
    if meters is not None and (type(meters) not in (float, int) or not isfinite(meters) or meters < 0):
        raise ValueError("baseline.meters must be a finite, non-negative number")
    for field in ("vision_ids", "product_ids"):
        if field not in expected:
            continue
        ids = expected[field]
        if not isinstance(ids, list) or any(
            value is not None and (not isinstance(value, str) or value not in store.PRODUCTS)
            for value in ids
        ):
            raise ValueError(field + " must be an array of catalog IDs or null")
    if not expected["review"] and (
        not expected.get("product_ids") or any(value is None for value in expected["product_ids"])
    ):
        raise ValueError("an automatic route needs an ordered list of expected product IDs")
    if "agent_items" in expected:
        items = expected["agent_items"]
        if not isinstance(items, list) or not 1 <= len(items) <= pipeline.MAX_ITEMS or any(
            not isinstance(item, str) or not item.strip() or len(item) > 160 for item in items
        ):
            raise ValueError("agent_items must be a non-empty array of short grocery items")
        if len(items) != len(expected.get("product_ids", [])):
            raise ValueError("agent_items and product_ids must have the same length")


def validate_inputs():
    issues = []
    seen_photos = {}
    case_ids = set()
    sets = {}
    features = {name: [] for name in SETS}
    for name, minimum in zip(SETS, (check.MIN_CASES, check.MIN_REFERENCE, check.MIN_SHIFTED)):
        filename = f"{name}.csv"
        try:
            rows = read_rows(filename, check.CASE_COLUMNS if name == "cases" else ["input"])
        except (OSError, ValueError) as error:
            issues.append(f"{filename}: {error}")
            rows = []
        sets[name] = rows
        if len(rows) < minimum:
            issues.append(f"{filename}: {len(rows)} of at least {minimum} rows supplied")
        for i, row in enumerate(rows, 2):
            location = f"{filename} row {i}"
            try:
                if name == "cases":
                    validate_case(row)
                    case_id = row["id"].strip()
                    if case_id in case_ids:
                        raise ValueError("duplicate case ID")
                    case_ids.add(case_id)
                raw_path = Path(row["input"].strip())
                path = (ROOT / raw_path).resolve()
                if raw_path.is_absolute() or not path.is_relative_to(INPUTS.resolve()) or not path.is_file():
                    raise ValueError("input must point to an existing image below eval/inputs/")
                brightness = pipeline.photo_brightness(path)
                with path.open("rb") as image_file:
                    digest = hashlib.file_digest(image_file, "sha256").hexdigest()
                if digest in seen_photos:
                    raise ValueError(f"identical photo already used in {seen_photos[digest]}")
                seen_photos[digest] = location
                if name != "cases" or row["kind"] == "normal":
                    features[name].append({"mean_photo_brightness": brightness})
            except (ValueError, OSError, TypeError) as error:
                issues.append(f"{location}: {error}")
    counts = Counter((row.get("kind") or "").strip().lower() for row in sets["cases"])
    for kind, minimum in check.KINDS.items():
        if counts[kind] < minimum:
            issues.append(f"cases.csv: {counts[kind]} of at least {minimum} {kind} cases supplied")
    return features, issues


def preview(features):
    reference = pd.DataFrame(features["reference"])
    normal = monitor.drift(reference, pd.DataFrame(features["cases"]))
    shifted = monitor.drift(reference, pd.DataFrame(features["shifted"]))
    for label, result in (("Normal evaluation photos", normal), ("Shifted photos", shifted)):
        print(f"\n{label}\n{result.to_string(index=False)}")
    return not normal["alert"].any() and shifted["alert"].any()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-reference", action="store_true", help="save the verified reference features")
    args = parser.parse_args()
    features, issues = validate_inputs()
    if issues:
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit("Resolve the input issues before running Gemini.")
    if not preview(features):
        raise SystemExit("The current drift rule must stay quiet on normal photos and flag the shifted photos.")
    print("\nOffline drift preview passed. No Gemini requests were made.")
    if args.write_reference:
        pd.DataFrame(features["reference"]).to_csv(monitor.REFERENCE_FILE, index=False)
        print("Wrote monitoring/reference_features.csv.")


if __name__ == "__main__":
    main()

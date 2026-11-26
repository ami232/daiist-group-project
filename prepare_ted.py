"""Measure and build a de-identified, linked TED modeling cohort."""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CALLS = "ID_NOTICE_CN FUTURE_CAN_ID FUTURE_CAN_ID_ESTIMATED YEAR ISO_COUNTRY_CODE CANCELLED ID_TYPE CPV TYPE_OF_CONTRACT LOTS_NUMBER VALUE_EURO DURATION TOP_TYPE CAE_TYPE CRIT_PRICE_WEIGHT CRIT_CODE".split()
AWARDS = "ID_NOTICE_CAN YEAR ISO_COUNTRY_CODE CANCELLED LOTS_NUMBER ID_AWARD INFO_ON_NON_AWARD NUMBER_OFFERS".split()


def chunks(path, columns):
    return pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False, chunksize=100_000)


def select_calls(path):
    pieces = []
    for rows in chunks(path, CALLS):
        scope = rows.ISO_COUNTRY_CODE.eq("ES") & rows.TYPE_OF_CONTRACT.eq("W")
        scope &= rows.CPV.str.startswith("45") & rows.ID_TYPE.eq("2")
        pieces.append(rows.loc[scope])
    return pd.concat(pieces, ignore_index=True)


def select_awards(path, ids):
    pieces = []
    for rows in chunks(path, AWARDS):
        pieces.append(rows.loc[rows.ID_NOTICE_CAN.isin(ids)])
    return pd.concat(pieces, ignore_index=True)


def build(calls_path, awards_path):
    raw = select_calls(calls_path)
    candidates = raw.loc[
        raw.LOTS_NUMBER.eq("1") & ~raw.CANCELLED.eq("1")
        & raw.FUTURE_CAN_ID.ne("") & ~raw.FUTURE_CAN_ID_ESTIMATED.eq("1")
    ].drop_duplicates()
    # A single-lot call must point to exactly one award notice and vice versa.
    candidates = candidates[~candidates.ID_NOTICE_CN.duplicated(False)]
    candidates = candidates[~candidates.FUTURE_CAN_ID.duplicated(False)]
    awards = select_awards(awards_path, set(candidates.FUTURE_CAN_ID))
    awards = awards.loc[
        awards.LOTS_NUMBER.eq("1") & ~awards.CANCELLED.eq("1")
        & awards.ISO_COUNTRY_CODE.eq("ES")
    ].drop_duplicates()
    awards = awards[~awards.ID_NOTICE_CAN.duplicated(False)]
    awards["offers"] = pd.to_numeric(awards.NUMBER_OFFERS, errors="coerce")
    awards = awards.loc[
        awards.offers.ge(1) & (awards.offers % 1).eq(0)
        & awards.INFO_ON_NON_AWARD.isin(["", "N"])
    ]
    joined = candidates.merge(awards, left_on="FUTURE_CAN_ID", right_on="ID_NOTICE_CAN",
                              suffixes=("", "_award"), validate="one_to_one")
    joined = joined.loc[joined.YEAR_award.ge(joined.YEAR)]
    budget = pd.to_numeric(joined.VALUE_EURO, errors="coerce")
    months = pd.to_numeric(joined.DURATION, errors="coerce")
    price_weight = pd.to_numeric(joined.CRIT_PRICE_WEIGHT, errors="coerce")
    dataset = pd.DataFrame({
        "notice_id": joined.ID_NOTICE_CN,
        "award_id": joined.ID_NOTICE_CAN,
        "year": joined.YEAR.astype(int),
        "award_year": joined.YEAR_award.astype(int),
        "cpv": joined.CPV,
        "budget_eur": budget.where(budget.gt(1)),
        "duration_months": months.where(months.gt(0)),
        "price_weight": price_weight.where(price_weight.between(0, 100)),
        "award_basis": joined.CRIT_CODE.replace({"L": "price", "M": "price_quality", "": "unknown"}),
        "procedure": joined.TOP_TYPE.replace("", "unknown"),
        "buyer_type": joined.CAE_TYPE.replace("", "unknown"),
        "offers": joined.offers.astype(int),
    }).sort_values(["year", "notice_id"])
    audit = {
        "spanish_construction_calls": int(raw.ID_NOTICE_CN.nunique()),
        "unambiguous_single_lot_call_links": len(candidates),
        "linked_awards_with_offer_counts": len(dataset),
        "by_call_year": {str(k): int(v) for k, v in dataset.year.value_counts().sort_index().items()},
        "reported_budgets": int(dataset.budget_eur.notna().sum()),
    }
    return dataset, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calls_zip", type=Path)
    parser.add_argument("awards_zip", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    dataset, audit = build(args.calls_zip, args.awards_zip)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output / "ted_construction.csv", index=False)
    (args.output / "ted_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

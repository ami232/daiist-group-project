from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 42

TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def main():
    df = pd.read_csv(DATA_DIR / "ai4i2020.csv")

    print("DATASET OVERVIEW")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Unique IDs: {df['UDI'].nunique()}")
    print(f"Machine failures: {df[TARGET].sum()}")
    print(f"Failure rate: {df[TARGET].mean():.2%}")

    print("\nPRODUCT TYPES")
    print(df["Type"].value_counts())

    print("\nFAILURE MODES")
    print(df[FAILURE_MODES].sum())

    any_failure_mode = df[FAILURE_MODES].any(axis=1)

    print("\nLABEL INCONSISTENCIES")
    print(
        "Failure mode without machine failure:",
        (any_failure_mode & (df[TARGET] == 0)).sum()
    )
    print(
        "Machine failure without failure mode:",
        (~any_failure_mode & (df[TARGET] == 1)).sum()
    )

    train, remaining = train_test_split(
        df,
        test_size=0.4,
        random_state=SEED,
        stratify=df[TARGET]
    )

    validation, test = train_test_split(
        remaining,
        test_size=0.5,
        random_state=SEED,
        stratify=remaining[TARGET]
    )

    assert set(train["UDI"]).isdisjoint(validation["UDI"])
    assert set(train["UDI"]).isdisjoint(test["UDI"])
    assert set(validation["UDI"]).isdisjoint(test["UDI"])
    assert len(train) + len(validation) + len(test) == len(df)

    print("\nDATASET SPLITS")

    for name, split in [
        ("train", train),
        ("validation", validation),
        ("test", test)
    ]:
        split = split.sort_values("UDI")
        split.to_csv(DATA_DIR / f"{name}.csv", index=False)

        print(f"\n{name.upper()}")
        print(f"Rows: {len(split)}")
        print(f"Failures: {split[TARGET].sum()}")
        print(f"Failure rate: {split[TARGET].mean():.2%}")
        print(split[FAILURE_MODES].sum())

    print("\nAll splits saved successfully.")


if __name__ == "__main__":
    main()
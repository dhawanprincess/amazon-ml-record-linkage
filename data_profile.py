import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"

files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv"
]

for filename in files:
    path = TRAIN_DIR / filename

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    total_rows = 0
    missing = None
    countries = set()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=100_000
    ):
        total_rows += len(chunk)

        current_missing = chunk.isna().sum()

        if missing is None:
            missing = current_missing
        else:
            missing += current_missing

        if "country" in chunk.columns:
            countries.update(
                chunk["country"].dropna().astype(str).unique()
            )

    print("Total rows:", total_rows)

    print("\nMissing values:")
    print(missing)

    if countries:
        print("\nCountries:")
        print(sorted(countries))
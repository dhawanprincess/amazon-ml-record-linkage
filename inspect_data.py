import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

train_dir = BASE_DIR / "dataset" / "train"

files = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv"
]

for file in files:
    path = train_dir / file

    print("\n" + "=" * 70)
    print(f"FILE: {file}")
    print("=" * 70)

    df = pd.read_csv(path, sep="\t", nrows=5)

    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.to_string(index=False))

    print("\nData types:")
    print(df.dtypes)
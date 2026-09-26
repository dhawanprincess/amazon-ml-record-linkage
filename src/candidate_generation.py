import pandas as pd
from pathlib import Path
import sys

# Allow importing normalization.py
sys.path.append(str(Path(__file__).resolve().parent))

from normalization import normalize_business_name, normalize_address


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"


def load_sample(filename, nrows=10000):
    """Load only a small sample while testing."""
    path = TRAIN_DIR / filename

    return pd.read_csv(
        path,
        sep="\t",
        nrows=nrows
    )


def prepare_dataframe(df):
    """Add normalized name and address columns."""

    df = df.copy()

    df["name_norm"] = df["business_name"].fillna("").apply(
        normalize_business_name
    )

    df["address_norm"] = df["business_address"].fillna("").apply(
        normalize_address
    )

    return df


def create_blocks(source1, source2):
    """
    Create simple candidate blocks using:
    1. Country
    2. First character of normalized business name
    """

    source1 = source1.copy()
    source2 = source2.copy()

    source1["name_block"] = source1["name_norm"].str[:1]
    source2["name_block"] = source2["name_norm"].str[:1]

    candidates = source1.merge(
        source2,
        on=["country", "name_block"],
        suffixes=("_s1", "_s2")
    )

    return candidates


if __name__ == "__main__":

    print("Loading sample data...")

    s1 = load_sample("train_source1.tsv", 1000)
    s2 = load_sample("train_source2.tsv", 10000)

    print("S1 rows:", len(s1))
    print("S2 rows:", len(s2))

    print("\nNormalizing...")
    s1 = prepare_dataframe(s1)
    s2 = prepare_dataframe(s2)

    print("Creating candidate blocks...")

    candidates = create_blocks(s1, s2)

    print("\nCandidate pairs generated:", len(candidates))

    print("\nSample candidates:")
    print(
        candidates[
            [
                "entity_id_s1",
                "business_name_s1",
                "entity_id_s2",
                "business_name_s2",
                "country"
            ]
        ].head(10).to_string(index=False)
    )
import pandas as pd
from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).resolve().parent))

from normalization import normalize_business_name, normalize_address


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"


def load_sample(filename, nrows):
    path = TRAIN_DIR / filename

    return pd.read_csv(
        path,
        sep="\t",
        nrows=nrows
    )


def prepare_dataframe(df):
    df = df.copy()

    df["name_norm"] = (
        df["business_name"]
        .fillna("")
        .apply(normalize_business_name)
    )

    df["address_norm"] = (
        df["business_address"]
        .fillna("")
        .apply(normalize_address)
    )

    # First meaningful token of business name
    df["name_first_token"] = (
        df["name_norm"]
        .str.split()
        .str[0]
        .fillna("")
    )

    # First 3 characters of normalized name
    df["name_prefix"] = df["name_norm"].str[:3]

    # First useful address token
    df["address_first_token"] = (
        df["address_norm"]
        .str.split()
        .str[0]
        .fillna("")
    )

    return df


def create_candidates(s1, s2):

    candidate_sets = []

    # --------------------------------------------------
    # BLOCK 1: Country + first 3 characters of name
    # --------------------------------------------------

    block1 = s1.merge(
        s2,
        on=["country", "name_prefix"],
        suffixes=("_s1", "_s2")
    )

    candidate_sets.append(
        block1[
            [
                "entity_id_s1",
                "entity_id_s2"
            ]
        ]
    )

    # --------------------------------------------------
    # BLOCK 2: Country + first meaningful name token
    # --------------------------------------------------

    block2 = s1.merge(
        s2,
        on=["country", "name_first_token"],
        suffixes=("_s1", "_s2")
    )

    candidate_sets.append(
        block2[
            [
                "entity_id_s1",
                "entity_id_s2"
            ]
        ]
    )

    # --------------------------------------------------
    # BLOCK 3: Exact normalized business name
    # --------------------------------------------------

    block3 = s1.merge(
        s2,
        on=["country", "name_norm"],
        suffixes=("_s1", "_s2")
    )

    candidate_sets.append(
        block3[
            [
                "entity_id_s1",
                "entity_id_s2"
            ]
        ]
    )

    # --------------------------------------------------
    # Combine all blocks
    # --------------------------------------------------

    candidates = pd.concat(
        candidate_sets,
        ignore_index=True
    )

    candidates = candidates.drop_duplicates()

    return candidates


if __name__ == "__main__":

    print("Loading sample data...")

    s1 = load_sample("train_source1.tsv", 1000)
    s2 = load_sample("train_source2.tsv", 10000)

    print("S1 rows:", len(s1))
    print("S2 rows:", len(s2))

    print("\nPreparing data...")

    s1 = prepare_dataframe(s1)
    s2 = prepare_dataframe(s2)

    print("Generating candidates...")

    candidates = create_candidates(s1, s2)

    print("\nUnique candidate pairs:", len(candidates))

    print("\nSample:")
    print(candidates.head(20).to_string(index=False))
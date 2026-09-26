import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import os
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# SETTINGS
# ============================================================

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"

S1_SAMPLE = 1000
CHUNK_SIZE = 200_000

TOP_K = 20

CACHE_DIR = "output/v4_cache"

os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = value.replace("&", " and ")

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def make_text(name, address):

    name = normalize_text(name)
    address = normalize_text(address)

    return f"{name} {address}".strip()


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("=" * 80)
print("V4 CANDIDATE GENERATION")
print("=" * 80)

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=S1_SAMPLE
)

s1_texts = []

for _, row in s1.iterrows():

    s1_texts.append(
        make_text(
            row["business_name"],
            row["business_address"]
        )
    )

print(
    f"S1 entities: {len(s1):,}"
)


# ============================================================
# LOAD SOURCE CORPUS
# ============================================================

def load_source(path, source_name):

    print(f"\nLoading {source_name}...")

    records = []

    chunk_number = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=CHUNK_SIZE
    ):

        chunk_number += 1

        print(
            f"  Processing chunk {chunk_number} "
            f"({len(chunk):,} rows)"
        )

        for _, row in chunk.iterrows():

            text = make_text(
                row["business_name"],
                row["business_address"]
            )

            records.append(
                (
                    row["entity_id"],
                    text
                )
            )

    print(
        f"{source_name} records loaded: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# BUILD RETRIEVAL CORPUS
# ============================================================

s2_records = load_source(
    S2_PATH,
    "Source 2"
)

s3_records = load_source(
    S3_PATH,
    "Source 3"
)


# ============================================================
# NOTE
# ============================================================

print("\n" + "=" * 80)
print("CORPUS PREPARATION COMPLETE")
print("=" * 80)

print(
    f"S1 records : {len(s1_texts):,}"
)

print(
    f"S2 records : {len(s2_records):,}"
)

print(
    f"S3 records : {len(s3_records):,}"
)

print("\nNext stage will build the retrieval index.")
print("No candidate pairs have been generated yet.")

print("\nDone.")

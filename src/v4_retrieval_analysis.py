import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_S1 = 1000
TOP_K = 20
CHUNK_SIZE = 200_000

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"
GT_PATH = "dataset/train/train_ground_truth.tsv"


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

    value = re.sub(r"\s+", " ", value).strip()

    return value


def make_text(name, address):

    name = normalize_text(name)
    address = normalize_text(address)

    return f"{name} {address}".strip()


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=SAMPLE_S1
)

print(f"S1 sample: {len(s1):,}")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t"
)

sample_ids = set(s1["entity_id"])

gt_sample = gt[
    gt["source1_entity_id"].isin(sample_ids)
].copy()


# ============================================================
# COLLECT TRUE MATCH IDS
# ============================================================

true_s2_ids = set()
true_s3_ids = set()

for _, row in gt_sample.iterrows():

    matched = row["matched_entity_ids"]

    if pd.isna(matched):
        continue

    matched = str(matched).strip()

    if not matched:
        continue

    for entity_id in matched.split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            true_s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            true_s3_ids.add(entity_id)


print(
    f"True S2 IDs in sample : {len(true_s2_ids):,}"
)

print(
    f"True S3 IDs in sample : {len(true_s3_ids):,}"
)


# ============================================================
# LOCATE TRUE RECORDS IN FULL S2/S3
# ============================================================

def load_true_records(path, wanted_ids):

    records = {}

    print(f"\nScanning {path}...")

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=CHUNK_SIZE
    ):

        selected = chunk[
            chunk["entity_id"].isin(wanted_ids)
        ]

        for _, row in selected.iterrows():

            records[row["entity_id"]] = {
                "name": row["business_name"],
                "address": row["business_address"],
                "country": row["country"]
            }

        if len(records) == len(wanted_ids):
            break

    return records


true_s2_records = load_true_records(
    S2_PATH,
    true_s2_ids
)

true_s3_records = load_true_records(
    S3_PATH,
    true_s3_ids
)


print(
    f"S2 records found : {len(true_s2_records):,}"
)

print(
    f"S3 records found : {len(true_s3_records):,}"
)


# ============================================================
# CREATE TRUE MATCH DATASET
# ============================================================

true_records = []

for _, row in gt_sample.iterrows():

    matched = row["matched_entity_ids"]

    if pd.isna(matched):
        continue

    matched = str(matched).strip()

    if not matched:
        continue

    s1_id = row["source1_entity_id"]

    s1_row = s1[
        s1["entity_id"] == s1_id
    ]

    if s1_row.empty:
        continue

    s1_row = s1_row.iloc[0]

    s1_text = make_text(
        s1_row["business_name"],
        s1_row["business_address"]
    )

    for entity_id in matched.split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            record = true_s2_records.get(entity_id)
            source = "S2"

        elif entity_id.startswith("S3-"):
            record = true_s3_records.get(entity_id)
            source = "S3"

        else:
            continue

        if record is None:
            continue

        match_text = make_text(
            record["name"],
            record["address"]
        )

        true_records.append({
            "s1_id": s1_id,
            "match_id": entity_id,
            "source": source,
            "s1_text": s1_text,
            "match_text": match_text
        })


true_df = pd.DataFrame(true_records)


print(
    f"\nTrue pairs available for analysis: "
    f"{len(true_df):,}"
)


# ============================================================
# TF-IDF ANALYSIS
# ============================================================

print("\nBuilding TF-IDF representation...")

all_text = pd.concat(
    [
        true_df["s1_text"],
        true_df["match_text"]
    ],
    ignore_index=True
)


vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=1,
    sublinear_tf=True
)


matrix = vectorizer.fit_transform(all_text)


s1_matrix = matrix[
    :len(true_df)
]

match_matrix = matrix[
    len(true_df):
]


# ============================================================
# NEAREST NEIGHBOR SEARCH
# ============================================================

print("Running nearest-neighbor search...")

nn = NearestNeighbors(
    n_neighbors=min(TOP_K, len(true_df)),
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

nn.fit(match_matrix)

distances, indices = nn.kneighbors(
    s1_matrix
)


# ============================================================
# NOTE
# ============================================================
# This retrieval test compares each S1 against the collection
# of TRUE MATCH RECORDS only.
#
# Therefore this is NOT a real blocking recall score.
# It measures whether character TF-IDF can recognize
# true-match text similarity when the correct records are
# available in the retrieval pool.
# ============================================================


# ============================================================
# TOP-K HIT ANALYSIS
# ============================================================

hit_counts = {
    1: 0,
    5: 0,
    10: 0,
    20: 0
}

evaluated = 0


for i, row in true_df.iterrows():

    target_id = row["match_id"]

    retrieved_ids = [
        true_df.iloc[idx]["match_id"]
        for idx in indices[i]
    ]

    evaluated += 1

    for k in hit_counts:

        if target_id in retrieved_ids[:k]:
            hit_counts[k] += 1


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("V4 CHARACTER TF-IDF RETRIEVAL ANALYSIS")
print("=" * 80)

print(
    f"True pairs evaluated : {evaluated:,}"
)

for k, count in hit_counts.items():

    recall = (
        count / evaluated * 100
        if evaluated
        else 0
    )

    print(
        f"Top-{k:<2} retrieval hit : "
        f"{count:,} "
        f"({recall:.2f}%)"
    )


# ============================================================
# EXAMPLES
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE RETRIEVAL RESULTS")
print("=" * 80)

for i in range(min(5, len(true_df))):

    row = true_df.iloc[i]

    print("\nS1:")
    print(row["s1_text"])

    print("\nTRUE MATCH:")
    print(row["match_id"])
    print(row["match_text"])

    print("\nTOP RETRIEVED:")

    for rank, idx in enumerate(
        indices[i][:5],
        start=1
    ):

        retrieved = true_df.iloc[idx]

        print(
            f"{rank}. "
            f"{retrieved['match_id']} | "
            f"{retrieved['match_text'][:150]}"
        )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 80)
print("V4 analysis complete.")
print("=" * 80)
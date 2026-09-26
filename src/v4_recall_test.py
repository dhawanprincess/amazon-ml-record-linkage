import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# LAPTOP-SAFE SETTINGS
# ============================================================

S1_ROWS = 100
TARGET_ROWS = 1000
TOP_K = 10

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

    value = str(value).lower()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(r"\s+", " ", value).strip()

    return value


def make_text(name, address):

    return (
        normalize_text(name)
        + " "
        + normalize_text(address)
    ).strip()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("V4 TOP-K RECALL TEST")
print("=" * 70)

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=S1_ROWS
)

print(f"S1: {len(s1):,}")


print("\nLoading Ground Truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t"
)

# Only the first 100 S1 entities
gt = gt[
    gt["source1_entity_id"].isin(
        s1["entity_id"]
    )
].copy()

print(f"Ground-truth rows: {len(gt):,}")


# ============================================================
# BUILD TRUE MATCH DICTIONARY
# ============================================================

true_matches = {}

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]
    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        true_matches[s1_id] = set()
        continue

    ids = {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }

    true_matches[s1_id] = ids


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def test_source(
    source1_df,
    source_path,
    source_name
):

    print("\n" + "-" * 70)
    print(f"Testing {source_name}")
    print("-" * 70)

    target = pd.read_csv(
        source_path,
        sep="\t",
        nrows=TARGET_ROWS
    )

    print(
        f"{source_name} rows loaded: "
        f"{len(target):,}"
    )

    target["match_text"] = target.apply(
        lambda x: make_text(
            x["business_name"],
            x["business_address"]
        ),
        axis=1
    )

    source1_df = source1_df.copy()

    source1_df["match_text"] = source1_df.apply(
        lambda x: make_text(
            x["business_name"],
            x["business_address"]
        ),
        axis=1
    )

    print("Building character TF-IDF...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        sublinear_tf=True
    )

    target_matrix = vectorizer.fit_transform(
        target["match_text"]
    )

    query_matrix = vectorizer.transform(
        source1_df["match_text"]
    )

    nn = NearestNeighbors(
        n_neighbors=min(TOP_K, len(target)),
        metric="cosine",
        algorithm="brute"
    )

    nn.fit(target_matrix)

    distances, indices = nn.kneighbors(
        query_matrix
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    total_true = 0
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0

    evaluated_entities = 0

    for i, (_, row) in enumerate(
        source1_df.iterrows()
    ):

        s1_id = row["entity_id"]

        true_ids = {
            x for x in true_matches.get(
                s1_id,
                set()
            )
            if x.startswith(
                "S2-" if source_name == "Source 2"
                else "S3-"
            )
        }

        if not true_ids:
            continue

        evaluated_entities += 1
        total_true += len(true_ids)

        retrieved_ids = [
            target.iloc[idx]["entity_id"]
            for idx in indices[i]
        ]

        if any(
            x in true_ids
            for x in retrieved_ids[:1]
        ):
            top1_hits += 1

        if any(
            x in true_ids
            for x in retrieved_ids[:5]
        ):
            top5_hits += 1

        if any(
            x in true_ids
            for x in retrieved_ids[:10]
        ):
            top10_hits += 1

    print("\nRESULTS")

    print(
        f"Entities with true {source_name} matches: "
        f"{evaluated_entities}"
    )

    print(
        f"Total true matches: {total_true}"
    )

    if evaluated_entities > 0:

        print(
            f"Top-1 entity hit rate : "
            f"{top1_hits / evaluated_entities * 100:.2f}%"
        )

        print(
            f"Top-5 entity hit rate : "
            f"{top5_hits / evaluated_entities * 100:.2f}%"
        )

        print(
            f"Top-10 entity hit rate: "
            f"{top10_hits / evaluated_entities * 100:.2f}%"
        )

    else:
        print("No true matches found in this sample.")

    return


# ============================================================
# TEST S2
# ============================================================

test_source(
    s1,
    S2_PATH,
    "Source 2"
)


# ============================================================
# TEST S3
# ============================================================

test_source(
    s1,
    S3_PATH,
    "Source 3"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("RECALL TEST COMPLETE")
print("=" * 70)
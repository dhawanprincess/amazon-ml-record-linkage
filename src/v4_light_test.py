import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# SETTINGS — LAPTOP SAFE
# ============================================================

S1_ROWS = 100
S2_ROWS = 1000
S3_ROWS = 1000
TOP_K = 10

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"


# ============================================================
# TEXT NORMALIZATION
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

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def make_text(name, address):

    return (
        normalize_text(name)
        + " "
        + normalize_text(address)
    ).strip()


# ============================================================
# LOAD SMALL SAMPLE
# ============================================================

print("=" * 70)
print("LIGHTWEIGHT V4 RETRIEVAL TEST")
print("=" * 70)

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=S1_ROWS
)

print(f"S1 loaded: {len(s1):,}")


print("\nLoading Source 2...")

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    nrows=S2_ROWS
)

print(f"S2 loaded: {len(s2):,}")


print("\nLoading Source 3...")

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    nrows=S3_ROWS
)

print(f"S3 loaded: {len(s3):,}")


# ============================================================
# CREATE TEXT
# ============================================================

print("\nCreating text representations...")

s1["match_text"] = s1.apply(
    lambda x: make_text(
        x["business_name"],
        x["business_address"]
    ),
    axis=1
)

s2["match_text"] = s2.apply(
    lambda x: make_text(
        x["business_name"],
        x["business_address"]
    ),
    axis=1
)

s3["match_text"] = s3.apply(
    lambda x: make_text(
        x["business_name"],
        x["business_address"]
    ),
    axis=1
)


# ============================================================
# FUNCTION FOR RETRIEVAL
# ============================================================

def retrieve_candidates(
    source1_df,
    target_df,
    source_name
):

    print(
        f"\nBuilding TF-IDF index for {source_name}..."
    )

    corpus = target_df["match_text"].tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True
    )

    target_matrix = vectorizer.fit_transform(
        corpus
    )

    query_matrix = vectorizer.transform(
        source1_df["match_text"]
    )

    nn = NearestNeighbors(
        n_neighbors=min(
            TOP_K,
            len(target_df)
        ),
        metric="cosine",
        algorithm="brute"
    )

    nn.fit(target_matrix)

    distances, indices = nn.kneighbors(
        query_matrix
    )

    candidate_pairs = set()

    for i, source1_row in source1_df.iterrows():

        for target_index in indices[i]:

            target_id = target_df.iloc[
                target_index
            ]["entity_id"]

            candidate_pairs.add(
                (
                    source1_row["entity_id"],
                    target_id
                )
            )

    print(
        f"{source_name} candidate pairs: "
        f"{len(candidate_pairs):,}"
    )

    return candidate_pairs


# ============================================================
# RETRIEVE S2
# ============================================================

s2_candidates = retrieve_candidates(
    s1,
    s2,
    "Source 2"
)


# ============================================================
# RETRIEVE S3
# ============================================================

s3_candidates = retrieve_candidates(
    s1,
    s3,
    "Source 3"
)


# ============================================================
# COMBINE
# ============================================================

all_candidates = (
    s2_candidates |
    s3_candidates
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"S1 entities       : {len(s1):,}"
)

print(
    f"S2 entities       : {len(s2):,}"
)

print(
    f"S3 entities       : {len(s3):,}"
)

print(
    f"S2 candidates     : {len(s2_candidates):,}"
)

print(
    f"S3 candidates     : {len(s3_candidates):,}"
)

print(
    f"Total candidates  : {len(all_candidates):,}"
)


# ============================================================
# SAMPLE
# ============================================================

print("\nSample candidates:")

for s1_id, match_id in list(
    all_candidates
)[:20]:

    print(
        f"{s1_id} -> {match_id}"
    )


print("\n" + "=" * 70)
print("Lightweight test complete.")
print("=" * 70)
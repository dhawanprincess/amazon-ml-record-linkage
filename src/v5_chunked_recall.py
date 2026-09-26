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

S1_ROWS = 100

CHUNK_SIZE = 50_000

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
# LOAD S1
# ============================================================

print("=" * 70)
print("V5 CHUNKED FULL-SOURCE RETRIEVAL")
print("=" * 70)

print("\nLoading first 100 Source 1 entities...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=S1_ROWS
)

s1["match_text"] = [
    make_text(n, a)
    for n, a in zip(
        s1["business_name"],
        s1["business_address"]
    )
]

print(f"S1 loaded: {len(s1):,}")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t"
)

gt = gt[
    gt["source1_entity_id"].isin(
        s1["entity_id"]
    )
].copy()


# ============================================================
# TRUE MATCH DICTIONARY
# ============================================================

true_matches = {}

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        true_matches[s1_id] = set()
    else:
        true_matches[s1_id] = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }


# ============================================================
# CHUNKED RETRIEVAL
# ============================================================

def retrieve_full_source(
    source_path,
    source_prefix,
    source_name
):

    print("\n" + "=" * 70)
    print(f"PROCESSING {source_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    # Fit TF-IDF vocabulary ONLY on S1 queries.
    # This keeps memory much smaller.
    # --------------------------------------------------------

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        sublinear_tf=True,
        max_features=100_000
    )

    # Fit using S1 only
    vectorizer.fit(
        s1["match_text"]
    )

    query_matrix = vectorizer.transform(
        s1["match_text"]
    )

    # --------------------------------------------------------
    # Store best candidates across chunks
    # --------------------------------------------------------

    best_candidates = [
        []
        for _ in range(len(s1))
    ]

    processed_rows = 0
    chunk_number = 0

    reader = pd.read_csv(
        source_path,
        sep="\t",
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        chunk_number += 1
        processed_rows += len(chunk)

        print(
            f"\rProcessing {source_name}: "
            f"{processed_rows:,} rows",
            end=""
        )

        # -----------------------------------------------
        # Create text
        # -----------------------------------------------

        texts = [
            make_text(n, a)
            for n, a in zip(
                chunk["business_name"],
                chunk["business_address"]
            )
        ]

        # -----------------------------------------------
        # TF-IDF
        # -----------------------------------------------

        chunk_matrix = vectorizer.transform(
            texts
        )

        # -----------------------------------------------
        # Similarity
        #
        # cosine similarity = 1 - cosine distance
        # -----------------------------------------------

        similarities = query_matrix @ chunk_matrix.T

        similarities = similarities.toarray()

        # -----------------------------------------------
        # Keep only TOP_K from this chunk
        # -----------------------------------------------

        for i in range(len(s1)):

            row_scores = similarities[i]

            k = min(
                TOP_K,
                len(row_scores)
            )

            if k == 0:
                continue

            top_indices = np.argpartition(
                row_scores,
                -k
            )[-k:]

            for idx in top_indices:

                score = row_scores[idx]

                entity_id = chunk.iloc[
                    idx
                ]["entity_id"]

                best_candidates[i].append(
                    (
                        float(score),
                        entity_id
                    )
                )

        # -----------------------------------------------
        # Keep only global TOP_K so far
        # -----------------------------------------------

        for i in range(len(s1)):

            best_candidates[i].sort(
                reverse=True
            )

            best_candidates[i] = (
                best_candidates[i][:TOP_K]
            )

        # -----------------------------------------------
        # Free chunk memory
        # -----------------------------------------------

        del chunk_matrix
        del similarities
        del texts

    print("\n")

    # ========================================================
    # EVALUATION
    # ========================================================

    evaluated = 0

    top1_hits = 0
    top5_hits = 0
    top10_hits = 0

    total_true = 0
    retrieved_true = 0

    for i, row in s1.iterrows():

        s1_id = row["entity_id"]

        true_ids = {
            x
            for x in true_matches.get(
                s1_id,
                set()
            )
            if x.startswith(source_prefix)
        }

        if not true_ids:
            continue

        evaluated += 1
        total_true += len(true_ids)

        retrieved_ids = [
            entity_id
            for score, entity_id
            in best_candidates[i]
        ]

        # Entity-level hit
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

        # Pair-level recall
        retrieved_true += len(
            set(retrieved_ids) & true_ids
        )

    print("-" * 70)
    print(f"{source_name} RESULTS")
    print("-" * 70)

    print(
        f"Entities with true matches : {evaluated}"
    )

    print(
        f"Total true matches         : {total_true}"
    )

    print(
        f"True matches retrieved     : {retrieved_true}"
    )

    if total_true > 0:

        print(
            f"Pair Top-10 recall         : "
            f"{retrieved_true / total_true * 100:.2f}%"
        )

    if evaluated > 0:

        print(
            f"Entity Top-1 hit rate      : "
            f"{top1_hits / evaluated * 100:.2f}%"
        )

        print(
            f"Entity Top-5 hit rate      : "
            f"{top5_hits / evaluated * 100:.2f}%"
        )

        print(
            f"Entity Top-10 hit rate     : "
            f"{top10_hits / evaluated * 100:.2f}%"
        )

    # ========================================================
    # SHOW EXAMPLES
    # ========================================================

    print("\nSample retrieved results:")

    for i in range(
        min(5, len(s1))
    ):

        print(
            "\nS1:",
            s1.iloc[i]["entity_id"]
        )

        print(
            "Name:",
            s1.iloc[i]["business_name"]
        )

        print(
            "Top candidates:"
        )

        for score, entity_id in (
            best_candidates[i][:5]
        ):

            print(
                f"  {entity_id} "
                f"(score={score:.4f})"
            )


# ============================================================
# SOURCE 2
# ============================================================

retrieve_full_source(
    S2_PATH,
    "S2-",
    "Source 2"
)


# ============================================================
# SOURCE 3
# ============================================================

retrieve_full_source(
    S3_PATH,
    "S3-",
    "Source 3"
)


print("\n" + "=" * 70)
print("V5 TEST COMPLETE")
print("=" * 70)
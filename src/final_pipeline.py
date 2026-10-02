# src/final_pipeline.py

import os
import re
import sys
import pandas as pd
from rapidfuzz.fuzz import ratio

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# CONFIG — SAFE DRY RUN
# ============================================================

TEST_S1 = "dataset/test/test_source1.tsv"
TEST_S2 = "dataset/test/test_source2.tsv"
TEST_S3 = "dataset/test/test_source3.tsv"

S1_LIMIT = 100
TARGET_LIMIT = 5000

TOP_K = 5
MATCH_THRESHOLD = 72

CHUNK_SIZE = 1000

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def norm(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    x = re.sub(r"\s+", " ", x).strip()

    return x


def get_tokens(text):
    return {
        t for t in text.split()
        if len(t) >= 3
    }


def get_address_number(address):
    if not address:
        return ""

    m = re.search(r"\b\d+[a-zA-Z]?\b", address)

    return m.group(0).lower() if m else ""


# ============================================================
# LOAD S1 — ONLY 100
# ============================================================

print("=" * 70)
print("SAFE LIGHTWEIGHT ENTITY RESOLUTION TEST")
print("=" * 70)

print("\nLoading TEST Source 1...")

s1 = pd.read_csv(
    TEST_S1,
    sep="\t",
    nrows=S1_LIMIT,
    dtype=str
).fillna("")

print(f"S1 rows loaded: {len(s1)}")


# ============================================================
# PREPARE S1
# ============================================================

print("\nPreparing S1 records...")

s1_records = {}

for _, row in s1.iterrows():

    sid = row["entity_id"]

    name = norm(row["business_name"])
    address = norm(row["business_address"])
    country = norm(row["country"])

    s1_records[sid] = {
        "entity_id": sid,
        "business_name": name,
        "business_address": address,
        "country": country,
        "name_tokens": get_tokens(name),
        "address_number": get_address_number(address)
    }


print(f"S1 records prepared: {len(s1_records)}")


# ============================================================
# BUILD SMALL INDEX
# ============================================================

print("\nBuilding lightweight S1 index...")

token_index = {}
address_index = {}
exact_name_index = {}

for sid, rec in s1_records.items():

    # Name tokens
    for token in rec["name_tokens"]:

        token_index.setdefault(token, set()).add(sid)

    # Address number
    number = rec["address_number"]

    if number:
        address_index.setdefault(
            (rec["country"], number),
            set()
        ).add(sid)

    # Exact name
    if rec["business_name"]:

        exact_name_index.setdefault(
            (rec["country"], rec["business_name"]),
            set()
        ).add(sid)


print(f"Name index tokens : {len(token_index)}")
print(f"Address index     : {len(address_index)}")
print(f"Exact name index  : {len(exact_name_index)}")


# ============================================================
# SCORE
# ============================================================

def score_pair(s1rec, target):

    name_score = ratio(
        s1rec["business_name"],
        target["business_name"]
    )

    address_score = ratio(
        s1rec["business_address"],
        target["business_address"]
    )

    score = (
        name_score * 0.55 +
        address_score * 0.45
    )

    # Exact country bonus
    if (
        s1rec["country"]
        and target["country"]
        and s1rec["country"] == target["country"]
    ):
        score += 5

    # Same address number bonus
    if (
        s1rec["address_number"]
        and target["address_number"]
        and
        s1rec["address_number"]
        ==
        target["address_number"]
    ):
        score += 8

    return min(score, 100)


# ============================================================
# PROCESS TARGET SOURCE
# ============================================================

def process_source(path, source_name):

    print("\n" + "-" * 70)
    print(f"PROCESSING {source_name}")
    print("-" * 70)

    target = pd.read_csv(
        path,
        sep="\t",
        nrows=TARGET_LIMIT,
        dtype=str
    ).fillna("")

    print(
        f"{source_name} rows loaded: {len(target)}"
    )

    results = {}
    candidate_rows = []

    total = len(target)

    for i, (_, row) in enumerate(target.iterrows(), start=1):

        if i % 500 == 0 or i == 1:
            print(
                f"{source_name}: processed "
                f"{i}/{total}"
            )

        tid = row["entity_id"]

        target_name = norm(row["business_name"])
        target_address = norm(row["business_address"])
        target_country = norm(row["country"])

        target_tokens = get_tokens(target_name)

        target_number = get_address_number(
            target_address
        )

        # ----------------------------------------------------
        # Candidate generation
        # ----------------------------------------------------

        candidates = set()

        # Exact name
        if target_name:

            candidates.update(
                exact_name_index.get(
                    (target_country, target_name),
                    set()
                )
            )

        # Address number
        if target_number:

            candidates.update(
                address_index.get(
                    (target_country, target_number),
                    set()
                )
            )

        # Name tokens
        for token in target_tokens:

            ids = token_index.get(token)

            if ids:
                candidates.update(ids)

        # ----------------------------------------------------
        # HARD LIMIT
        # ----------------------------------------------------

        if len(candidates) > 30:

            # Prefer candidates sharing address number
            preferred = set()

            if target_number:

                preferred.update(
                    address_index.get(
                        (target_country, target_number),
                        set()
                    )
                )

            # Then exact-name candidates
            if target_name:

                preferred.update(
                    exact_name_index.get(
                        (target_country, target_name),
                        set()
                    )
                )

            if preferred:

                candidates = preferred

            else:

                candidates = set(
                    list(candidates)[:30]
                )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        scored = []

        for sid in candidates:

            s1rec = s1_records[sid]

            target_rec = {
                "business_name": target_name,
                "business_address": target_address,
                "country": target_country,
                "address_number": target_number
            }

            score = score_pair(
                s1rec,
                target_rec
            )

            scored.append(
                (score, sid)
            )

        scored.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        top = scored[:TOP_K]

        # ----------------------------------------------------
        # Save candidates
        # ----------------------------------------------------

        for score, sid in top:

            candidate_rows.append({
                "source1_entity_id": sid,
                "source_entity_id": tid,
                "source": source_name,
                "score": round(score, 2)
            })

        # ----------------------------------------------------
        # Matches
        # ----------------------------------------------------

        matches = [
            sid
            for score, sid in top
            if score >= MATCH_THRESHOLD
        ]

        results[tid] = matches

    print(
        f"\n{source_name} processing complete."
    )

    print(
        f"{source_name} candidate rows: "
        f"{len(candidate_rows)}"
    )

    return results, candidate_rows


# ============================================================
# RUN
# ============================================================

s2_results, s2_candidates = process_source(
    TEST_S2,
    "S2"
)

s3_results, s3_candidates = process_source(
    TEST_S3,
    "S3"
)


# ============================================================
# COMBINE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CREATING OUTPUT")
print("=" * 70)

all_results = {}

for sid in s1_records:

    matches = []

    # This dry run only has target-side results for
    # the first 5K rows.
    # We therefore keep the S1 entity in the output.

    all_results[sid] = matches


# Add S2/S3 matches
for tid, matches in s2_results.items():

    for sid in matches:

        if sid in all_results:
            all_results[sid].append(tid)


for tid, matches in s3_results.items():

    for sid in matches:

        if sid in all_results:
            all_results[sid].append(tid)


# ============================================================
# MATCHING RESULTS
# ============================================================

matching_rows = []

for sid in s1_records:

    matches = list(
        dict.fromkeys(
            all_results[sid]
        )
    )

    matching_rows.append({
        "source1_entity_id": sid,
        "matched_entity_ids": ",".join(matches)
    })


matching_df = pd.DataFrame(
    matching_rows
)

matching_path = os.path.join(
    OUTPUT_DIR,
    "matching_results.tsv"
)

matching_df.to_csv(
    matching_path,
    sep="\t",
    index=False
)


# ============================================================
# CANDIDATE OUTPUT
# ============================================================

candidate_df = pd.DataFrame(
    s2_candidates + s3_candidates
)

candidate_path = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

candidate_df.to_csv(
    candidate_path,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DRY RUN COMPLETE")
print("=" * 70)

print(
    f"S1 entities       : {len(matching_df)}"
)

print(
    f"Candidate pairs   : {len(candidate_df)}"
)

print(
    f"Matching file     : {matching_path}"
)

print(
    f"Candidate file    : {candidate_path}"
)

print("\nDO NOT submit these files yet.")
print("This is only the safe 100 × 5K dry run.")
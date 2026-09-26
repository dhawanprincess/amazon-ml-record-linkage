import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

S1_ROWS = 100
SOURCE_ROWS = 250_000
CHUNK_SIZE = 25_000

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

    return re.sub(r"\s+", " ", value).strip()


def first_token(value):

    text = normalize_text(value)

    return text.split()[0] if text else ""


def address_number(value):

    if pd.isna(value):
        return ""

    match = re.search(
        r"\b\d{1,6}\b",
        str(value)
    )

    return match.group() if match else ""


# ============================================================
# LOAD S1
# ============================================================

print("=" * 70)
print("V6 BLOCKING RECALL TEST")
print("=" * 70)

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=S1_ROWS
)

s1["country_norm"] = (
    s1["country"]
    .fillna("")
    .astype(str)
    .str.lower()
)

s1["norm_name"] = s1["business_name"].map(
    normalize_text
)

s1["name_token"] = s1["business_name"].map(
    first_token
)

s1["address_number"] = s1["business_address"].map(
    address_number
)


# ============================================================
# BLOCK LOOKUPS
# ============================================================

address_blocks = {}
token_blocks = {}
prefix_blocks = {}

for _, row in s1.iterrows():

    s1_id = row["entity_id"]
    country = row["country_norm"]

    number = row["address_number"]

    if number:
        key = country + "|" + number

        address_blocks.setdefault(
            key, set()
        ).add(s1_id)

    token = row["name_token"]

    if token:
        key = country + "|" + token

        token_blocks.setdefault(
            key, set()
        ).add(s1_id)

    prefix = row["norm_name"][:3]

    if prefix:
        key = country + "|" + prefix

        prefix_blocks.setdefault(
            key, set()
        ).add(s1_id)


# ============================================================
# GROUND TRUTH
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
]

true_matches = {}

for _, row in gt.iterrows():

    value = row["matched_entity_ids"]

    if pd.isna(value) or not str(value).strip():

        true_matches[
            row["source1_entity_id"]
        ] = set()

    else:

        true_matches[
            row["source1_entity_id"]
        ] = {
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        }


# ============================================================
# PROCESS SOURCE
# ============================================================

def test_source(
    path,
    prefix,
    source_name
):

    print("\n" + "-" * 70)
    print(f"Testing {source_name}")
    print("-" * 70)

    surviving_pairs = set()
    rows_processed = 0

    reader = pd.read_csv(
        path,
        sep="\t",
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        remaining = SOURCE_ROWS - rows_processed

        if remaining <= 0:
            break

        if len(chunk) > remaining:
            chunk = chunk.iloc[:remaining]

        rows_processed += len(chunk)

        for _, row in chunk.iterrows():

            entity_id = row["entity_id"]

            country = (
                ""
                if pd.isna(row["country"])
                else str(row["country"]).lower()
            )

            name = normalize_text(
                row["business_name"]
            )

            token = first_token(
                row["business_name"]
            )

            number = address_number(
                row["business_address"]
            )

            # --------------------------------------------
            # BLOCK 1
            # --------------------------------------------

            if number:

                key = country + "|" + number

                for s1_id in address_blocks.get(
                    key, ()
                ):

                    surviving_pairs.add(
                        (s1_id, entity_id)
                    )

            # --------------------------------------------
            # BLOCK 2
            # --------------------------------------------

            if token:

                key = country + "|" + token

                for s1_id in token_blocks.get(
                    key, ()
                ):

                    surviving_pairs.add(
                        (s1_id, entity_id)
                    )

            # --------------------------------------------
            # BLOCK 3
            # --------------------------------------------

            prefix3 = name[:3]

            if prefix3:

                key = country + "|" + prefix3

                for s1_id in prefix_blocks.get(
                    key, ()
                ):

                    surviving_pairs.add(
                        (s1_id, entity_id)
                    )

        print(
            f"\rProcessed: {rows_processed:,} | "
            f"Candidates: {len(surviving_pairs):,}",
            end=""
        )

    print("\n")

    # ========================================================
    # RECALL
    # ========================================================

    total_true = 0
    survived_true = 0

    entities_with_true = 0
    entities_hit = 0

    for s1_id, true_ids in true_matches.items():

        source_true = {
            x for x in true_ids
            if x.startswith(prefix)
        }

        if not source_true:
            continue

        entities_with_true += 1

        survived_for_entity = (
            source_true
            &
            {
                target_id
                for source_id, target_id
                in surviving_pairs
                if source_id == s1_id
            }
        )

        total_true += len(source_true)
        survived_true += len(
            survived_for_entity
        )

        if survived_for_entity:
            entities_hit += 1

    print("=" * 70)
    print(f"{source_name} RECALL")
    print("=" * 70)

    print(
        f"True matches             : {total_true:,}"
    )

    print(
        f"True matches survived    : "
        f"{survived_true:,}"
    )

    print(
        f"Entities with true match : "
        f"{entities_with_true:,}"
    )

    print(
        f"Entities with ≥1 hit     : "
        f"{entities_hit:,}"
    )

    if total_true:

        print(
            f"PAIR BLOCKING RECALL     : "
            f"{survived_true / total_true * 100:.2f}%"
        )

    if entities_with_true:

        print(
            f"ENTITY BLOCKING RECALL   : "
            f"{entities_hit / entities_with_true * 100:.2f}%"
        )

    print(
        f"Candidate pairs          : "
        f"{len(surviving_pairs):,}"
    )


# ============================================================
# TEST S2
# ============================================================

test_source(
    S2_PATH,
    "S2-",
    "Source 2"
)


# ============================================================
# TEST S3
# ============================================================

test_source(
    S3_PATH,
    "S3-",
    "Source 3"
)


print("\n" + "=" * 70)
print("BLOCKING RECALL TEST COMPLETE")
print("=" * 70)
import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import pandas as pd


# ============================================================
# SETTINGS — VERY LIGHTWEIGHT
# ============================================================

S1_ROWS = 100

SOURCE_ROWS = 250_000
CHUNK_SIZE = 25_000

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"


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


def first_token(text):

    text = normalize_text(text)

    if not text:
        return ""

    return text.split()[0]


def address_number(address):

    if pd.isna(address):
        return ""

    match = re.search(
        r"\b\d{1,6}\b",
        str(address)
    )

    if match:
        return match.group()

    return ""


# ============================================================
# PREPARE S1
# ============================================================

print("=" * 70)
print("V6 SAFE BLOCKING TEST")
print("=" * 70)

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=S1_ROWS
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

print(
    f"S1 loaded: {len(s1):,}"
)


# ============================================================
# CREATE BLOCK KEYS
# ============================================================

# country + address number
s1["block_address"] = (
    s1["country"].fillna("").astype(str).str.lower()
    + "|"
    + s1["address_number"]
)

# country + first name token
s1["block_token"] = (
    s1["country"].fillna("").astype(str).str.lower()
    + "|"
    + s1["name_token"]
)

# country + first 3 characters of name
s1["block_prefix"] = (
    s1["country"].fillna("").astype(str).str.lower()
    + "|"
    + s1["norm_name"].str[:3]
)


# ============================================================
# CREATE LOOKUP DICTIONARIES
# ============================================================

address_blocks = {}
token_blocks = {}
prefix_blocks = {}

for _, row in s1.iterrows():

    idx = row["entity_id"]

    if row["block_address"].split("|")[-1]:
        address_blocks.setdefault(
            row["block_address"],
            set()
        ).add(idx)

    if row["block_token"].split("|")[-1]:
        token_blocks.setdefault(
            row["block_token"],
            set()
        ).add(idx)

    if row["block_prefix"].split("|")[-1]:
        prefix_blocks.setdefault(
            row["block_prefix"],
            set()
        ).add(idx)


# ============================================================
# PROCESS SOURCE
# ============================================================

def process_source(
    path,
    source_name
):

    print("\n" + "-" * 70)
    print(f"Processing {source_name}")
    print("-" * 70)

    candidates = set()

    rows_processed = 0

    reader = pd.read_csv(
        path,
        sep="\t",
        chunksize=CHUNK_SIZE
    )

    for chunk in reader:

        rows_processed += len(chunk)

        # Stop after SOURCE_ROWS
        if rows_processed > SOURCE_ROWS:

            extra = rows_processed - SOURCE_ROWS

            chunk = chunk.iloc[
                :-extra
            ]

            rows_processed = SOURCE_ROWS

        # ----------------------------------------------------
        # Normalize only necessary fields
        # ----------------------------------------------------

        chunk["norm_name"] = chunk[
            "business_name"
        ].map(normalize_text)

        chunk["name_token"] = chunk[
            "business_name"
        ].map(first_token)

        chunk["address_number"] = chunk[
            "business_address"
        ].map(address_number)

        chunk["country_norm"] = (
            chunk["country"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        # ----------------------------------------------------
        # BLOCK 1: COUNTRY + ADDRESS NUMBER
        # ----------------------------------------------------

        for _, row in chunk.iterrows():

            country = row["country_norm"]
            number = row["address_number"]

            if number:

                key = (
                    country
                    + "|"
                    + number
                )

                matches = address_blocks.get(
                    key,
                    set()
                )

                for s1_id in matches:

                    candidates.add(
                        (
                            s1_id,
                            row["entity_id"]
                        )
                    )

        # ----------------------------------------------------
        # BLOCK 2: COUNTRY + FIRST NAME TOKEN
        # ----------------------------------------------------

        for _, row in chunk.iterrows():

            token = row["name_token"]

            if token:

                key = (
                    row["country_norm"]
                    + "|"
                    + token
                )

                matches = token_blocks.get(
                    key,
                    set()
                )

                for s1_id in matches:

                    candidates.add(
                        (
                            s1_id,
                            row["entity_id"]
                        )
                    )

        # ----------------------------------------------------
        # BLOCK 3: COUNTRY + NAME PREFIX
        # ----------------------------------------------------

        for _, row in chunk.iterrows():

            prefix = row["norm_name"][:3]

            if prefix:

                key = (
                    row["country_norm"]
                    + "|"
                    + prefix
                )

                matches = prefix_blocks.get(
                    key,
                    set()
                )

                for s1_id in matches:

                    candidates.add(
                        (
                            s1_id,
                            row["entity_id"]
                        )
                    )

        print(
            f"\r{source_name}: "
            f"{rows_processed:,} / {SOURCE_ROWS:,} rows | "
            f"candidates: {len(candidates):,}",
            end=""
        )

        if rows_processed >= SOURCE_ROWS:
            break

    print("\n")

    return candidates


# ============================================================
# RUN S2
# ============================================================

s2_candidates = process_source(
    S2_PATH,
    "Source 2"
)


# ============================================================
# RUN S3
# ============================================================

s3_candidates = process_source(
    S3_PATH,
    "Source 3"
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"S1 entities          : {len(s1):,}"
)

print(
    f"S2 rows tested       : {SOURCE_ROWS:,}"
)

print(
    f"S3 rows tested       : {SOURCE_ROWS:,}"
)

print(
    f"S2 candidate pairs   : {len(s2_candidates):,}"
)

print(
    f"S3 candidate pairs   : {len(s3_candidates):,}"
)

print(
    f"Total candidate pairs: "
    f"{len(s2_candidates | s3_candidates):,}"
)

print("\nSample candidates:")

for pair in list(
    s2_candidates | s3_candidates
)[:20]:

    print(
        pair[0],
        "->",
        pair[1]
    )


print("\n" + "=" * 70)
print("SAFE BLOCKING TEST COMPLETE")
print("=" * 70)
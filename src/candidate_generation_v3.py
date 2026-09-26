import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_S1 = 1000
SAMPLE_S2 = 10000
SAMPLE_S3 = 10000

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"


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


def normalize_name(value):
    return normalize_text(value)


def normalize_address(value):
    return normalize_text(value)


# ============================================================
# ADDRESS NUMBER EXTRACTION
# ============================================================

def extract_address_numbers(value):

    if pd.isna(value):
        return set()

    return set(
        re.findall(r"\d+", str(value))
    )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=SAMPLE_S1
)

print(f"S1 rows: {len(s1):,}")


print("Loading Source 2...")

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    nrows=SAMPLE_S2
)

print(f"S2 rows: {len(s2):,}")


print("Loading Source 3...")

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    nrows=SAMPLE_S3
)

print(f"S3 rows: {len(s3):,}")


# ============================================================
# NORMALIZE
# ============================================================

print("\nNormalizing data...")

for df in [s1, s2, s3]:

    df["norm_name"] = df["business_name"].apply(
        normalize_name
    )

    df["norm_address"] = df["business_address"].apply(
        normalize_address
    )

    df["address_numbers"] = df["business_address"].apply(
        extract_address_numbers
    )


# ============================================================
# BUILD BLOCKING INDEXES
# ============================================================

print("\nBuilding blocking indexes...")


# ------------------------------------------------------------
# INDEX 1: COUNTRY + ADDRESS NUMBER
# ------------------------------------------------------------

number_index_s2 = {}
number_index_s3 = {}


for idx, row in s2.iterrows():

    country = str(row["country"]).lower().strip()

    for number in row["address_numbers"]:

        key = (country, number)

        number_index_s2.setdefault(
            key,
            []
        ).append(row["entity_id"])


for idx, row in s3.iterrows():

    country = str(row["country"]).lower().strip()

    for number in row["address_numbers"]:

        key = (country, number)

        number_index_s3.setdefault(
            key,
            []
        ).append(row["entity_id"])


# ------------------------------------------------------------
# INDEX 2: COUNTRY + EXACT NORMALIZED NAME
# ------------------------------------------------------------

name_index_s2 = {}
name_index_s3 = {}


for _, row in s2.iterrows():

    name = row["norm_name"]

    if not name:
        continue

    country = str(row["country"]).lower().strip()

    key = (country, name)

    name_index_s2.setdefault(
        key,
        []
    ).append(row["entity_id"])


for _, row in s3.iterrows():

    name = row["norm_name"]

    if not name:
        continue

    country = str(row["country"]).lower().strip()

    key = (country, name)

    name_index_s3.setdefault(
        key,
        []
    ).append(row["entity_id"])


# ------------------------------------------------------------
# INDEX 3: COUNTRY + FIRST NAME TOKEN
# ------------------------------------------------------------

token_index_s2 = {}
token_index_s3 = {}


for _, row in s2.iterrows():

    name = row["norm_name"]

    if not name:
        continue

    tokens = name.split()

    if not tokens:
        continue

    first_token = tokens[0]

    country = str(row["country"]).lower().strip()

    key = (country, first_token)

    token_index_s2.setdefault(
        key,
        []
    ).append(row["entity_id"])


for _, row in s3.iterrows():

    name = row["norm_name"]

    if not name:
        continue

    tokens = name.split()

    if not tokens:
        continue

    first_token = tokens[0]

    country = str(row["country"]).lower().strip()

    key = (country, first_token)

    token_index_s3.setdefault(
        key,
        []
    ).append(row["entity_id"])


# ============================================================
# CANDIDATE GENERATION
# ============================================================

print("\nGenerating candidates...")

candidate_pairs = set()


for _, row in s1.iterrows():

    s1_id = row["entity_id"]

    country = str(
        row["country"]
    ).lower().strip()

    # --------------------------------------------------------
    # BLOCK 1: COUNTRY + ADDRESS NUMBER
    # --------------------------------------------------------

    for number in row["address_numbers"]:

        key = (country, number)

        for s2_id in number_index_s2.get(
            key,
            []
        ):

            candidate_pairs.add(
                (s1_id, s2_id)
            )

        for s3_id in number_index_s3.get(
            key,
            []
        ):

            candidate_pairs.add(
                (s1_id, s3_id)
            )

    # --------------------------------------------------------
    # BLOCK 2: COUNTRY + EXACT NAME
    # --------------------------------------------------------

    name = row["norm_name"]

    if name:

        key = (country, name)

        for s2_id in name_index_s2.get(
            key,
            []
        ):

            candidate_pairs.add(
                (s1_id, s2_id)
            )

        for s3_id in name_index_s3.get(
            key,
            []
        ):

            candidate_pairs.add(
                (s1_id, s3_id)
            )

    # --------------------------------------------------------
    # BLOCK 3: COUNTRY + FIRST NAME TOKEN
    # --------------------------------------------------------

    tokens = name.split()

    if tokens:

        first_token = tokens[0]

        key = (country, first_token)

        for s2_id in token_index_s2.get(
            key,
            []
        ):

            candidate_pairs.add(
                (s1_id, s2_id)
            )

        for s3_id in token_index_s3.get(
            key,
            []
        ):

            candidate_pairs.add(
                (s1_id, s3_id)
            )


# ============================================================
# SEPARATE S2 / S3 CANDIDATES
# ============================================================

s2_candidates = [
    pair
    for pair in candidate_pairs
    if pair[1].startswith("S2-")
]

s3_candidates = [
    pair
    for pair in candidate_pairs
    if pair[1].startswith("S3-")
]


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("V3 BLOCKING RESULTS")
print("=" * 70)

print(
    f"Total candidate pairs : "
    f"{len(candidate_pairs):,}"
)

print(
    f"S2 candidate pairs    : "
    f"{len(s2_candidates):,}"
)

print(
    f"S3 candidate pairs    : "
    f"{len(s3_candidates):,}"
)

print(
    f"S1 entities analyzed  : "
    f"{len(s1):,}"
)


# ============================================================
# SAMPLE CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE CANDIDATE PAIRS")
print("=" * 70)

for s1_id, match_id in list(
    candidate_pairs
)[:20]:

    print(
        f"{s1_id} -> {match_id}"
    )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("V3 candidate generation complete.")
print("=" * 70)
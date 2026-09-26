import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import pandas as pd
from rapidfuzz.fuzz import ratio


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 1000

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

    # Unicode normalization
    value = value.replace("&", " and ")

    # Remove punctuation
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value):
    return normalize_text(value)


def normalize_address(value):
    return normalize_text(value)


# ============================================================
# ADDRESS HELPERS
# ============================================================

def extract_numbers(value):

    if pd.isna(value):
        return set()

    return set(re.findall(r"\d+", str(value)))


def address_tokens(value):

    normalized = normalize_address(value)

    if not normalized:
        return set()

    return set(normalized.split())


def token_overlap(a, b):

    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=SAMPLE_SIZE
)

s1 = s1.set_index("entity_id")

sample_s1_ids = set(s1.index)

print(f"S1 sample: {len(s1):,}")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t"
)

gt = gt[
    gt["source1_entity_id"].isin(sample_s1_ids)
].copy()

print(f"Ground truth rows: {len(gt):,}")


# ============================================================
# COLLECT TRUE MATCH PAIRS
# ============================================================

true_pairs = []

s2_ids = set()
s3_ids = set()

for _, row in gt.iterrows():

    matched = row["matched_entity_ids"]

    if pd.isna(matched):
        continue

    matched = str(matched).strip()

    if not matched:
        continue

    ids = [
        x.strip()
        for x in matched.split(",")
        if x.strip()
    ]

    for match_id in ids:

        if match_id.startswith("S2-"):
            s2_ids.add(match_id)
            true_pairs.append(
                (row["source1_entity_id"], match_id, "S2")
            )

        elif match_id.startswith("S3-"):
            s3_ids.add(match_id)
            true_pairs.append(
                (row["source1_entity_id"], match_id, "S3")
            )


print(f"\nTrue S2 pairs: {len([x for x in true_pairs if x[2] == 'S2']):,}")
print(f"True S3 pairs: {len([x for x in true_pairs if x[2] == 'S3']):,}")


# ============================================================
# LOAD REQUIRED S2 RECORDS
# ============================================================

print("\nScanning Source 2...")

s2_records = {}

for chunk in pd.read_csv(
    S2_PATH,
    sep="\t",
    chunksize=200_000
):

    selected = chunk[
        chunk["entity_id"].isin(s2_ids)
    ]

    for _, row in selected.iterrows():

        s2_records[row["entity_id"]] = {
            "name": row["business_name"],
            "address": row["business_address"],
            "country": row["country"]
        }

    if len(s2_records) == len(s2_ids):
        break


# ============================================================
# LOAD REQUIRED S3 RECORDS
# ============================================================

print("Scanning Source 3...")

s3_records = {}

for chunk in pd.read_csv(
    S3_PATH,
    sep="\t",
    chunksize=200_000
):

    selected = chunk[
        chunk["entity_id"].isin(s3_ids)
    ]

    for _, row in selected.iterrows():

        s3_records[row["entity_id"]] = {
            "name": row["business_name"],
            "address": row["business_address"],
            "country": row["country"]
        }

    if len(s3_records) == len(s3_ids):
        break


# ============================================================
# ANALYZE TRUE PAIRS
# ============================================================

print("\n" + "=" * 80)
print("TRUE MATCH SIGNAL ANALYSIS")
print("=" * 80)

total = 0

name_exact = 0
name_high = 0
name_medium = 0

address_exact = 0
address_high = 0
address_medium = 0

same_country = 0
same_number = 0

address_token_high = 0


for s1_id, match_id, source in true_pairs:

    if s1_id not in s1.index:
        continue

    if source == "S2":
        record = s2_records.get(match_id)
    else:
        record = s3_records.get(match_id)

    if record is None:
        continue

    s1_row = s1.loc[s1_id]

    s1_name = normalize_name(
        s1_row["business_name"]
    )

    match_name = normalize_name(
        record["name"]
    )

    s1_address = normalize_address(
        s1_row["business_address"]
    )

    match_address = normalize_address(
        record["address"]
    )

    # --------------------------------------------------------
    # NAME SIMILARITY
    # --------------------------------------------------------

    name_score = ratio(
        s1_name,
        match_name
    ) if s1_name and match_name else 0

    # --------------------------------------------------------
    # ADDRESS SIMILARITY
    # --------------------------------------------------------

    address_score = ratio(
        s1_address,
        match_address
    ) if s1_address and match_address else 0

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    if (
        not pd.isna(s1_row["country"])
        and not pd.isna(record["country"])
        and str(s1_row["country"]).strip().lower()
        == str(record["country"]).strip().lower()
    ):
        same_country += 1

    # --------------------------------------------------------
    # NAME COUNTS
    # --------------------------------------------------------

    if s1_name and s1_name == match_name:
        name_exact += 1

    if name_score >= 85:
        name_high += 1

    if name_score >= 70:
        name_medium += 1

    # --------------------------------------------------------
    # ADDRESS COUNTS
    # --------------------------------------------------------

    if s1_address and s1_address == match_address:
        address_exact += 1

    if address_score >= 85:
        address_high += 1

    if address_score >= 70:
        address_medium += 1

    # --------------------------------------------------------
    # ADDRESS NUMBER
    # --------------------------------------------------------

    s1_numbers = extract_numbers(
        s1_row["business_address"]
    )

    match_numbers = extract_numbers(
        record["address"]
    )

    if s1_numbers and match_numbers:
        if s1_numbers & match_numbers:
            same_number += 1

    # --------------------------------------------------------
    # ADDRESS TOKEN OVERLAP
    # --------------------------------------------------------

    overlap = token_overlap(
        address_tokens(s1_address),
        address_tokens(match_address)
    )

    if overlap >= 0.50:
        address_token_high += 1

    total += 1


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nTotal valid true pairs analyzed :", f"{total:,}")

if total > 0:

    print("\n" + "-" * 80)
    print("NAME SIGNALS")
    print("-" * 80)

    print(
        f"Exact normalized name : "
        f"{name_exact:,} "
        f"({name_exact / total * 100:.2f}%)"
    )

    print(
        f"Name similarity >= 85 : "
        f"{name_high:,} "
        f"({name_high / total * 100:.2f}%)"
    )

    print(
        f"Name similarity >= 70 : "
        f"{name_medium:,} "
        f"({name_medium / total * 100:.2f}%)"
    )

    print("\n" + "-" * 80)
    print("ADDRESS SIGNALS")
    print("-" * 80)

    print(
        f"Exact normalized address : "
        f"{address_exact:,} "
        f"({address_exact / total * 100:.2f}%)"
    )

    print(
        f"Address similarity >= 85 : "
        f"{address_high:,} "
        f"({address_high / total * 100:.2f}%)"
    )

    print(
        f"Address similarity >= 70 : "
        f"{address_medium:,} "
        f"({address_medium / total * 100:.2f}%)"
    )

    print(
        f"Address token overlap >= 50% : "
        f"{address_token_high:,} "
        f"({address_token_high / total * 100:.2f}%)"
    )

    print("\n" + "-" * 80)
    print("OTHER SIGNALS")
    print("-" * 80)

    print(
        f"Same country : "
        f"{same_country:,} "
        f"({same_country / total * 100:.2f}%)"
    )

    print(
        f"Same address number : "
        f"{same_number:,} "
        f"({same_number / total * 100:.2f}%)"
    )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 80)
print("Analysis complete.")
print("=" * 80)
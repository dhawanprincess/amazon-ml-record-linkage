import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 1000
NUM_EXAMPLES = 2

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"
GT_PATH = "dataset/train/train_ground_truth.tsv"


# ============================================================
# LOAD SOURCE 1 SAMPLE
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=SAMPLE_SIZE
)

s1 = s1.set_index("entity_id")

sample_s1_ids = set(s1.index)

print(f"Source 1 sample loaded: {len(s1):,}")


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

print(f"Ground truth rows for sample: {len(gt):,}")


# ============================================================
# COLLECT TRUE MATCH IDs
# ============================================================

s2_ids = set()
s3_ids = set()

examples = []

for _, row in gt.iterrows():

    matched = row["matched_entity_ids"]

    # Ignore missing matches
    if pd.isna(matched):
        continue

    matched = str(matched).strip()

    # Ignore empty matches
    if not matched:
        continue

    ids = [
        x.strip()
        for x in matched.split(",")
        if x.strip()
    ]

    # Collect S2 and S3 IDs
    for entity_id in ids:

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)

    # Store only first NUM_EXAMPLES
    if len(examples) < NUM_EXAMPLES:
        examples.append(
            (
                row["source1_entity_id"],
                ids
            )
        )


# ============================================================
# PRINT MATCH ID SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TRUE MATCH ID SUMMARY")
print("=" * 80)

print(f"S2 true IDs : {len(s2_ids):,}")
print(f"S3 true IDs : {len(s3_ids):,}")
print(f"Examples    : {len(examples):,}")


# ============================================================
# LOAD REQUIRED SOURCE 2 RECORDS
# ============================================================

print("\nScanning Source 2...")

s2_records = {}

for chunk in pd.read_csv(
    S2_PATH,
    sep="\t",
    chunksize=200_000
):

    matched = chunk[
        chunk["entity_id"].isin(s2_ids)
    ]

    for _, row in matched.iterrows():

        s2_records[row["entity_id"]] = {
            "name": row["business_name"],
            "address": row["business_address"],
            "country": row["country"]
        }

    if len(s2_records) == len(s2_ids):
        break


# ============================================================
# LOAD REQUIRED SOURCE 3 RECORDS
# ============================================================

print("Scanning Source 3...")

s3_records = {}

for chunk in pd.read_csv(
    S3_PATH,
    sep="\t",
    chunksize=200_000
):

    matched = chunk[
        chunk["entity_id"].isin(s3_ids)
    ]

    for _, row in matched.iterrows():

        s3_records[row["entity_id"]] = {
            "name": row["business_name"],
            "address": row["business_address"],
            "country": row["country"]
        }

    if len(s3_records) == len(s3_ids):
        break


# ============================================================
# DISPLAY COMPLETE TRUE MATCH EXAMPLES
# ============================================================

print("\n" + "=" * 100)
print("TRUE MATCH ANALYSIS")
print("=" * 100)


for i, (s1_id, matched_ids) in enumerate(examples, start=1):

    source1 = s1.loc[s1_id]

    print("\n" + "-" * 100)
    print(f"EXAMPLE {i}")
    print("-" * 100)

    # --------------------------------------------------------
    # SOURCE 1
    # --------------------------------------------------------

    print("\nSOURCE 1")
    print(f"ID      : {s1_id}")
    print(f"Name    : {source1['business_name']}")
    print(f"Address : {source1['business_address']}")
    print(f"Country : {source1['country']}")

    # --------------------------------------------------------
    # TRUE MATCHES
    # --------------------------------------------------------

    print("\nTRUE MATCHES")

    for match_id in matched_ids:

        if match_id.startswith("S2-"):

            record = s2_records.get(match_id)
            source = "S2"

        elif match_id.startswith("S3-"):

            record = s3_records.get(match_id)
            source = "S3"

        else:

            record = None
            source = "UNKNOWN"

        if record is None:

            print(f"\n{source}")
            print(f"ID      : {match_id}")
            print("Record  : NOT FOUND")

            continue

        print(f"\n{source}")
        print(f"ID      : {match_id}")
        print(f"Name    : {record['name']}")
        print(f"Address : {record['address']}")
        print(f"Country : {record['country']}")


# ============================================================
# RECORD LOOKUP SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("RECORD LOOKUP SUMMARY")
print("=" * 100)

print(f"S2 true IDs expected : {len(s2_ids):,}")
print(f"S2 records found     : {len(s2_records):,}")

print(f"S3 true IDs expected : {len(s3_ids):,}")
print(f"S3 records found     : {len(s3_records):,}")

print("\nAnalysis complete.")
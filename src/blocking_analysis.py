import pandas as pd
from collections import Counter

# ============================================================
# FILE PATHS
# ============================================================

S1_PATH = "dataset/train/train_source1.tsv"
GT_PATH = "dataset/train/train_ground_truth.tsv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=1000
)

print("Loading Ground Truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t"
)

print("Data loaded successfully.")


# ============================================================
# FILTER GROUND TRUTH FOR SAMPLE SOURCE 1
# ============================================================

sample_s1_ids = set(s1["entity_id"])

gt_sample = gt[
    gt["source1_entity_id"].isin(sample_s1_ids)
].copy()


# ============================================================
# ANALYZE MATCHES
# ============================================================

total_s1 = len(s1)

with_matches = []
without_matches = 0

s2_count = 0
s3_count = 0

match_counts = []


for _, row in gt_sample.iterrows():

    matched_ids = row["matched_entity_ids"]

    # Handle empty / missing matches
    if pd.isna(matched_ids) or str(matched_ids).strip() == "":
        without_matches += 1
        continue

    matched_ids = str(matched_ids).strip()

    if not matched_ids:
        without_matches += 1
        continue

    with_matches.append(row["source1_entity_id"])

    ids = [
        x.strip()
        for x in matched_ids.split(",")
        if x.strip()
    ]

    match_counts.append(len(ids))

    for entity_id in ids:

        if entity_id.startswith("S2-"):
            s2_count += 1

        elif entity_id.startswith("S3-"):
            s3_count += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Sample Source 1 entities : {total_s1:,}")
print(f"Entities with matches    : {len(with_matches):,}")
print(f"Entities without matches : {without_matches:,}")
print(f"Total S2 matches         : {s2_count:,}")
print(f"Total S3 matches         : {s3_count:,}")


# ============================================================
# MATCH COUNT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("MATCH COUNT DISTRIBUTION")
print("=" * 70)

if match_counts:

    counter = Counter(match_counts)

    for count in sorted(counter):
        print(
            f"{count} match(es) : "
            f"{counter[count]:,} Source 1 entities"
        )

else:
    print("No matched entities found.")


# ============================================================
# SAMPLE TRUE MATCHES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE TRUE MATCHES")
print("=" * 70)

sample_matches = gt_sample[
    gt_sample["matched_entity_ids"].notna() &
    (gt_sample["matched_entity_ids"].astype(str).str.strip() != "")
].head(10)

for _, row in sample_matches.iterrows():

    print(
        f"{row['source1_entity_id']} "
        f"-> {row['matched_entity_ids']}"
    )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("Analysis complete.")
print("=" * 70)
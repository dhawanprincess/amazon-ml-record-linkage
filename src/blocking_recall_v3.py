import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from candidate_generation_v3 import (
    s1,
    s2,
    s3,
    candidate_pairs
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

GT_PATH = "dataset/train/train_ground_truth.tsv"

print("Loading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t"
)


# ============================================================
# SAMPLE S1 IDS
# ============================================================

sample_s1_ids = set(
    s1["entity_id"]
)


gt_sample = gt[
    gt["source1_entity_id"].isin(sample_s1_ids)
].copy()


# ============================================================
# BUILD TRUE PAIR SET
# ============================================================

true_pairs = set()

true_s2 = set()
true_s3 = set()


for _, row in gt_sample.iterrows():

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

        pair = (
            row["source1_entity_id"],
            match_id
        )

        true_pairs.add(pair)

        if match_id.startswith("S2-"):
            true_s2.add(pair)

        elif match_id.startswith("S3-"):
            true_s3.add(pair)


# ============================================================
# CALCULATE RECALL
# ============================================================

surviving_pairs = (
    true_pairs &
    candidate_pairs
)

surviving_s2 = (
    true_s2 &
    candidate_pairs
)

surviving_s3 = (
    true_s3 &
    candidate_pairs
)


# ============================================================
# RESULTS
# ============================================================

total_true = len(true_pairs)

total_true_s2 = len(true_s2)
total_true_s3 = len(true_s3)

surviving_total = len(surviving_pairs)

surviving_s2_count = len(surviving_s2)
surviving_s3_count = len(surviving_s3)


print("\n" + "=" * 70)
print("V3 BLOCKING RECALL")
print("=" * 70)

print(
    f"Total true matches       : "
    f"{total_true:,}"
)

print(
    f"True S2 matches          : "
    f"{total_true_s2:,}"
)

print(
    f"True S3 matches          : "
    f"{total_true_s3:,}"
)

print(
    f"Surviving true matches   : "
    f"{surviving_total:,}"
)

print(
    f"Surviving S2 matches     : "
    f"{surviving_s2_count:,}"
)

print(
    f"Surviving S3 matches     : "
    f"{surviving_s3_count:,}"
)


# ============================================================
# RECALL
# ============================================================

if total_true > 0:

    overall_recall = (
        surviving_total /
        total_true *
        100
    )

    print(
        f"\nOverall Blocking Recall : "
        f"{overall_recall:.2f}%"
    )


if total_true_s2 > 0:

    s2_recall = (
        surviving_s2_count /
        total_true_s2 *
        100
    )

    print(
        f"S2 Blocking Recall      : "
        f"{s2_recall:.2f}%"
    )


if total_true_s3 > 0:

    s3_recall = (
        surviving_s3_count /
        total_true_s3 *
        100
    )

    print(
        f"S3 Blocking Recall      : "
        f"{s3_recall:.2f}%"
    )


# ============================================================
# MISSED TRUE MATCHES
# ============================================================

missed = true_pairs - candidate_pairs

print(
    f"\nMissed true matches      : "
    f"{len(missed):,}"
)


print("\n" + "=" * 70)
print("Analysis complete.")
print("=" * 70)

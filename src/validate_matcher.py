import pandas as pd
import sys

sys.stdout.reconfigure(encoding="utf-8")

TRAIN_S1 = "dataset/train/train_source1.tsv"
GROUND_TRUTH = "dataset/train/train_ground_truth.tsv"

S1_LIMIT = 100

print("=" * 70)
print("MATCHER VALIDATION — 100 TRAINING S1")
print("=" * 70)

# ------------------------------------------------------------
# LOAD TRAINING S1
# ------------------------------------------------------------

print("\nLoading training Source 1...")

s1 = pd.read_csv(
    TRAIN_S1,
    sep="\t",
    nrows=S1_LIMIT,
    dtype=str
).fillna("")

print(f"S1 rows: {len(s1)}")


# ------------------------------------------------------------
# LOAD GROUND TRUTH
# ------------------------------------------------------------

print("\nLoading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str
).fillna("")

gt = gt[
    gt["source1_entity_id"].isin(
        s1["entity_id"]
    )
].copy()

print(f"Ground-truth rows: {len(gt)}")


# ------------------------------------------------------------
# PARSE TRUE MATCHES
# ------------------------------------------------------------

true_matches = {}

for _, row in gt.iterrows():

    sid = row["source1_entity_id"]

    raw = str(
        row["matched_entity_ids"]
    ).strip()

    if not raw:
        true_matches[sid] = set()
        continue

    ids = {
        x.strip()
        for x in raw.split(",")
        if x.strip()
    }

    true_matches[sid] = ids


# ------------------------------------------------------------
# SHOW BASIC GROUND TRUTH STATISTICS
# ------------------------------------------------------------

entities_with_match = sum(
    1
    for ids in true_matches.values()
    if ids
)

entities_without_match = (
    len(true_matches)
    - entities_with_match
)

total_true_pairs = sum(
    len(ids)
    for ids in true_matches.values()
)

print("\n" + "-" * 70)
print("GROUND TRUTH SUMMARY")
print("-" * 70)

print(
    f"Entities with matches    : "
    f"{entities_with_match}"
)

print(
    f"Entities without matches : "
    f"{entities_without_match}"
)

print(
    f"Total true match pairs   : "
    f"{total_true_pairs}"
)


# ------------------------------------------------------------
# CHECK THE DRY-RUN OUTPUT
# ------------------------------------------------------------

OUTPUT = "output/matching_results.tsv"

print("\nLoading matcher output...")

pred = pd.read_csv(
    OUTPUT,
    sep="\t",
    dtype=str
).fillna("")

pred = pred[
    pred["source1_entity_id"].isin(
        s1["entity_id"]
    )
]

print(
    f"Prediction rows: {len(pred)}"
)


# ------------------------------------------------------------
# PARSE PREDICTIONS
# ------------------------------------------------------------

predicted_matches = {}

for _, row in pred.iterrows():

    sid = row["source1_entity_id"]

    raw = str(
        row["matched_entity_ids"]
    ).strip()

    if not raw:
        predicted_matches[sid] = set()
        continue

    ids = {
        x.strip()
        for x in raw.split(",")
        if x.strip()
    }

    predicted_matches[sid] = ids


# ------------------------------------------------------------
# CALCULATE METRICS
# ------------------------------------------------------------

TP = 0
FP = 0
FN = 0

for sid in true_matches:

    actual = true_matches.get(
        sid,
        set()
    )

    predicted = predicted_matches.get(
        sid,
        set()
    )

    TP += len(
        actual & predicted
    )

    FP += len(
        predicted - actual
    )

    FN += len(
        actual - predicted
    )


precision = (
    TP / (TP + FP)
    if TP + FP > 0
    else 0
)

recall = (
    TP / (TP + FN)
    if TP + FN > 0
    else 0
)

if precision + recall > 0:

    f05 = (
        1.25 * precision * recall
        /
        (0.25 * precision + recall)
    )

else:
    f05 = 0


# ------------------------------------------------------------
# ENTITY-LEVEL EXACTNESS
# ------------------------------------------------------------

exact_entities = 0

for sid in true_matches:

    if (
        true_matches.get(sid, set())
        ==
        predicted_matches.get(sid, set())
    ):
        exact_entities += 1


entity_accuracy = (
    exact_entities / len(true_matches)
    if true_matches
    else 0
)


# ------------------------------------------------------------
# PRINT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"True Positives  : {TP}")
print(f"False Positives : {FP}")
print(f"False Negatives : {FN}")

print(
    f"\nPrecision       : "
    f"{precision:.4f}"
)

print(
    f"Recall          : "
    f"{recall:.4f}"
)

print(
    f"F0.5            : "
    f"{f05:.4f}"
)

print(
    f"Exact entities  : "
    f"{exact_entities}/{len(true_matches)}"
)

print(
    f"Entity accuracy : "
    f"{entity_accuracy:.4f}"
)

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
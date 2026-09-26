import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from candidate_generation_v2 import (
    load_sample,
    prepare_dataframe,
    create_candidates
)

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"


# --------------------------------------------------
# Load sample
# --------------------------------------------------

S1_SAMPLE_SIZE = 1000
S2_SAMPLE_SIZE = 10000

print("Loading Source 1...")
s1 = load_sample("train_source1.tsv", S1_SAMPLE_SIZE)

print("Loading Source 2...")
s2 = load_sample("train_source2.tsv", S2_SAMPLE_SIZE)

print("Preparing data...")
s1 = prepare_dataframe(s1)
s2 = prepare_dataframe(s2)


# --------------------------------------------------
# Generate candidate pairs
# --------------------------------------------------

print("Generating candidates...")

candidates = create_candidates(s1, s2)

candidate_set = set(
    zip(
        candidates["entity_id_s1"],
        candidates["entity_id_s2"]
    )
)

print("Candidate pairs:", len(candidate_set))


# --------------------------------------------------
# Load ground truth
# --------------------------------------------------

print("Loading ground truth...")

ground_truth = pd.read_csv(
    TRAIN_DIR / "train_ground_truth.tsv",
    sep="\t"
)


# Only Source 1 entities present in our sample
sample_s1_ids = set(s1["entity_id"])

ground_truth = ground_truth[
    ground_truth["source1_entity_id"].isin(sample_s1_ids)
]


# --------------------------------------------------
# Extract true S2 matches
# --------------------------------------------------

true_pairs = set()

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]
    matched_ids = row["matched_entity_ids"]

    if pd.isna(matched_ids):
        continue

    matched_ids = str(matched_ids).strip()

    if not matched_ids:
        continue

    for entity_id in matched_ids.split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            true_pairs.add((s1_id, entity_id))


# --------------------------------------------------
# Calculate blocking recall
# --------------------------------------------------

surviving_matches = true_pairs.intersection(candidate_set)

total_true_matches = len(true_pairs)
surviving_true_matches = len(surviving_matches)

if total_true_matches > 0:
    recall = (
        surviving_true_matches /
        total_true_matches
    )

    recall_percent = recall * 100
else:
    recall_percent = 0


print("\n" + "=" * 60)
print("BLOCKING RECALL RESULTS")
print("=" * 60)

print("True S1-S2 matches:", total_true_matches)
print("Matches surviving blocking:", surviving_true_matches)
print(f"Blocking Recall: {recall_percent:.2f}%")
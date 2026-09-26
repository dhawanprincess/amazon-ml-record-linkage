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
# GENERIC BUSINESS WORDS
# ============================================================

STOPWORDS = {
    "the",
    "and",
    "company",
    "companies",
    "corporation",
    "corp",
    "inc",
    "incorporated",
    "limited",
    "ltd",
    "llc",
    "llp",
    "private",
    "pvt",
    "plc",
    "group",
    "holdings",
    "services",
    "service",
    "solutions",
    "enterprise",
    "enterprises",
    "international",
    "india",
    "usa",
    "united",
    "states",
}


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

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def useful_tokens(value):

    text = normalize_text(value)

    tokens = text.split()

    result = set()

    for token in tokens:

        if len(token) < 3:
            continue

        if token in STOPWORDS:
            continue

        result.add(token)

    return result


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
print("V7 HIGH-RECALL BLOCKING TEST")
print("=" * 70)

print("\nLoading Source 1...")

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

s1["norm_name"] = s1[
    "business_name"
].map(normalize_text)

s1["name_tokens"] = s1[
    "business_name"
].map(useful_tokens)

s1["address_tokens"] = s1[
    "business_address"
].map(useful_tokens)

s1["address_number"] = s1[
    "business_address"
].map(address_number)


# ============================================================
# BUILD BLOCK INDEXES
# ============================================================

name_token_blocks = {}
address_token_blocks = {}
number_blocks = {}
exact_name_blocks = {}

for _, row in s1.iterrows():

    s1_id = row["entity_id"]
    country = row["country_norm"]

    # --------------------------------------------------------
    # NAME TOKENS
    # --------------------------------------------------------

    for token in row["name_tokens"]:

        key = country + "|N|" + token

        name_token_blocks.setdefault(
            key,
            set()
        ).add(s1_id)

    # --------------------------------------------------------
    # ADDRESS TOKENS
    # --------------------------------------------------------

    for token in row["address_tokens"]:

        key = country + "|A|" + token

        address_token_blocks.setdefault(
            key,
            set()
        ).add(s1_id)

    # --------------------------------------------------------
    # ADDRESS NUMBER
    # --------------------------------------------------------

    number = row["address_number"]

    if number:

        key = country + "|D|" + number

        number_blocks.setdefault(
            key,
            set()
        ).add(s1_id)

    # --------------------------------------------------------
    # EXACT NORMALIZED NAME
    # --------------------------------------------------------

    if row["norm_name"]:

        key = country + "|E|" + row["norm_name"]

        exact_name_blocks.setdefault(
            key,
            set()
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

def process_source(
    path,
    prefix,
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

        remaining = SOURCE_ROWS - rows_processed

        if remaining <= 0:
            break

        if len(chunk) > remaining:

            chunk = chunk.iloc[
                :remaining
            ]

        rows_processed += len(chunk)

        # ----------------------------------------------------
        # Process each source record
        # ----------------------------------------------------

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

            name_tokens = useful_tokens(
                row["business_name"]
            )

            address_tokens = useful_tokens(
                row["business_address"]
            )

            number = address_number(
                row["business_address"]
            )

            matched_s1 = set()

            # ------------------------------------------------
            # BLOCK 1 — NAME TOKEN
            # ------------------------------------------------

            for token in name_tokens:

                key = country + "|N|" + token

                matched_s1.update(
                    name_token_blocks.get(
                        key,
                        ()
                    )
                )

            # ------------------------------------------------
            # BLOCK 2 — ADDRESS TOKEN
            # ------------------------------------------------

            for token in address_tokens:

                key = country + "|A|" + token

                matched_s1.update(
                    address_token_blocks.get(
                        key,
                        ()
                    )
                )

            # ------------------------------------------------
            # BLOCK 3 — ADDRESS NUMBER
            # ------------------------------------------------

            if number:

                key = country + "|D|" + number

                matched_s1.update(
                    number_blocks.get(
                        key,
                        ()
                    )
                )

            # ------------------------------------------------
            # BLOCK 4 — EXACT NAME
            # ------------------------------------------------

            if name:

                key = country + "|E|" + name

                matched_s1.update(
                    exact_name_blocks.get(
                        key,
                        ()
                    )
                )

            # ------------------------------------------------
            # STORE
            # ------------------------------------------------

            for s1_id in matched_s1:

                candidates.add(
                    (
                        s1_id,
                        entity_id
                    )
                )

        print(
            f"\rProcessed: {rows_processed:,} | "
            f"Candidates: {len(candidates):,}",
            end=""
        )

        if rows_processed >= SOURCE_ROWS:
            break

    print("\n")

    # ========================================================
    # RECALL
    # ========================================================

    total_true = 0
    survived_true = 0

    entities_with_true = 0
    entities_hit = 0

    # Build candidate lookup for faster evaluation
    candidate_lookup = {}

    for s1_id, target_id in candidates:

        candidate_lookup.setdefault(
            s1_id,
            set()
        ).add(target_id)

    for s1_id, true_ids in true_matches.items():

        source_true = {
            x
            for x in true_ids
            if x.startswith(prefix)
        }

        if not source_true:
            continue

        entities_with_true += 1
        total_true += len(source_true)

        retrieved = (
            candidate_lookup.get(
                s1_id,
                set()
            )
        )

        survived = (
            source_true & retrieved
        )

        survived_true += len(survived)

        if survived:
            entities_hit += 1

    # ========================================================
    # RESULTS
    # ========================================================

    print("=" * 70)
    print(f"{source_name} RESULTS")
    print("=" * 70)

    print(
        f"True matches             : {total_true:,}"
    )

    print(
        f"True matches survived    : {survived_true:,}"
    )

    print(
        f"Entities with true match : "
        f"{entities_with_true:,}"
    )

    print(
        f"Entities with >=1 hit    : "
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
        f"{len(candidates):,}"
    )


# ============================================================
# S2
# ============================================================

process_source(
    S2_PATH,
    "S2-",
    "Source 2"
)


# ============================================================
# S3
# ============================================================

process_source(
    S3_PATH,
    "S3-",
    "Source 3"
)


print("\n" + "=" * 70)
print("V7 TEST COMPLETE")
print("=" * 70)
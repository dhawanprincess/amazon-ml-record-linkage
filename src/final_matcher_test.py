import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import pandas as pd
from rapidfuzz.fuzz import ratio


# ============================================================
# SETTINGS
# ============================================================

S1_ROWS = 100
TARGET_ROWS = 25_000

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def norm(value):
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


def tokens(value):
    return set(
        x for x in norm(value).split()
        if len(x) >= 3
    )


def address_number(value):

    if pd.isna(value):
        return ""

    m = re.search(
        r"\b\d{1,6}\b",
        str(value)
    )

    return m.group() if m else ""


# ============================================================
# COMBINED TEXT
# ============================================================

def combined(name, address):
    return (
        norm(name) + " " +
        norm(address)
    ).strip()


# ============================================================
# SCORE
# ============================================================

def similarity(s1_name, s1_addr, s2_name, s2_addr):

    name1 = norm(s1_name)
    name2 = norm(s2_name)

    addr1 = norm(s1_addr)
    addr2 = norm(s2_addr)

    name_score = ratio(
        name1,
        name2
    )

    addr_score = ratio(
        addr1,
        addr2
    )

    full_score = ratio(
        combined(s1_name, s1_addr),
        combined(s2_name, s2_addr)
    )

    # Address number agreement
    n1 = address_number(s1_addr)
    n2 = address_number(s2_addr)

    number_bonus = 10 if (
        n1 and n2 and n1 == n2
    ) else 0

    # Token overlap
    name_tokens_1 = tokens(s1_name)
    name_tokens_2 = tokens(s2_name)

    if name_tokens_1 and name_tokens_2:

        overlap = (
            len(
                name_tokens_1 &
                name_tokens_2
            )
            /
            len(
                name_tokens_1 |
                name_tokens_2
            )
        )

    else:
        overlap = 0

    score = (
        0.40 * name_score
        + 0.30 * addr_score
        + 0.30 * full_score
        + number_bonus
        + 10 * overlap
    )

    return score


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("FINAL MATCHER — SMALL TEST")
print("=" * 70)

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=S1_ROWS
)

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    nrows=TARGET_ROWS
)

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    nrows=TARGET_ROWS
)

print(
    f"S1: {len(s1):,} | "
    f"S2: {len(s2):,} | "
    f"S3: {len(s3):,}"
)


# ============================================================
# MATCH FUNCTION
# ============================================================

def match_source(
    s1_df,
    target_df,
    source_name
):

    print("\n" + "-" * 70)
    print(source_name)
    print("-" * 70)

    results = {}

    for count, (_, row1) in enumerate(
        s1_df.iterrows(),
        start=1
    ):

        scored = []

        for _, row2 in target_df.iterrows():

            # Country must match
            country1 = norm(row1["country"])
            country2 = norm(row2["country"])

            if (
                country1
                and country2
                and country1 != country2
            ):
                continue

            score = similarity(
                row1["business_name"],
                row1["business_address"],
                row2["business_name"],
                row2["business_address"]
            )

            scored.append(
                (
                    score,
                    row2["entity_id"]
                )
            )

        scored.sort(
            reverse=True
        )

        results[
            row1["entity_id"]
        ] = scored[:10]

        if count % 10 == 0:
            print(
                f"Processed S1: "
                f"{count}/{len(s1_df)}"
            )

    return results


# ============================================================
# RUN
# ============================================================

s2_results = match_source(
    s1,
    s2,
    "SOURCE 2"
)

s3_results = match_source(
    s1,
    s3,
    "SOURCE 3"
)


# ============================================================
# SHOW RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE FINAL SCORES")
print("=" * 70)

for s1_id in list(
    s2_results.keys()
)[:5]:

    print(
        "\nS1:",
        s1_id
    )

    print("S2:")

    for score, entity_id in (
        s2_results[s1_id][:5]
    ):

        print(
            f"  {entity_id} "
            f"score={score:.2f}"
        )

    print("S3:")

    for score, entity_id in (
        s3_results[s1_id][:5]
    ):

        print(
            f"  {entity_id} "
            f"score={score:.2f}"
        )


print("\n" + "=" * 70)
print("SMALL MATCHER TEST COMPLETE")
print("=" * 70)
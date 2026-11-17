
# ============================================================
# PROJECT: AI-BASED NETWORK ATTACK DETECTION
# STEP 9: FINAL LABEL VERIFICATION
#
# Objective:
# 1. Count benign and attack records.
# 2. Check attack categories.
# 3. Verify labels in the complete cleaned datasets.
#
# Input: Cleaned fixed datasets
# ============================================================

import pandas as pd

files = [
    "results/NF-UNSW-NB15-v3_cleaned_fixed.csv",
    "results/NF-CICIDS2018-v3_cleaned_fixed.csv"
]

CHUNK_SIZE = 200000

for file in files:

    print("\n" + "=" * 65)
    print("FINAL LABEL CHECK:", file)
    print("=" * 65)

    label_counts = {}
    attack_counts = {}

    # Read dataset in chunks to save memory
    for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):

        # Count binary labels
        current_labels = chunk["Label"].value_counts(dropna=False)

        for value, count in current_labels.items():
            label_counts[value] = (
                label_counts.get(value, 0) + count
            )

        # Count attack categories
        current_attacks = chunk["Attack"].value_counts(dropna=False)

        for value, count in current_attacks.items():
            attack_counts[value] = (
                attack_counts.get(value, 0) + count
            )

    # Display binary label distribution
    print("\nLABEL DISTRIBUTION:")

    for value, count in sorted(label_counts.items()):
        print(f"{value}: {count:,}")

    # Display attack category distribution
    print("\nATTACK CATEGORY DISTRIBUTION:")

    for value, count in sorted(
        attack_counts.items(),
        key=lambda x: str(x[0])
    ):
        print(f"{value}: {count:,}")

    print("\nTotal Label Records:", f"{sum(label_counts.values()):,}")
    print("Total Attack Records:", f"{sum(attack_counts.values()):,}")

print("\nFINAL LABEL VERIFICATION COMPLETED!")

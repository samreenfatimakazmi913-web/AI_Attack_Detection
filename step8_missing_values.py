
# ============================================================
# PROJECT: AI-BASED NETWORK ATTACK DETECTION
# STEP 8: MISSING VALUES VERIFICATION
#
# Objective:
# 1. Check missing values in cleaned datasets.
# 2. Identify columns containing missing values.
# 3. Count total missing values.
# 4. Preserve all original records.
#
# Input: Cleaned fixed datasets
# ============================================================

import pandas as pd
import os

# Folder containing cleaned datasets
DATASET_DIR = "results"

# Use the newly verified cleaned files
files = [
    "NF-UNSW-NB15-v3_cleaned_fixed.csv",
    "NF-CICIDS2018-v3_cleaned_fixed.csv"
]

CHUNK_SIZE = 200000

# Check each dataset separately
for file in files:

    print("\n" + "=" * 60)
    print("CHECKING:", file)
    print("=" * 60)

    file_path = os.path.join(DATASET_DIR, file)

    missing_counts = None
    total_rows = 0

    # Read data in chunks to save RAM
    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE
    ):

        total_rows += len(chunk)

        # Count missing values in every column
        current_missing = chunk.isnull().sum()

        if missing_counts is None:
            missing_counts = current_missing
        else:
            missing_counts += current_missing

    # Display columns containing missing values
    print("\nColumns With Missing Values:")

    print(
        missing_counts[missing_counts > 0]
    )

    print("\nTotal Missing Values:",
          f"{missing_counts.sum():,}")

    print("Total Rows:", f"{total_rows:,}")

print("\nMISSING VALUES VERIFICATION COMPLETED!")

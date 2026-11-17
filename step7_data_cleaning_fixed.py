
# ============================================================
# PROJECT: AI-BASED NETWORK ATTACK DETECTION
# STEP 7: SAFE DATA CLEANING
#
# Purpose:
# Clean infinity values without overwriting existing files.
# ============================================================

import pandas as pd
import numpy as np
import os

INPUT_DIR = "datasets"
OUTPUT_DIR = "results"

CHUNK_SIZE = 200000

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [
    "NF-UNSW-NB15-v3.csv",
    "NF-CICIDS2018-v3.csv"
]

for file in files:

    print("\n" + "=" * 65)
    print("PROCESSING:", file)
    print("=" * 65)

    input_path = os.path.join(INPUT_DIR, file)

    # New output filename; old cleaned files remain untouched
    output_name = file.replace(".csv", "_cleaned_fixed.csv")
    output_path = os.path.join(OUTPUT_DIR, output_name)

    # Temporary file used until processing completes
    temp_path = output_path + ".tmp"

    if os.path.exists(temp_path):
        os.remove(temp_path)

    total_rows = 0
    total_inf_before = 0
    total_inf_after = 0
    total_missing = 0
    first_chunk = True

    try:

        for chunk in pd.read_csv(
            input_path,
            chunksize=CHUNK_SIZE
        ):

            total_rows += len(chunk)

            numeric_columns = chunk.select_dtypes(
                include=np.number
            ).columns

            total_inf_before += np.isinf(
                chunk[numeric_columns]
            ).sum().sum()

            # Replace infinity with missing values
            chunk[numeric_columns] = chunk[
                numeric_columns
            ].replace([np.inf, -np.inf], np.nan)

            total_inf_after += np.isinf(
                chunk[numeric_columns]
            ).sum().sum()

            total_missing += chunk.isna().sum().sum()

            # Write first chunk, then append remaining chunks
            chunk.to_csv(
                temp_path,
                mode="w" if first_chunk else "a",
                index=False,
                header=first_chunk
            )

            first_chunk = False

        # Confirm that a file was actually written
        if first_chunk:
            raise RuntimeError("No data chunks were processed.")

        # Verify saved row count before finalizing
        saved_rows = 0

        for saved_chunk in pd.read_csv(
            temp_path,
            chunksize=CHUNK_SIZE
        ):
            saved_rows += len(saved_chunk)

        if saved_rows != total_rows:
            raise RuntimeError(
                f"Row mismatch! Processed={total_rows}, Saved={saved_rows}"
            )

        # Rename temporary file after successful verification
        os.replace(temp_path, output_path)

        print("\nCLEANING SUCCESSFUL")
        print("Rows Processed:", f"{total_rows:,}")
        print("Rows Saved:", f"{saved_rows:,}")
        print("Infinity Before:", f"{total_inf_before:,}")
        print("Infinity After:", f"{total_inf_after:,}")
        print("Missing Values:", f"{total_missing:,}")
        print("Saved File:", output_path)

    except Exception as error:

        print("\nCLEANING FAILED:", error)
        print("Temporary file, if any, has not been finalized.")

print("\nSAFE CLEANING PROCESS FINISHED!")

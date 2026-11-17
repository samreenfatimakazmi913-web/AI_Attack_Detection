
import pandas as pd
from pathlib import Path
import os

# Define project directories.
BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "results" / "splits"
OUTPUT_DIR = BASE_DIR / "results" / "imputed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define UNSW training and testing files.
TRAIN_FILE = SPLIT_DIR / "NF-UNSW-NB15-v3_train.csv"
TEST_FILE = SPLIT_DIR / "NF-UNSW-NB15-v3_test.csv"

# Features containing missing values.
MISSING_FEATURES = [
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES"
]

# Read data in chunks to reduce RAM usage.
CHUNK_SIZE = 100_000

# Define output files.
TRAIN_OUTPUT = OUTPUT_DIR / "NF-UNSW-NB15-v3_train_imputed.csv"
TEST_OUTPUT = OUTPUT_DIR / "NF-UNSW-NB15-v3_test_imputed.csv"

TRAIN_TEMP = Path(str(TRAIN_OUTPUT) + ".tmp")
TEST_TEMP = Path(str(TEST_OUTPUT) + ".tmp")


# Check input files.
if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Testing file not found: {TEST_FILE}")


print("=" * 60)
print("STEP 13: MEDIAN IMPUTATION")
print("=" * 60)


# Calculate medians using training data only.
print("\nCalculating medians from training data...")

training_features = pd.read_csv(
    TRAIN_FILE,
    usecols=MISSING_FEATURES
)

training_medians = training_features.median()

del training_features

# Verify that medians were calculated successfully.
if training_medians.isna().any():
    raise ValueError("Median calculation failed!")

print("\nTraining Median Values:")

for feature, median_value in training_medians.items():
    print(f"{feature}: {median_value}")


# Function to impute missing values in chunks.
def impute_file(input_file, output_temp, medians):

    total_rows = 0
    missing_before = 0
    missing_after = 0
    first_chunk = True

    for chunk in pd.read_csv(
        input_file,
        chunksize=CHUNK_SIZE
    ):

        # Count missing values before imputation.
        missing_before += int(
            chunk[MISSING_FEATURES].isna().sum().sum()
        )

        # Fill missing values using training medians.
        chunk[MISSING_FEATURES] = chunk[
            MISSING_FEATURES
        ].fillna(medians)

        # Count missing values after imputation.
        missing_after += int(
            chunk[MISSING_FEATURES].isna().sum().sum()
        )

        # Save processed chunks.
        chunk.to_csv(
            output_temp,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk
        )

        first_chunk = False
        total_rows += len(chunk)

        print(f"Processed rows: {total_rows:,}")

    return total_rows, missing_before, missing_after


try:

    # Process training data.
    print("\nImputing training data...")

    train_rows, train_before, train_after = impute_file(
        TRAIN_FILE,
        TRAIN_TEMP,
        training_medians
    )

    # Process testing data using the same training medians.
    print("\nImputing testing data...")

    test_rows, test_before, test_after = impute_file(
        TEST_FILE,
        TEST_TEMP,
        training_medians
    )

    # Verify that no missing values remain.
    if train_after != 0:
        raise ValueError("Missing values remain in training data!")

    if test_after != 0:
        raise ValueError("Missing values remain in testing data!")

    # Verify training and testing row counts.
    expected_train_rows = 1_892_294
    expected_test_rows = 473_130

    if train_rows != expected_train_rows:
        raise ValueError("Training row count verification failed!")

    if test_rows != expected_test_rows:
        raise ValueError("Testing row count verification failed!")

    # Verify total records.
    if train_rows + test_rows != 2_365_424:
        raise ValueError("Total row count verification failed!")

    # Replace final files after successful verification.
    os.replace(TRAIN_TEMP, TRAIN_OUTPUT)
    os.replace(TEST_TEMP, TEST_OUTPUT)

except Exception:

    # Remove incomplete temporary files.
    if TRAIN_TEMP.exists():
        TRAIN_TEMP.unlink()

    if TEST_TEMP.exists():
        TEST_TEMP.unlink()

    raise


# Display final report.
print("\n" + "=" * 60)
print("MEDIAN IMPUTATION COMPLETED!")
print("=" * 60)

print("\nTRAINING DATA REPORT")
print("-" * 40)
print(f"Rows processed: {train_rows:,}")
print(f"Missing before: {train_before:,}")
print(f"Missing after: {train_after:,}")

print("\nTESTING DATA REPORT")
print("-" * 40)
print(f"Rows processed: {test_rows:,}")
print(f"Missing before: {test_before:,}")
print(f"Missing after: {test_after:,}")

print("\nROW COUNT VERIFICATION")
print("-" * 40)
print("Training rows verified!")
print("Testing rows verified!")
print("Total rows verified: 2,365,424")

print("\nTraining medians were used for both datasets.")

print(f"\nTraining output: {TRAIN_OUTPUT}")
print(f"Testing output: {TEST_OUTPUT}")

print("\nCICIDS2018 has zero missing values.")
print("Its original split files remain unchanged.")

print("\nALL CHECKS PASSED!")

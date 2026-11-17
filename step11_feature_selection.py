
import pandas as pd
from pathlib import Path

# Define project paths.
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

DATASETS = [
    RESULTS_DIR / "NF-UNSW-NB15-v3_cleaned_fixed.csv",
    RESULTS_DIR / "NF-CICIDS2018-v3_cleaned_fixed.csv"
]

# These are the 10 additional temporal features in NetFlow v3.
TEMPORAL_FEATURES = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV"
]

# Read column names from the first dataset.
columns = pd.read_csv(DATASETS[0], nrows=0).columns.tolist()

# Exclude target columns from model input features.
target_columns = ["Label", "Attack"]
all_features = [
    col for col in columns
    if col not in target_columns
]

# Check that all temporal features exist.
missing_temporal = [
    feature for feature in TEMPORAL_FEATURES
    if feature not in all_features
]

if missing_temporal:
    raise ValueError(
        f"Temporal features missing: {missing_temporal}"
    )

# The 43 old features are all input features
# except the 10 additional temporal features.
OLD_FEATURES = [
    feature for feature in all_features
    if feature not in TEMPORAL_FEATURES
]

# Verify the expected feature counts.
assert len(all_features) == 53
assert len(OLD_FEATURES) == 43
assert len(TEMPORAL_FEATURES) == 10

# Verify that both datasets have the same feature columns.
for dataset_path in DATASETS:
    dataset_columns = pd.read_csv(
        dataset_path, nrows=0
    ).columns.tolist()

    dataset_features = [
        col for col in dataset_columns
        if col not in target_columns
    ]

    if set(dataset_features) != set(all_features):
        raise ValueError(
            f"Feature mismatch in {dataset_path.name}"
        )

# Create a report showing both feature groups.
report_path = RESULTS_DIR / "feature_selection_summary.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("FEATURE SELECTION REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write("OLD FEATURE SET: 43 FEATURES\n")
    file.write("-" * 50 + "\n")

    for number, feature in enumerate(OLD_FEATURES, 1):
        file.write(f"{number}. {feature}\n")

    file.write("\nTEMPORAL FEATURE SET: 10 FEATURES\n")
    file.write("-" * 50 + "\n")

    for number, feature in enumerate(TEMPORAL_FEATURES, 1):
        file.write(f"{number}. {feature}\n")

    file.write("\nTOTAL INPUT FEATURES: 53\n")
    file.write("TARGET COLUMNS: Label, Attack\n")

print("FEATURE SELECTION COMPLETED!")
print("-" * 50)
print(f"Total input features: {len(all_features)}")
print(f"Old features: {len(OLD_FEATURES)}")
print(f"Additional temporal features: {len(TEMPORAL_FEATURES)}")
print(f"Target columns excluded: {target_columns}")
print(f"\nReport saved at: {report_path}")

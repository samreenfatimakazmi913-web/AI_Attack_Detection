
# ============================================================
# PROJECT: AI-BASED NETWORK ATTACK DETECTION
# STEP 10: FEATURE INSPECTION
#
# Objective:
# 1. Display all dataset columns.
# 2. Identify numerical and categorical columns.
# 3. Separate target columns from input features.
# 4. Compare both datasets.
# ============================================================

import pandas as pd

files = [
    "results/NF-UNSW-NB15-v3_cleaned_fixed.csv",
    "results/NF-CICIDS2018-v3_cleaned_fixed.csv"
]

for file in files:

    print("\n" + "=" * 70)
    print("FEATURE INSPECTION:", file)
    print("=" * 70)

    # Read only column names and data types
    data = pd.read_csv(file, nrows=1000)

    # Target columns are not input features
    target_columns = ["Label", "Attack"]

    feature_columns = [
        col for col in data.columns
        if col not in target_columns
    ]

    numerical_features = data[
        feature_columns
    ].select_dtypes(include="number").columns.tolist()

    categorical_features = data[
        feature_columns
    ].select_dtypes(exclude="number").columns.tolist()

    print("\nTotal Columns:", len(data.columns))
    print("Total Input Features:", len(feature_columns))
    print("Numerical Features:", len(numerical_features))
    print("Categorical Features:", len(categorical_features))

    print("\nALL INPUT FEATURES:")

    for index, column in enumerate(feature_columns, start=1):
        print(index, "-", column)

    print("\nNUMERICAL FEATURES:")
    for column in numerical_features:
        print("-", column)

    print("\nCATEGORICAL FEATURES:")
    for column in categorical_features:
        print("-", column)

    print("\nTARGET COLUMNS:", target_columns)

print("\nFEATURE INSPECTION COMPLETED!")

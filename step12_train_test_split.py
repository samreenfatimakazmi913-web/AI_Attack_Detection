
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Define project directories.
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SPLIT_DIR = RESULTS_DIR / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Define the cleaned datasets.
DATASETS = [
    RESULTS_DIR / "NF-UNSW-NB15-v3_cleaned_fixed.csv",
    RESULTS_DIR / "NF-CICIDS2018-v3_cleaned_fixed.csv"
]

# Define splitting settings.
TRAIN_RATIO = 0.80
CHUNK_SIZE = 100_000
RANDOM_SEED = 42

# Process each dataset separately.
for dataset_path in DATASETS:

    print("\n" + "=" * 60)
    print(f"PROCESSING: {dataset_path.name}")
    print("=" * 60)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    # Generate output file names.
    dataset_name = dataset_path.stem.replace(
        "_cleaned_fixed", ""
    )

    train_path = SPLIT_DIR / f"{dataset_name}_train.csv"
    test_path = SPLIT_DIR / f"{dataset_name}_test.csv"

    # Temporary files protect existing outputs until processing succeeds.
    train_temp = Path(str(train_path) + ".tmp")
    test_temp = Path(str(test_path) + ".tmp")

    # Create a reproducible random number generator.
    rng = np.random.default_rng(RANDOM_SEED)

    total_rows = 0
    train_rows = 0
    test_rows = 0

    label_counts = {
        "train": {},
        "test": {}
    }

    train_first_chunk = True
    test_first_chunk = True

    try:
        # Read the dataset in small chunks to save RAM.
        for chunk in pd.read_csv(
            dataset_path,
            chunksize=CHUNK_SIZE
        ):

            # Randomly assign each row to training or testing.
            random_values = rng.random(len(chunk))
            train_mask = random_values < TRAIN_RATIO

            train_chunk = chunk.loc[train_mask]
            test_chunk = chunk.loc[~train_mask]

            # Write training rows.
            if len(train_chunk) > 0:
                train_chunk.to_csv(
                    train_temp,
                    mode="w" if train_first_chunk else "a",
                    index=False,
                    header=train_first_chunk
                )
                train_first_chunk = False

            # Write testing rows.
            if len(test_chunk) > 0:
                test_chunk.to_csv(
                    test_temp,
                    mode="w" if test_first_chunk else "a",
                    index=False,
                    header=test_first_chunk
                )
                test_first_chunk = False

            # Update row counts.
            total_rows += len(chunk)
            train_rows += len(train_chunk)
            test_rows += len(test_chunk)

            # Track binary label distribution.
            for split_name, split_chunk in [
                ("train", train_chunk),
                ("test", test_chunk)
            ]:
                counts = split_chunk["Label"].value_counts()

                for label, count in counts.items():
                    key = str(label)
                    label_counts[split_name][key] = (
                        label_counts[split_name].get(key, 0)
                        + int(count)
                    )

            print(
                f"Processed: {total_rows:,} rows | "
                f"Training: {train_rows:,} | "
                f"Testing: {test_rows:,}"
            )

        # Verify that no rows were lost or duplicated.
        if train_rows + test_rows != total_rows:
            raise ValueError("Row count verification failed!")

        if train_first_chunk or test_first_chunk:
            raise ValueError("One of the split files is empty!")

        # Replace final files only after successful processing.
        os.replace(train_temp, train_path)
        os.replace(test_temp, test_path)

    except Exception:
        # Remove incomplete temporary files after an error.
        if train_temp.exists():
            train_temp.unlink()

        if test_temp.exists():
            test_temp.unlink()

        raise

    # Display the final split report.
    print("\nSPLIT COMPLETED!")
    print("-" * 50)
    print(f"Total rows: {total_rows:,}")
    print(
        f"Training rows: {train_rows:,} "
        f"({train_rows / total_rows * 100:.2f}%)"
    )
    print(
        f"Testing rows: {test_rows:,} "
        f"({test_rows / total_rows * 100:.2f}%)"
    )

    print("\nTraining Label Distribution:")
    print(label_counts["train"])

    print("\nTesting Label Distribution:")
    print(label_counts["test"])

    print(f"\nTraining file: {train_path}")
    print(f"Testing file: {test_path}")

print("\n" + "=" * 60)
print("ALL DATASETS SPLIT SUCCESSFULLY!")
print("=" * 60)

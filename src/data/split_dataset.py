import pandas as pd
from sklearn.model_selection import train_test_split
import os


'''Paths (relative to project root)'''

# This script lives inside src/data/, so we move two levels up
# to reach the main project directory.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # project root
# Full path to the extracted features CSV file
FEATURES_PATH = os.path.join(BASE_DIR, "data/processed/processedfeatures.csv")
# Folder where the split CSV files will be saved
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed")


'''Load features CSV'''

def load_features(path=FEATURES_PATH):
    # Make sure the features file exists before loading
    if not os.path.exists(path):
        raise FileNotFoundError(f"features.csv not found at {path}")
    return pd.read_csv(path)

'''Save train/val/test splits'''

def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, output_dir=OUTPUT_DIR):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine features and labels back into full dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Write each split to a CSV file
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # Print summary information
    print("Train/Validation/Test splits saved successfully.")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

'''Main function'''

def main():
    print(f"[INFO] Loading features from {FEATURES_PATH} ...")
    # Load full feature dataset
    df = load_features()

    # Make sure the label column exists
    if "label" not in df.columns:
        raise ValueError("ERROR: features.csv must contain a 'label' column.")

    # Separate features and labels
    X = df.drop(columns=["label"])
    y = df["label"]

    print("[INFO] Splitting into Train (70%), Val (10%), Test (20%) ...")

    # First split:
    #   70% → train
    #   30% → temp (will be further split into val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Second split:
    # temp (30%) → validation (10%) + test (20%)
    # test_size=2/3 ensures the final ratios are correct
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(2/3), random_state=42, stratify=y_temp
    )

    # Save all 3 splits
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test)

'''Entry point'''

if __name__ == "__main__":
    main()

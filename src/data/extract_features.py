import scipy.io
import numpy as np
import pandas as pd
import os

# Folder paths for raw input files and processed output
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def extract_features(signal):
    
    # Compute basic statistical features from the full signal
    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": np.sqrt(np.mean(signal**2)),
        "skewness": pd.Series(signal).skew(),
        "kurtosis": pd.Series(signal).kurt(),
        "peak": np.max(np.abs(signal)),
        "crest_factor": np.max(np.abs(signal)) / (np.sqrt(np.mean(signal**2)) + 1e-10),
    }
    return features

def get_signal_key(mat_keys):
    # Convert keys to a normal Python list
    keys = list(mat_keys)

    # Pick the correct channel from the .mat file
    # These are the common vibration channels in CWRU datasets
    for target in ["DE_time", "FE_time", "BA_time"]:
        for k in keys:
            if k.endswith(target):
                return k
    return None  # No recognizable vibration channel found

def main():
    rows = []

    # Loop through all .mat files in the raw data directory
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".mat"):
            continue

        file_path = os.path.join(RAW_DIR, filename)
        mat = scipy.io.loadmat(file_path)

        # Find which key contains the vibration signal
        signal_key = get_signal_key(mat.keys())
        if signal_key is None:
            print(f"No usable vibration signal found in {filename}")
            continue
        
        # Convert MATLAB array to a 1D NumPy array
        signal = mat[signal_key].squeeze()

        # Assign class label based on filename
        if "normal" in filename:
            label = "normal"
        elif "ball" in filename:
            label = "ball_fault"
        elif "inner" in filename:
            label = "inner_race_fault"
        elif "outer" in filename:
            label = "outer_race_fault"
        else:
            label = "unknown"

        # Extract statistical features from the entire signal
        features = extract_features(signal)
        features["file"] = filename
        features["label"] = label

        rows.append(features)

    # Save all extracted features into a CSV file
    df = pd.DataFrame(rows)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_DIR, "features.csv"), index=False)

    print("Feature extraction complete!")
    print(df.head())

if __name__ == "__main__":
    main()

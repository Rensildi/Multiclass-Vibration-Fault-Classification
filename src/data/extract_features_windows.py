import scipy.io as sio
import numpy as np
import pandas as pd
import os


'''Paths (relative to project root)'''

# Find the project root based on the file's location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # project root
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
OUTPUT_FILE = os.path.join(BASE_DIR, "data/processed/processedfeatures.csv")


'''Configuration'''

# Window size and overlap for feature extraction
WINDOW_SIZE = 2048
STEP_SIZE = 1024  # 50% overlap


'''Feature extraction functions'''

def extract_stats(signal):
    # Basic statistical features from the window
    s = pd.Series(signal)
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": np.sqrt(np.mean(signal**2)),
        "peak": np.max(np.abs(signal)),
        "skewness": s.skew(),
        "kurtosis": s.kurt()
    }

'''Load MATLAB vibration file'''

def load_signal(mat_file):
    # Load the .mat file and return the DE_time channel
    mat = sio.loadmat(mat_file)

    # Pick DE_time channel ==> for vibration datasets
    for key in mat.keys():
        if "DE_time" in key:
            return np.array(mat[key]).flatten()

    raise ValueError(f"No DE_time channel found in {mat_file}")


'''Create sliding windows'''

def sliding_windows(signal, size, step):
     # Cut the signal into overlapping windows
    windows = []
    for start in range(0, len(signal) - size, step):
        windows.append(signal[start:start + size])
    return windows


'''Main function'''

def main():
    # Make sure raw data folder exists
    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

    rows = []

    # Map each .mat file to a numeric label
    label_map = {
        "normal.mat": 0,
        "ball_fault_0.mat": 1,
        "inner_race_fault_0.mat": 2,
        "outer_race_fault_0.mat": 3
    }

    print(f"[INFO] Starting window-based feature extraction from {RAW_DIR} ...")

    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".mat"):
            continue

        label = label_map.get(filename)
        if label is None:
            print(f"[WARN] Unknown file: {filename}, skipping.")
            continue

        file_path = os.path.join(RAW_DIR, filename)
        print(f"[INFO] Loading {filename} ...")

        signal = load_signal(file_path)
        windows = sliding_windows(signal, WINDOW_SIZE, STEP_SIZE)

        print(f"  -> {len(windows)} windows extracted.")

        for w in windows:
            feats = extract_stats(w)
            feats["label"] = label
            rows.append(feats)

    # Create output folder if needed
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} samples to {OUTPUT_FILE}")


'''Entry point'''

if __name__ == "__main__":
    main()

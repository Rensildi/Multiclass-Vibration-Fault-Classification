import scipy.io
import os

# Folder containing the raw .mat vibration files
RAW_DIR = "data/raw"

# Loop through every file in the raw data directory
for f in os.listdir(RAW_DIR):
    # Only process MATLAB .mat files
    if f.endswith(".mat"):
        print("\nFile:", f)
        # Load the contents of the .mat file
        mat = scipy.io.loadmat(os.path.join(RAW_DIR, f))
        # Print all top-level keys stored in the .mat file
        # (Helps identify which key contains the vibration signal)
        print("Keys:", mat.keys())

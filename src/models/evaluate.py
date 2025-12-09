import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


'''Paths (relative to project root)'''

# This script lives inside src/models/, so we move two levels up
# to reach the main project directory.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # project root
# Folder containing processed CSVs
DATA_DIR = os.path.join(BASE_DIR, "data/processed")
# Folder containing saved trained models
MODEL_DIR = os.path.join(BASE_DIR, "models")
# Folder to save evaluation reports and confusion matrix images
OUTPUT_DIR = os.path.join(BASE_DIR, "reports")


'''Load test dataset'''

def load_data():
    # Construct path to test CSV
    test_path = os.path.join(DATA_DIR, "test.csv")
    
    # Make sure the file exists
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find test CSV in {DATA_DIR}")

    # Load CSV using pandas
    test = pd.read_csv(test_path)
    
    # Separate features and labels
    X_test = test.drop(columns=["label"])
    y_test = test["label"]

    return X_test, y_test


'''Load model & scaler'''

def load_model(model_name):
    # Construct paths to the model and optional scaler
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")

    # Ensure the model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the trained model
    model = joblib.load(model_path)
    # Load the scaler if it exists
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    return model, scaler

'''Evaluate model'''

def evaluate_model(model_name, X_test, y_test):
    print(f"\n[INFO] Evaluating {model_name}...")

    # Load model and optional scaler
    model, scaler = load_model(model_name)

    # Scale data if scaler exists
    if scaler:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Accuracy: {acc:.4f}")

    # Print classification report
    print("[INFO] Classification Report:")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    # Ensure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save classification report to a text file
    report_path = os.path.join(OUTPUT_DIR, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Compute and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    cm_path = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"Saved confusion matrix and report for {model_name}.")

    return acc

'''Main function'''

def main():
    print(f"[INFO] Loading test data from {DATA_DIR}...")
    X_test, y_test = load_data()

    # List of models to evaluate
    models = [
        "logistic_regression",
        "svm_rbf",
        "random_forest",
        "gradient_boosting",
        "mlp_neural_net"
    ]

    results = {}

    print("[INFO] Starting evaluation...")

    # Evaluate each model
    for model_name in models:
        acc = evaluate_model(model_name, X_test, y_test)
        results[model_name] = acc

    # Print final summary of accuracies
    print("\n============= FINAL TEST ACCURACY =============")
    for model_name, acc in results.items():
        print(f"{model_name:20s} | Acc: {acc:.4f}")
    print("================================================")

'''Entry point'''

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

'''Paths (relative to project root)'''

# This script is inside src/models/, so we move two levels up
# to get to the project root directory.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # project root
# Folder containing processed CSVs
DATA_DIR = os.path.join(BASE_DIR, "data/processed")
# Folder where trained models will be saved
MODEL_DIR = os.path.join(BASE_DIR, "models")

'''Load dataset'''

def load_data():
    # Paths to train and validation CSV files
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")

    # Check that files exist
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Could not find train or val CSV files in {DATA_DIR}")
    
    # Read CSVs
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    # Separate features and labels
    X_train = train.drop(columns=["label"])
    y_train = train["label"]

    X_val = val.drop(columns=["label"])
    y_val = val["label"]

    return X_train, X_val, y_train, y_val

'''Ensure model directory exists'''

def create_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)

'''Train & evaluate model'''

def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val, scaler=None):
    print(f"\n[INFO] Training {model_name}...")

    # Apply scaler if provided
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

    # Compute evaluation metrics
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")

    print(f"[RESULT] {model_name} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}.pkl"))
    
    # Save scaler if applicable
    if scaler:
        joblib.dump(scaler, os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl"))

    print(f"Saved {model_name} model.")

    return acc, f1

'''Main function'''

def main():
    print(f"[INFO] Loading dataset from {DATA_DIR}...")
    X_train, X_val, y_train, y_val = load_data()

    # Make sure model folder exists
    create_model_dir()

    results = []

    # Scaler for models that require it
    scaler = StandardScaler()

    # List of models to train
    models = [
        ("logistic_regression", LogisticRegression(max_iter=300)),
        ("svm_rbf", SVC(kernel="rbf", probability=True)),
        ("random_forest", RandomForestClassifier(n_estimators=150)),
        ("gradient_boosting", GradientBoostingClassifier()),
        ("mlp_neural_net", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500))
    ]

    print("[INFO] Starting model training...")

    # Train each model
    for name, model in models:
        # Linear models, SVM, MLP -> scale features
        if name in ["logistic_regression", "svm_rbf", "mlp_neural_net"]:
            acc, f1 = train_and_evaluate_model(model, name, X_train, y_train, X_val, y_val, scaler=scaler)
        else:
            # Tree-based models do not require scaling
            acc, f1 = train_and_evaluate_model(model, name, X_train, y_train, X_val, y_val, scaler=None)

        results.append((name, acc, f1))

    # Print summary of all models
    print("\n============= SUMMARY =============")
    for name, acc, f1 in results:
        print(f"{name:20s} | Acc: {acc:.4f} | F1: {f1:.4f}")
    print("===================================")

'''Entry point'''

if __name__ == "__main__":
    main()
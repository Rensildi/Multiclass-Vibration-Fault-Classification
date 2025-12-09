# 📦 Multiclass Vibration Fault Classification

A machine-learning pipeline for classifying bearing vibration faults
using the CWRU (Case Western Reserve University) dataset.\
This project includes preprocessing, feature extraction, model training,
evaluation, and experiment reproducibility.

## 🚀 Features

-   Loads and preprocesses the CWRU bearing vibration dataset\
-   Extracts statistical time-domain features\
-   Trains multiple ML classifiers (Logistic Regression, Random Forest,
    Gradient Boosting, SVM, MLP)\
-   Splits data into 70% train, 10% validation, 20% test\
-   Saves metrics, confusion matrices, and trained models\
-   Fully reproducible train.py pipeline\
-   Outputs results for LaTeX report generation

## 🗂 Project Structure

    project/
    │
    ├── data/
    │   ├── raw/                                                        # Original .mat CWRU files
    │   └── processed/                                                  # Preprocessed CSV feature sets
    │
    ├── models/                                                         # Saved .joblib/.pkl trained models
    ├── reports/                                                        # Stored .png confusion matrices
    │   ├── gradient_boosting_classification_report.txt
    │   ├── logistic_regression_classification_report.txt
    │   ├── mlp_neural_net_classification_report.txt             
    |   ├── random_forest_classification_report.txt
    |   └── svm_rbf_classification_report.txt
    │
    ├── src/
    │   ├── data/                                                       # Loads & preprocesses raw .mat files
            ├── extract_features_windows.py
            ├── inspect_mat.py
            └── split_dataset.py
    |   └── models/                                                     
    │       ├── evaluate.py                                             # Evaluating data         
    |       └── train.py                                                # Full training & evaluation pipeline
    │
    ├── README.md
    └── requirements.txt

## ⚙️ Requirements

Install dependencies:

    pip install -r requirements.txt

requirements.txt should include:

    numpy
    pandas
    scipy
    scikit-learn
    matplotlib
    joblib

## ▶️ How to Run the Project

### 1. Place raw CWRU .mat files

Download from the official site and place them into:

    data/raw/

### 2. Preprocess the dataset

    python src/data/extract_features_windows.py

### 3. Train all models

    python src/data/train.py

### 4. Evaluate the data
    python src/data/evaluate.py

This generates: - metrics.csv\
- confusion matrices\
- trained models\
- logs

## 📊 Output

-   Metrics table comparing accuracy & F1\
-   PNG confusion matrices\
-   Saved trained models

## 📧 Author

Rensildi Kalanxhi Master's in AI Algorithms & Systems

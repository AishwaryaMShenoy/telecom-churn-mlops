from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from features import create_features


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "telecom_customers.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"


# --------------------------------------------------
# Load configuration
# --------------------------------------------------

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

TRAIN_MONTHS = config["train_months"]


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

print("\nCreating features...")

df = create_features(
    df,
    TRAIN_MONTHS
)


# --------------------------------------------------
# Select model features
# --------------------------------------------------

FEATURE_COLUMNS = [
    "avg_arpu",
    "arpu_change",
    "avg_recharge_amt",
    "recharge_change",
    "avg_total_og_mou",
    "og_change",
    "avg_total_ic_mou",
    "avg_2g_usage",
    "avg_3g_usage",
    "tenure_months",
]


X = df[FEATURE_COLUMNS]

y = df["churn"]


print(f"Number of features: {len(FEATURE_COLUMNS)}")


# --------------------------------------------------
# Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# Train model
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 45)
print("MODEL EVALUATION")
print("=" * 45)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

model_path = (
    MODEL_DIR
    / "random_forest.joblib"
)

joblib.dump(
    {
        "model": model,
        "features": FEATURE_COLUMNS,
    },
    model_path
)

print("\nModel saved to:")
print(model_path)
from pathlib import Path

import mlflow
import mlflow.sklearn

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
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)

from xgboost import XGBClassifier

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


# --------------------------------------------------
# Load configuration
# --------------------------------------------------

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

TRAIN_MONTHS = config["train_months"]


# --------------------------------------------------
# Load and prepare data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = create_features(
    df,
    TRAIN_MONTHS
)


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


# --------------------------------------------------
# MLflow experiment
# --------------------------------------------------

mlflow.set_experiment(
    "telecom-churn-models"
)


# --------------------------------------------------
# Evaluation helper
# --------------------------------------------------

def evaluate_model(model):

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            y_probability
        )
    }

    return metrics


# ==================================================
# RUN 1 — BASELINE RANDOM FOREST
# ==================================================

with mlflow.start_run(
    run_name="Random Forest - Baseline"
):

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    metrics = evaluate_model(model)

    mlflow.log_params({
        "model_type": "Random Forest",
        "n_estimators": 200,
        "class_weight": "balanced",
        "random_state": 42,
        "train_months": str(TRAIN_MONTHS)
    })

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        model,
        "model"
    )

    print("\nBaseline Random Forest")
    print(metrics)


# ==================================================
# RUN 2 — TUNED RANDOM FOREST
# ==================================================

with mlflow.start_run(
    run_name="Random Forest - Tuned"
):

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    param_distributions = {
        "n_estimators": [100, 200, 300, 400],

        "max_depth": [
            None,
            5,
            10,
            15,
            20
        ],

        "min_samples_split": [
            2,
            5,
            10
        ],

        "min_samples_leaf": [
            1,
            2,
            4
        ],

        "max_features": [
            "sqrt",
            "log2",
            None
        ]
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    search.fit(
        X_train,
        y_train
    )

    best_model = search.best_estimator_

    metrics = evaluate_model(
        best_model
    )

    mlflow.log_params({
        "model_type": "Random Forest",
        "tuning": "RandomizedSearchCV",
        "cv": 3,
        "n_iter": 20,
        "scoring": "f1"
    })

    mlflow.log_params(
        search.best_params_
    )

    mlflow.log_metrics(metrics)

    mlflow.log_metric(
        "best_cv_f1",
        search.best_score_
    )

    mlflow.sklearn.log_model(
        best_model,
        "model"
    )

    print("\nTuned Random Forest")
    print(metrics)

    print(
        "Best parameters:",
        search.best_params_
    )


# ==================================================
# RUN 3 — XGBOOST
# ==================================================

with mlflow.start_run(
    run_name="XGBoost"
):

    negative = (
        y_train == 0
    ).sum()

    positive = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative / positive
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model
    )

    mlflow.log_params({
        "model_type": "XGBoost",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42
    })

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        model,
        "model"
    )

    print("\nXGBoost")
    print(metrics)


print("\n========================================")
print("ALL MLflow EXPERIMENTS COMPLETED")
print("========================================")
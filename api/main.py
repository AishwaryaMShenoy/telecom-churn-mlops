from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys

# Allow importing from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from features import create_features


# --------------------------------------------------
# Project paths
# --------------------------------------------------

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "telecom_customers.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_tuned.joblib"
)


# --------------------------------------------------
# Load data and model
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]

FEATURE_COLUMNS = model_data["features"]


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

TRAIN_MONTHS = [6, 7, 8]

df_features = create_features(
    df,
    TRAIN_MONTHS
)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Telecom Customer Service API",
    description=(
        "Customer lookup and ML-based churn "
        "decision support system."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class CustomerRequest(BaseModel):

    phone_number: str


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Telecom Customer Service API is running"
    }


# --------------------------------------------------
# Customer lookup and churn prediction
# --------------------------------------------------

@app.post("/customer")
def customer_lookup(
    request: CustomerRequest
):

    phone_number = request.phone_number

    customer_rows = df[
        df["phone_number"].astype(str)
        == phone_number
    ]

    if customer_rows.empty:

        raise HTTPException(
            status_code=404,
            detail="Phone number not found."
        )

    # Get original customer record
    customer_index = customer_rows.index[0]

    customer = df.loc[
        customer_index
    ]

    # Get engineered features
    customer_features = df_features.loc[
        customer_index
    ]

    X_customer = pd.DataFrame(
        [
            customer_features[FEATURE_COLUMNS]
        ]
    )

    # Model prediction
    prediction = model.predict(
        X_customer
    )[0]

    probability = model.predict_proba(
        X_customer
    )[0][1]

    probability = float(probability)

    # --------------------------------------------------
    # Determine risk level
    # --------------------------------------------------

    if probability >= 0.70:

        risk_level = "HIGH"

        recommendation = (
            "High churn risk. Consider offering "
            "a retention plan, personalized offer, "
            "or additional customer support."
        )

    elif probability >= 0.40:

        risk_level = "MEDIUM"

        recommendation = (
            "Moderate churn risk. Review the "
            "customer's recent usage and consider "
            "a suitable retention offer."
        )

    else:

        risk_level = "LOW"

        recommendation = (
            "Low churn risk. Continue normal "
            "customer service."
        )

    # --------------------------------------------------
    # Monthly trend information
    # --------------------------------------------------

    monthly_data = {}

    for month in [6, 7, 8]:

        monthly_data[str(month)] = {
            "arpu": float(
                customer.get(
                    f"arpu_{month}",
                    0
                )
            ),

            "recharge": float(
                customer.get(
                    f"total_rech_amt_{month}",
                    0
                )
            ),

            "outgoing_usage": float(
                customer.get(
                    f"total_og_mou_{month}",
                    0
                )
            ),

            "incoming_usage": float(
                customer.get(
                    f"total_ic_mou_{month}",
                    0
                )
            ),

            "2g_usage": float(
                customer.get(
                    f"vol_2g_mb_{month}",
                    0
                )
            ),

            "3g_usage": float(
                customer.get(
                    f"vol_3g_mb_{month}",
                    0
                )
            )
        }

    # --------------------------------------------------
    # Customer profile
    # --------------------------------------------------

    profile = {
        "customer_id": str(
            customer["customer_id"]
        ),

        "phone_number": str(
            customer["phone_number"]
        ),

        "tenure_months": int(
            customer["tenure_months"]
        ),

        "average_arpu": round(
            float(
                customer_features["avg_arpu"]
            ),
            2
        ),

        "average_recharge": round(
            float(
                customer_features["avg_recharge_amt"]
            ),
            2
        )
    }

    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    return {

        "customer_profile": profile,

        "monthly_trends": monthly_data,

        "churn_assessment": {

            "prediction": int(
                prediction
            ),

            "probability": round(
                probability,
                4
            ),

            "risk_level": risk_level
        },

        "recommended_action": recommendation
    }
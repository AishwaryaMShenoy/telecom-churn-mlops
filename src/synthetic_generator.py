import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


# --------------------------------------------------
# Load configuration
# --------------------------------------------------

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)


TRAIN_MONTHS = config["train_months"]
TARGET_MONTH = config["target_month"]
N_CUSTOMERS = config["n_customers"]
RANDOM_SEED = config["random_seed"]

ALL_MONTHS = TRAIN_MONTHS + [TARGET_MONTH]

rng = np.random.default_rng(RANDOM_SEED)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def positive_normal(mean, std, size):
    """
    Generate normally distributed values while
    preventing negative telecom measurements.
    """
    values = rng.normal(mean, std, size)
    return np.maximum(values, 0)


def generate_monthly_feature(
    base_value,
    trend,
    noise,
    month_index,
    size
):
    """
    Generate a feature whose value changes over time.

    base_value : customer's initial value
    trend      : customer's growth/decline rate
    noise      : random variation
    month_index: position in the monthly sequence
    """

    values = (
        base_value
        * (1 + trend * month_index)
        + rng.normal(0, noise, size)
    )

    return np.maximum(values, 0)


# --------------------------------------------------
# Generate customers
# --------------------------------------------------

def generate_dataset():

    print("Generating synthetic telecom dataset...")
    print(f"Customers : {N_CUSTOMERS}")
    print(f"Months    : {ALL_MONTHS}")

    customer_ids = [
        f"CUST_{i:06d}"
        for i in range(1, N_CUSTOMERS + 1)
    ]

    df = pd.DataFrame({
        "customer_id": customer_ids
    })

    # --------------------------------------------------
    # Customer-level characteristics
    # --------------------------------------------------

    # General spending tendency
    customer_value = rng.lognormal(
        mean=4.8,
        sigma=0.45,
        size=N_CUSTOMERS
    )

    # Customer-specific behavioural trend
    #
    # Positive = customer usage generally increasing
    # Negative = customer usage generally declining
    customer_trend = rng.normal(
        loc=0.0,
        scale=0.035,
        size=N_CUSTOMERS
    )

    # Customer tenure in months
    tenure = rng.integers(
        low=1,
        high=72,
        size=N_CUSTOMERS
    )

    df["tenure_months"] = tenure

    # --------------------------------------------------
    # Generate monthly telecom behaviour
    # --------------------------------------------------

    for month_index, month in enumerate(ALL_MONTHS):

        suffix = f"_{month}"

        # ----------------------------------------------
        # ARPU
        # ----------------------------------------------

        arpu = generate_monthly_feature(
            base_value=customer_value,
            trend=customer_trend,
            noise=20,
            month_index=month_index,
            size=N_CUSTOMERS
        )

        # ----------------------------------------------
        # Recharge amount
        # ----------------------------------------------

        recharge = (
            arpu
            * rng.uniform(0.8, 1.4, N_CUSTOMERS)
            + rng.normal(0, 30, N_CUSTOMERS)
        )

        recharge = np.maximum(recharge, 0)

        # ----------------------------------------------
        # Number of recharges
        # ----------------------------------------------

        recharge_num = np.maximum(
            np.round(
                recharge / rng.uniform(
                    80,
                    180,
                    N_CUSTOMERS
                )
            ),
            0
        ).astype(int)

        # ----------------------------------------------
        # Voice outgoing usage
        # ----------------------------------------------

        outgoing_voice = (
            arpu
            * rng.uniform(1.0, 2.0, N_CUSTOMERS)
            + rng.normal(0, 80, N_CUSTOMERS)
        )

        outgoing_voice = np.maximum(
            outgoing_voice,
            0
        )

        # ----------------------------------------------
        # Voice incoming usage
        # ----------------------------------------------

        incoming_voice = (
            outgoing_voice
            * rng.uniform(0.4, 1.2, N_CUSTOMERS)
            + rng.normal(0, 40, N_CUSTOMERS)
        )

        incoming_voice = np.maximum(
            incoming_voice,
            0
        )

        # ----------------------------------------------
        # 2G data usage
        # ----------------------------------------------

        usage_2g = (
            arpu
            * rng.uniform(1.0, 5.0, N_CUSTOMERS)
            + rng.normal(0, 300, N_CUSTOMERS)
        )

        usage_2g = np.maximum(
            usage_2g,
            0
        )

        # ----------------------------------------------
        # 3G data usage
        # ----------------------------------------------

        usage_3g = (
            arpu
            * rng.uniform(2.0, 8.0, N_CUSTOMERS)
            + rng.normal(0, 500, N_CUSTOMERS)
        )

        usage_3g = np.maximum(
            usage_3g,
            0
        )

        # ----------------------------------------------
        # Data recharges
        # ----------------------------------------------

        data_recharge = np.maximum(
            usage_3g / rng.uniform(
                50,
                120,
                N_CUSTOMERS
            ),
            0
        )

        # ----------------------------------------------
        # Complaints
        # ----------------------------------------------

        complaints = rng.poisson(
            lam=np.maximum(
                1.5 - arpu / 500,
                0.2
            )
        )

        # ----------------------------------------------
        # Store columns
        # ----------------------------------------------

        df[f"arpu{suffix}"] = np.round(
            arpu,
            2
        )

        df[f"total_rech_amt{suffix}"] = np.round(
            recharge,
            2
        )

        df[f"total_rech_num{suffix}"] = recharge_num

        df[f"total_og_mou{suffix}"] = np.round(
            outgoing_voice,
            2
        )

        df[f"total_ic_mou{suffix}"] = np.round(
            incoming_voice,
            2
        )

        df[f"vol_2g_mb{suffix}"] = np.round(
            usage_2g,
            2
        )

        df[f"vol_3g_mb{suffix}"] = np.round(
            usage_3g,
            2
        )

        df[f"total_rech_data{suffix}"] = np.round(
            data_recharge,
            2
        )

        df[f"complaints{suffix}"] = complaints

    # --------------------------------------------------
    # Generate churn risk
    # --------------------------------------------------

    # Customers with declining behaviour are more likely
    # to churn.
    decline_strength = np.maximum(
        -customer_trend,
        0
    )

    # Low-value customers have slightly higher risk.
    low_value_risk = np.maximum(
        1 - customer_value / 500,
        0
    )

    # Combine behavioural signals.
    churn_score = (
        0.5 * decline_strength
        + 0.3 * low_value_risk
        + 0.2 * rng.random(N_CUSTOMERS)
    )

    # Convert score into probability.
    churn_probability = np.clip(
        0.01 + churn_score * 0.45,
        0.01,
        0.15
    )

    churn_flag = (
        rng.random(N_CUSTOMERS)
        < churn_probability
    )

    # --------------------------------------------------
    # Apply churn to target month
    # --------------------------------------------------

    target_suffix = f"_{TARGET_MONTH}"

    churn_columns = [
        f"total_og_mou{target_suffix}",
        f"total_ic_mou{target_suffix}",
        f"vol_2g_mb{target_suffix}",
        f"vol_3g_mb{target_suffix}",
        f"total_rech_amt{target_suffix}",
        f"total_rech_num{target_suffix}",
        f"total_rech_data{target_suffix}"
    ]

    # Churned customers become inactive.
    df.loc[churn_flag, churn_columns] = 0

    # ----------------------------------------------
    # Create explicit churn label
    # ----------------------------------------------

    df["churn"] = (
        (
            df[f"vol_2g_mb{target_suffix}"] == 0
        )
        &
        (
            df[f"vol_3g_mb{target_suffix}"] == 0
        )
        &
        (
            df[f"total_ic_mou{target_suffix}"] == 0
        )
        &
        (
            df[f"total_og_mou{target_suffix}"] == 0
        )
    ).astype(int)

    return df


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    dataset = generate_dataset()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "telecom_customers.csv"
    )

    dataset.to_csv(
        output_path,
        index=False
    )

    print("\nDataset generated successfully!")
    print(f"Saved to: {output_path}")

    print("\nShape:")
    print(dataset.shape)

    print("\nChurn distribution:")
    print(dataset["churn"].value_counts())

    print("\nFirst 10 columns:")
    print(dataset.columns[:10].tolist())
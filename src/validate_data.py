from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "telecom_customers.csv"
)


df = pd.read_csv(DATA_PATH)


print("=" * 50)
print("DATA VALIDATION")
print("=" * 50)

# 1. Shape
print("\nShape:")
print(df.shape)


# 2. Duplicate customers
print("\nDuplicate customer IDs:")
print(df["customer_id"].duplicated().sum())


# 3. Missing values
print("\nTotal missing values:")
print(df.isna().sum().sum())


# 4. Churn distribution
print("\nChurn distribution:")
print(df["churn"].value_counts())

print("\nChurn percentage:")
print(
    df["churn"].value_counts(normalize=True) * 100
)


# 5. Check monthly columns
monthly_columns = [
    column
    for column in df.columns
    if "_6" in column
    or "_7" in column
    or "_8" in column
    or "_9" in column
]

print("\nNumber of monthly columns:")
print(len(monthly_columns))


# 6. Check churn definition
target_churn_check = (
    (df["vol_2g_mb_9"] == 0)
    &
    (df["vol_3g_mb_9"] == 0)
    &
    (df["total_ic_mou_9"] == 0)
    &
    (df["total_og_mou_9"] == 0)
).astype(int)


print("\nChurn label matches target-month rule:")
print(
    (target_churn_check == df["churn"]).all()
)


# 7. Basic statistics
print("\nBasic statistics:")
print(
    df[
        [
            "arpu_6",
            "arpu_7",
            "arpu_8",
            "arpu_9",
            "total_rech_amt_6",
            "total_rech_amt_7",
            "total_rech_amt_8",
            "total_rech_amt_9",
        ]
    ].describe().round(2)
)


print("\n" + "=" * 50)
print("VALIDATION COMPLETE")
print("=" * 50)
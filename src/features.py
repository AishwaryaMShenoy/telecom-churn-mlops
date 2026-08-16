import pandas as pd


def create_features(
    df: pd.DataFrame,
    train_months: list[int]
) -> pd.DataFrame:

    df = df.copy()

    # -------------------------------
    # ARPU
    # -------------------------------

    arpu_cols = [
        f"arpu_{month}"
        for month in train_months
    ]

    df["avg_arpu"] = df[arpu_cols].mean(axis=1)

    df["arpu_change"] = (
        df[f"arpu_{train_months[-1]}"]
        - df[f"arpu_{train_months[0]}"]
    )

    # -------------------------------
    # Recharge
    # -------------------------------

    recharge_cols = [
        f"total_rech_amt_{month}"
        for month in train_months
    ]

    df["avg_recharge_amt"] = (
        df[recharge_cols].mean(axis=1)
    )

    df["recharge_change"] = (
        df[f"total_rech_amt_{train_months[-1]}"]
        - df[f"total_rech_amt_{train_months[0]}"]
    )

    # -------------------------------
    # Outgoing usage
    # -------------------------------

    og_cols = [
        f"total_og_mou_{month}"
        for month in train_months
    ]

    df["avg_total_og_mou"] = (
        df[og_cols].mean(axis=1)
    )

    df["og_change"] = (
        df[f"total_og_mou_{train_months[-1]}"]
        - df[f"total_og_mou_{train_months[0]}"]
    )

    # -------------------------------
    # Incoming usage
    # -------------------------------

    ic_cols = [
        f"total_ic_mou_{month}"
        for month in train_months
    ]

    df["avg_total_ic_mou"] = (
        df[ic_cols].mean(axis=1)
    )

    # -------------------------------
    # 2G usage
    # -------------------------------

    usage_2g_cols = [
        f"vol_2g_mb_{month}"
        for month in train_months
    ]

    df["avg_2g_usage"] = (
        df[usage_2g_cols].mean(axis=1)
    )

    # -------------------------------
    # 3G usage
    # -------------------------------

    usage_3g_cols = [
        f"vol_3g_mb_{month}"
        for month in train_months
    ]

    df["avg_3g_usage"] = (
        df[usage_3g_cols].mean(axis=1)
    )

    return df
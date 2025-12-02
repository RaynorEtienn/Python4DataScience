import pandas as pd
import numpy as np


def label_churn(df, window_days=10):
    """
    Adds a 'churn' column to the dataframe.
    churn = 1 if the event occurred within 'window_days' before the user's Cancellation Confirmation.
    churn = 0 otherwise.
    """
    df = df.copy()
    # Identify Churn Timestamp
    churn_events = (
        df[df["page"] == "Cancellation Confirmation"]
        .groupby("userId")["ts"]
        .min()
        .reset_index()
    )
    churn_events.columns = ["userId", "churn_ts"]

    # Merge
    df = df.merge(churn_events, on="userId", how="left")

    # Define Window
    churn_window_delta = pd.Timedelta(days=window_days)

    # Create column
    df["churn"] = 0
    mask_churn_window = (
        df["churn_ts"].notna()
        & (df["ts"] >= (df["churn_ts"] - churn_window_delta))
        & (df["ts"] <= df["churn_ts"])
    )
    df.loc[mask_churn_window, "churn"] = 1

    return df


def extract_seasonality(df):
    """
    Extracts temporal features from 'ts':
    - hour
    - dayofweek
    - is_weekend
    """
    df = df.copy()
    df["hour"] = df["ts"].dt.hour
    df["dayofweek"] = df["ts"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df


def extract_user_attributes(df):
    """
    Extracts user-level attributes:
    - account_age_days
    - platform (from userAgent)
    - state (from location)
    """
    df = df.copy()
    # Account Age
    df["account_age_days"] = (df["ts"] - df["registration"]).dt.total_seconds() / (
        24 * 3600
    )

    # Platform (Simple extraction)
    # Assuming userAgent contains strings like 'Macintosh', 'Windows', 'Linux', 'iPhone'
    def get_platform(ua):
        if pd.isna(ua):
            return "Unknown"
        ua = str(ua).lower()
        if "macintosh" in ua or "mac os" in ua:
            return "Mac"
        elif "windows" in ua:
            return "Windows"
        elif "linux" in ua:
            return "Linux"
        elif "iphone" in ua or "ipad" in ua:
            return "iOS"
        elif "android" in ua:
            return "Android"
        else:
            return "Other"

    df["platform"] = df["userAgent"].apply(get_platform)

    # State (from location 'City, State')
    def get_state(loc):
        if pd.isna(loc):
            return "Unknown"
        parts = str(loc).split(",")
        if len(parts) > 1:
            return parts[1].strip()
        return "Unknown"

    df["state"] = df["location"].apply(get_state)

    return df


def extract_behavioral_flags(df):
    """
    Extracts behavioral flags:
    - thumbs_up
    - thumbs_down
    - roll_advert
    - downgrade (visited 'Submit Downgrade')
    """
    df = df.copy()
    df["thumbs_up"] = (df["page"] == "Thumbs Up").astype(int)
    df["thumbs_down"] = (df["page"] == "Thumbs Down").astype(int)
    df["roll_advert"] = (df["page"] == "Roll Advert").astype(int)
    df["downgrade"] = (df["page"] == "Submit Downgrade").astype(int)
    return df


def aggregate_session_metrics(df):
    """
    Aggregates metrics by userId (and potentially session):
    - error_count (status 404)
    - redirect_count (status 307)
    """
    df = df.copy()
    df["is_error"] = (df["status"] == 404).astype(int)
    df["is_redirect"] = (df["status"] == 307).astype(int)
    return df

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

    # Ensure datetime types
    # 'ts' is in milliseconds (int64) in the raw parquet files
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["registration"] = pd.to_datetime(df["registration"])

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


def aggregate_user_features(df):
    """
    Aggregates event-level data into a single row per user.
    Includes rolling window features (last 1, 3, 7 days).
    """
    df = df.copy()

    # 1. Identify Churn Target
    churn_users = df[df["page"] == "Cancellation Confirmation"]["userId"].unique()

    # 2. Calculate Reference Time (Last active time per user)
    # For churners, this is the churn time. For others, it's the end of their history.
    user_max_ts = (
        df.groupby("userId")["ts"]
        .max()
        .reset_index()
        .rename(columns={"ts": "last_active"})
    )
    df = df.merge(user_max_ts, on="userId")

    # 3. Calculate Time Delta for Rolling Windows
    df["days_from_end"] = (df["last_active"] - df["ts"]).dt.total_seconds() / (
        24 * 3600
    )

    # 4. Base Aggregation (Static & Total Counts)
    # Ensure we have the necessary columns from previous steps
    if "is_error" not in df.columns:
        df["is_error"] = (df["status"] == 404).astype(int)
    if "is_song" not in df.columns:
        df["is_song"] = (df["page"] == "NextSong").astype(int)
    if "is_thumbs_up" not in df.columns:
        df["is_thumbs_up"] = (df["page"] == "Thumbs Up").astype(int)
    if "is_thumbs_down" not in df.columns:
        df["is_thumbs_down"] = (df["page"] == "Thumbs Down").astype(int)
    if "is_ad" not in df.columns:
        df["is_ad"] = (df["page"] == "Roll Advert").astype(int)
    if "downgrade" not in df.columns:
        df["downgrade"] = (df["page"] == "Submit Downgrade").astype(int)

    g = df.groupby("userId")
    user_features = g.agg(
        {
            "gender": "first",
            "level": "last",  # Current level
            "registration": "first",
            "platform": "first",
            "state": "first",  # Kept for Frequency Encoding
            "last_active": "max",
            "is_thumbs_up": "sum",
            "is_thumbs_down": "sum",
            "is_ad": "sum",
            "is_error": "sum",
            "is_song": "sum",
            "length": "sum",  # Total listening time
            "downgrade": "max",  # Has ever downgraded
        }
    )

    # 5. Rolling Window Aggregations
    for days in [1, 3, 7, 14, 30]:
        # Filter events within the window
        window_mask = df["days_from_end"] <= days
        window_df = df[window_mask]

        # Group and aggregate
        window_agg = (
            window_df.groupby("userId")
            .agg(
                {
                    "is_song": "sum",
                    "is_error": "sum",
                    "length": "sum",
                    "artist": "nunique",  # Diversity
                    "song": "nunique",  # Diversity
                }
            )
            .rename(
                columns={
                    "is_song": f"songs_last_{days}d",
                    "is_error": f"errors_last_{days}d",
                    "length": f"listen_time_last_{days}d",
                    "artist": f"unique_artists_last_{days}d",
                    "song": f"unique_songs_last_{days}d",
                }
            )
        )

        # Merge back (fill NaN with 0 for users with no activity in window)
        user_features = user_features.join(window_agg).fillna(0)

    # 6. Derived Ratios & Features
    user_features["account_lifetime"] = (
        user_features["last_active"] - user_features["registration"]
    ).dt.total_seconds() / (24 * 3600)
    user_features["avg_songs_per_day"] = user_features["is_song"] / (
        user_features["account_lifetime"] + 1
    )
    user_features["thumbs_ratio"] = user_features["is_thumbs_up"] / (
        user_features["is_thumbs_up"] + user_features["is_thumbs_down"] + 1
    )
    # Clip errors_per_song to handle extreme outliers (e.g. users with 0 songs and many errors)
    user_features["errors_per_song"] = (
        user_features["is_error"] / (user_features["is_song"] + 1)
    ).clip(upper=5.0)

    # --- Advanced Features (Trends, Gaps, Session Quality) ---

    # A. Trends (7d vs 30d)
    # Avoid division by zero by adding small epsilon or checking for 0
    user_features["trend_songs_7d_vs_30d"] = user_features["songs_last_7d"] / (
        (user_features["songs_last_30d"] / 4) + 0.1
    )
    user_features["trend_listen_time_7d_vs_30d"] = user_features[
        "listen_time_last_7d"
    ] / ((user_features["listen_time_last_30d"] / 4) + 0.1)

    # B. Gap Analysis (Recency & Regularity)
    # Calculate average gap between sessions (approximate by days with activity)
    # We need to go back to event level for this, or approximate.
    # Approximation: Account Lifetime / Total Sessions (sessionId count)
    # Let's get total sessions first
    total_sessions = df.groupby("userId")["sessionId"].nunique()
    user_features = user_features.join(total_sessions.rename("total_sessions"))

    user_features["avg_days_between_sessions"] = (
        user_features["account_lifetime"] / user_features["total_sessions"]
    )
    # Recency is implicitly captured by 'days_from_end' logic if we had a global reference,
    # but here 'last_active' is per user.
    # However, we can calculate 'days_since_last_session' relative to the dataset end if needed,
    # but for churn prediction, the 'last_active' IS the churn point or max date.
    # A better metric might be: Did they have a long gap BEFORE their last session?
    # That's hard to get from just aggregations.
    # Let's stick to avg_days_between_sessions.

    # C. Session Quality
    user_features["avg_songs_per_session"] = (
        user_features["is_song"] / user_features["total_sessions"]
    )
    user_features["avg_session_duration"] = (
        user_features["length"] / user_features["total_sessions"]
    )

    # 7. Set Target
    user_features["target"] = 0
    user_features.loc[user_features.index.isin(churn_users), "target"] = 1

    # 8. Frequency Encoding for State
    # Calculate frequency of each state
    state_freq = user_features["state"].value_counts(normalize=True)
    # Map frequency to a new column
    user_features["state_freq"] = user_features["state"].map(state_freq)

    # 9. Cleanup for Modeling
    # Drop raw timestamps and high-cardinality categoricals (original state)
    cols_to_drop = ["registration", "last_active", "state"]
    user_features = user_features.drop(
        columns=[c for c in cols_to_drop if c in user_features.columns]
    )

    return user_features

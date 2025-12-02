import pandas as pd


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

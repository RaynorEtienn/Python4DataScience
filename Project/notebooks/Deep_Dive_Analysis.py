import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def analyze_churn_behavior_optimized():
    print("\n--- ANALYZING CHURN BEHAVIOR (OPTIMIZED) ---")
    # Load only necessary columns
    # Adjust path to be relative to where the script is run or absolute
    # Assuming script is run from project root, data is in Project/data
    # But the script is in Project/notebooks.
    # Let's use absolute paths or careful relative paths.

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir, "../../data/train.parquet"
    )  # This points to root/data/train.parquet?
    # Wait, the workspace structure says data/ is in root AND Project/data/?
    # Workspace info:
    # data/ (root)
    # Project/data/ (project)

    # The previous script used '../data/train.parquet' relative to 'Project/notebooks', so it meant 'Project/data/train.parquet'.
    # Let's try that.

    data_path = os.path.join(base_dir, "../data/train.parquet")
    print(f"Loading data from {data_path}...")

    train_df = pd.read_parquet(data_path, columns=["userId", "ts", "page"])

    print(f"Data loaded. Shape: {train_df.shape}")

    # Sort by user and time
    print("Sorting data...")
    train_df = train_df.sort_values(["userId", "ts"])

    # Calculate gap with previous event
    print("Calculating gaps...")
    # We only care about the gap for the cancellation event
    # Shift(1) gives the timestamp of the previous event
    train_df["prev_ts"] = train_df.groupby("userId")["ts"].shift(1)

    # Filter for Churn Events
    churn_events = train_df[train_df["page"] == "Cancellation Confirmation"].copy()

    # Calculate gap only for these events
    churn_events["gap_days"] = (churn_events["ts"] - churn_events["prev_ts"]) / (
        1000 * 60 * 60 * 24
    )

    print(f"Total Churn Events: {len(churn_events)}")

    gaps = churn_events["gap_days"]

    print(f"Avg Gap before Cancellation: {gaps.mean():.4f} days")
    print(f"Median Gap before Cancellation: {gaps.median():.4f} days")
    print(f"Max Gap before Cancellation: {gaps.max():.4f} days")

    print("\n--- HYPOTHESIS CHECK ---")
    print(
        "If Median Gap before Churn is LOW (e.g. < 0.1), it means users are active immediately before cancelling."
    )
    print(
        "This implies that 'High Inactivity' (e.g. > 5 days) is NOT a signal of Churn, but rather of 'Dormancy' (Non-Churn)."
    )


if __name__ == "__main__":
    analyze_churn_behavior_optimized()

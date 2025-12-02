import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, roc_auc_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.abspath(".."))

try:
    from src.features import (
        extract_user_attributes,
        aggregate_user_features,
        generate_training_data,
    )
except ImportError:
    sys.path.append(os.path.abspath("../.."))
    from src.features import (
        extract_user_attributes,
        aggregate_user_features,
        generate_training_data,
    )


def run_deep_diagnostic():
    print("========================================================")
    print("       DEEP DIAGNOSTIC: TRAIN vs TEST DISTRIBUTION      ")
    print("========================================================")

    # 1. Load Data
    print("\n[1] Loading Data...")
    train_path = "../data/train.parquet"
    test_path = "../data/test.parquet"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Data files not found.")
        return

    train_df_raw = pd.read_parquet(train_path)
    test_df_raw = pd.read_parquet(test_path)

    print(f"Train Raw Shape: {train_df_raw.shape}")
    print(f"Test Raw Shape:  {test_df_raw.shape}")

    # 2. Generate Features (Replicating Modeling.ipynb logic)
    print("\n[2] Generating Features...")

    # A. Train Features (Snapshot Approach)
    print("  -> Processing Train (Snapshot)...")
    train_df_raw = extract_user_attributes(train_df_raw)
    # Note: This uses the CURRENT generate_training_data from src/features.py
    # which hopefully includes the "Safe Gaps" fix.
    df_train = generate_training_data(train_df_raw)

    # B. Test Features (Global Cutoff Approach)
    print("  -> Processing Test (Global Cutoff)...")
    test_df_raw = extract_user_attributes(test_df_raw)
    global_max_ts = test_df_raw["ts"].max()
    print(f"     Test Global Cutoff: {global_max_ts}")

    test_snapshot_df = pd.DataFrame(
        {"userId": test_df_raw["userId"].unique(), "cutoff_ts": global_max_ts}
    )
    df_test = aggregate_user_features(test_df_raw, snapshot_df=test_snapshot_df)
    df_test = df_test.reset_index(level="cutoff_ts", drop=True)

    # Align Columns
    cols_to_drop = ["target", "userId", "cutoff_ts"]
    X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
    y_train = df_train["target"]

    # Ensure Test has same columns
    missing_cols = set(X_train.columns) - set(df_test.columns)
    for c in missing_cols:
        df_test[c] = 0
    X_test = df_test[X_train.columns]

    print(f"  -> Train Features Shape: {X_train.shape}")
    print(f"  -> Test Features Shape:  {X_test.shape}")

    # 3. Distribution Analysis (The Core Issue)
    print("\n[3] CRITICAL: Distribution Analysis")

    feature_of_interest = "days_since_last_session"

    print(f"\n--- Analyzing '{feature_of_interest}' ---")

    # Get stats for Train (Non-Churn), Train (Churn), and Test
    train_0 = X_train[y_train == 0][feature_of_interest]
    train_1 = X_train[y_train == 1][feature_of_interest]
    test_dist = X_test[feature_of_interest]

    stats_df = pd.DataFrame(
        {
            "Train (No Churn)": train_0.describe(),
            "Train (Churn)": train_1.describe(),
            "TEST SET": test_dist.describe(),
        }
    )
    print(stats_df)

    # Check overlap
    max_train_0 = train_0.max()
    mean_test = test_dist.mean()

    print(f"\nMax Gap in Train (No Churn): {max_train_0:.2f} days")
    print(f"Mean Gap in Test Set:        {mean_test:.2f} days")

    if mean_test > max_train_0:
        print(
            "\n[!!!] ALARM: The Test Set has significantly larger gaps than the Non-Churners in Train."
        )
        print(
            "      The model likely interprets ANY gap > {:.2f} as 100% CHURN.".format(
                max_train_0
            )
        )
    else:
        print("\n[OK] Gap distributions seem somewhat aligned.")

    # 4. Model Predictions Analysis
    print("\n[4] Model Prediction Analysis")
    model_path = "../models/stacking_model.joblib"

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)

        # Predict
        print("Predicting on Test Set...")
        try:
            test_probs = model.predict_proba(X_test)[:, 1]

            print("\nTest Probability Stats:")
            print(pd.Series(test_probs).describe())

            # Correlation between Gap and Prediction
            corr = np.corrcoef(X_test[feature_of_interest], test_probs)[0, 1]
            print(f"\nCorrelation (Gap vs Prediction): {corr:.4f}")

            if corr > 0.5:
                print(
                    "      -> Strong positive correlation. The model relies heavily on the Gap."
                )

            # Histogram
            # plt.figure(figsize=(10, 4))
            # plt.hist(test_probs, bins=20)
            # plt.title("Distribution of Test Predictions")
            # plt.savefig("test_preds_dist.png")
            # print("      (Histogram saved to test_preds_dist.png)")

        except Exception as e:
            print(f"Could not predict: {e}")
    else:
        print("Model file not found. Skipping prediction analysis.")

    print("\n========================================================")
    print("                 DIAGNOSTIC COMPLETE                    ")
    print("========================================================")


if __name__ == "__main__":
    run_deep_diagnostic()

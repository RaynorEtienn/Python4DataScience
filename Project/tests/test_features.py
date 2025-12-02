import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import label_churn


class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Create sample data
        # User 1: Churns at T=100. Events at T=80, T=90, T=95, T=100.
        # Window=10.
        # T=100 (Churn event) -> churn=1
        # T=95 (100-5) -> churn=1
        # T=90 (100-10) -> churn=1 (inclusive boundary usually)
        # T=80 (100-20) -> churn=0

        # User 2: No churn.

        base_time = pd.Timestamp("2018-11-20")

        self.df = pd.DataFrame(
            {
                "userId": ["1", "1", "1", "1", "2", "2"],
                "ts": [
                    base_time - pd.Timedelta(days=20),  # T-20
                    base_time - pd.Timedelta(days=10),  # T-10
                    base_time - pd.Timedelta(days=5),  # T-5
                    base_time,  # T=0 (Churn)
                    base_time - pd.Timedelta(days=5),
                    base_time,
                ],
                "page": [
                    "NextSong",
                    "NextSong",
                    "NextSong",
                    "Cancellation Confirmation",
                    "NextSong",
                    "NextSong",
                ],
            }
        )

    def test_label_churn(self):
        # Apply label_churn with 10 day window
        df_labeled = label_churn(self.df, window_days=10)

        # Check User 1
        user1 = df_labeled[df_labeled["userId"] == "1"].sort_values("ts")  # type: ignore

        # T-20 should be 0
        self.assertEqual(user1.iloc[0]["churn"], 0)

        # T-10 should be 1 (boundary)
        self.assertEqual(user1.iloc[1]["churn"], 1)

        # T-5 should be 1
        self.assertEqual(user1.iloc[2]["churn"], 1)

        # T=0 (Churn event) should be 1
        self.assertEqual(user1.iloc[3]["churn"], 1)

        # Check User 2 (No churn)
        user2 = df_labeled[df_labeled["userId"] == "2"]
        self.assertTrue((user2["churn"] == 0).all())


if __name__ == "__main__":
    unittest.main()

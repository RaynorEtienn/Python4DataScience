import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cleaning import cast_types


class TestCleaning(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.data = {
            "userId": [123, 456, "789"],
            "ts": [1540518498000, 1541672419000, 1541826321000],
            "registration": [1536599027000, 1536127339000, 1535117067000],
            "page": ["NextSong", "NextSong", "NextSong"],
        }
        self.df = pd.DataFrame(self.data)

    def test_cast_types(self):
        cleaned_df = cast_types(self.df)

        # Check userId is string
        self.assertTrue(pd.api.types.is_string_dtype(cleaned_df["userId"]))
        self.assertEqual(cleaned_df["userId"].iloc[0], "123")

        # Check ts is datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_df["ts"]))
        # Check specific conversion (1540518498000 -> 2018-10-26 01:48:18)
        self.assertEqual(cleaned_df["ts"].iloc[0].year, 2018)

        # Check registration is datetime
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(cleaned_df["registration"])
        )


if __name__ == "__main__":
    unittest.main()

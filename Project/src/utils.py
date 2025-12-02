import pandas as pd
import numpy as np


def load_data(filepath):
    """Loads data from a parquet file."""
    return pd.read_parquet(filepath)


def downsample_data(df, fraction=0.1, random_state=42):
    """
    Downsamples the dataframe by selecting a fraction of unique users.
    Returns the filtered dataframe containing all events for the sampled users.
    """
    if len(df) > 100000:
        print(f"Downsampling to {fraction*100}% of users...")
        unique_users = df["userId"].astype(str).unique()
        sampled_users = pd.Series(unique_users).sample(
            frac=fraction, random_state=random_state
        )
        df_sampled = df[df["userId"].astype(str).isin(sampled_users)]
        print(f"New Dataset Shape: {df_sampled.shape}")
        return df_sampled
    return df

import pandas as pd


def cast_types(df):
    """
    Casts columns to appropriate types:
    - userId -> string
    - ts -> datetime
    - registration -> datetime
    """
    df = df.copy()
    df["userId"] = df["userId"].astype(str)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")

    # Handle registration
    try:
        df["registration"] = pd.to_datetime(df["registration"], unit="ms")
    except:
        df["registration"] = pd.to_datetime(df["registration"])

    return df


def clean_data(df):
    """
    Performs data cleaning steps:
    - Casts types
    - Drops leakage columns ('auth')
    - Drops redundant columns ('time')
    - Drops PII/irrelevant columns ('firstName', 'lastName')
    """
    df = cast_types(df)

    # Drop leakage
    if "auth" in df.columns:
        df = df.drop(columns=["auth"])

    # Drop redundant
    if "time" in df.columns:
        df = df.drop(columns=["time"])

    # Drop PII/Irrelevant
    cols_to_drop = ["firstName", "lastName"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df


def check_ts_vs_time(df):
    """
    Checks if 'time' column exists and compares it with 'ts'.
    """
    if "time" in df.columns:
        # Convert time to datetime if not already
        try:
            df["time_dt"] = pd.to_datetime(df["time"])
            # Check difference
            diff = (df["ts"] - df["time_dt"]).abs().sum()
            print(f"Total difference between 'ts' and 'time': {diff}")
            if diff.total_seconds() == 0:
                print(
                    "Columns 'ts' and 'time' are identical (after conversion). You can drop 'time'."
                )
            else:
                print("Columns 'ts' and 'time' are different.")
        except Exception as e:
            print(f"Could not compare 'ts' and 'time': {e}")
    else:
        print("'time' column not found in dataframe. 'ts' is the primary time column.")

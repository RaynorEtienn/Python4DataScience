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

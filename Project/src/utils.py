import pathlib
import pandas as pd
import json
import os
import numpy as np


def load_data(filepath):
    """Loads data from a parquet file."""
    return pd.read_parquet(filepath)


def downsample_data(df, fraction=0.1, random_state=42):
    """Downsamples the dataframe by selecting a fraction of unique users."""
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


# --- JSON Variable Helpers ---


def load_variables(file_name: str):
    """Reads the JSON file and returns the dictionary of variables."""
    if not os.path.exists(file_name):
        return {}
    try:
        with open(file_name, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_variables(data, file_name: str):
    """Saves the dictionary to a file using json.dumps."""
    # Ensure the directory exists if it's in a subfolder
    os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)

    json_str = json.dumps(data, indent=4)
    with open(file_name, "w") as f:
        f.write(json_str)
    print(f"-> Saved data to {file_name}")


def update_variable(key, value, file_name: str):
    """Loads current data, updates one key, and saves it back."""
    data = load_variables(file_name)
    print(f"Updating '{key}' from {data.get(key, 'Not Set')} to {value}...")
    data[key] = value
    save_variables(data, file_name)


# --- 1. ABSOLUTE PATH SETUP (The Fix) ---
# Get the location of THIS file (src/utils.py)
HERE = pathlib.Path(__file__).resolve()
# Go up two levels: src -> Project Root
PROJECT_ROOT = HERE.parent.parent

# Define absolute paths so they never break
BASE_REPORT_DIR = PROJECT_ROOT / "experiment_reports/experiments/"
VARIABLES_FILE = BASE_REPORT_DIR / "../variables.json"

# Ensure the report directory exists immediately
BASE_REPORT_DIR.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Robust encoder for NumPy types and generic objects"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        try:
            return super().default(obj)
        except TypeError:
            # Fallback for Models, DataFrames, etc. -> convert to String
            return str(obj)


def _get_current_id():
    """Internal: Reads the current ID from variables.json"""
    # If file doesn't exist, create it inside experiment_reports/
    if not VARIABLES_FILE.exists():
        with open(VARIABLES_FILE, "w") as f:
            json.dump({"experience_number": 1}, f)
        return 1

    with open(VARIABLES_FILE, "r") as f:
        data = json.load(f)
        return data.get("experience_number", 1)


def _increment_id():
    """Internal: Adds +1 to the ID in variables.json"""
    current = _get_current_id()
    with open(VARIABLES_FILE, "w") as f:
        json.dump({"experience_number": current + 1}, f, indent=4)
    print(f"🔄 Auto-Update: Next experiment will be #{current + 1}")


def save_report(report_name, content, is_last_report=False):
    """
    Saves the JSON to experiment_reports/experiment_XXX/report_name.json
    """
    # 1. Auto-fetch ID
    exp_id = _get_current_id()

    # 2. Inject ID into your report dictionary automatically
    if isinstance(content, dict):
        content["experience_number"] = exp_id

    # 3. Create Folder path: Project/experiment_reports/experiment_001
    target_dir = BASE_REPORT_DIR / f"experiment_{exp_id:03d}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # 4. Handle Filename
    if not report_name.endswith(".json"):
        report_name += ".json"

    file_path = target_dir / report_name

    # 5. Save
    with open(file_path, "w") as f:
        json.dump(content, f, indent=4, cls=NumpyEncoder)

    print(f"✅ Report saved: {file_path}")

    # 6. Handle counter
    if is_last_report:
        _increment_id()

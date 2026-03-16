import joblib
import pandas as pd
from pathlib import Path

def save_dataframe(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def save_array(arr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(arr, path)

def save_metadata(metadata: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(metadata, path)
import pandas as pd
from src.config import TIME_COL, TRAIN_FRAC, VAL_FRAC, TEST_FRAC

def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(TIME_COL).reset_index(drop=True)

def temporal_split(df: pd.DataFrame):
    assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-8, "Split fractions must sum to 1."

    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df
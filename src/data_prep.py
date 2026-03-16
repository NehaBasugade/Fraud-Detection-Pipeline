from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from src.config import DATA_PROCESSED, TARGET_COL

DROP_COLS = [
    "TransactionID",
    "TransactionDT",
    "isFraud",
    "D7",
    "dist2",
]

CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
]

@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    numeric_cols: List[str]
    categorical_cols: List[str]


def load_split_data() -> SplitData:
    train_df = pd.read_parquet(DATA_PROCESSED / "train_df.parquet")
    val_df = pd.read_parquet(DATA_PROCESSED / "val_df.parquet")
    test_df = pd.read_parquet(DATA_PROCESSED / "test_df.parquet")


    y_train = train_df[TARGET_COL].copy()
    y_val = val_df[TARGET_COL].copy()
    y_test = test_df[TARGET_COL].copy()

    X_train = train_df.drop(columns=DROP_COLS, errors="ignore").copy()
    X_val = val_df.drop(columns=DROP_COLS, errors="ignore").copy()
    X_test = test_df.drop(columns=DROP_COLS, errors="ignore").copy()

    categorical_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def build_logreg_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        sparse_threshold=0.3,
    )

    return preprocessor


def build_lgbm_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def fit_transform_splits(preprocessor, split_data: SplitData):
    X_train_t = preprocessor.fit_transform(split_data.X_train)
    X_val_t = preprocessor.transform(split_data.X_val)
    X_test_t = preprocessor.transform(split_data.X_test)
    return X_train_t, X_val_t, X_test_t

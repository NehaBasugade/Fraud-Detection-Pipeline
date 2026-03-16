import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.config import TARGET_COL, ID_COL, TIME_COL, ARTIFACTS_SCALERS

def build_feature_subsets(df: pd.DataFrame):
    exclude_cols = {TARGET_COL, ID_COL, TIME_COL}

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    numeric_cols = [
        c for c in feature_cols
        if df[c].dtype.kind in {"i", "u", "f"}
    ]

    categorical_cols = [
        c for c in feature_cols
        if df[c].dtype == "object"
    ]

    return feature_cols, numeric_cols, categorical_cols

def fit_numeric_preprocessor(train_df: pd.DataFrame, numeric_cols):
    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_num = num_imputer.fit_transform(train_df[numeric_cols])
    X_train_num = scaler.fit_transform(X_train_num)

    return num_imputer, scaler, X_train_num

def transform_numeric(df: pd.DataFrame, numeric_cols, num_imputer, scaler):
    X_num = num_imputer.transform(df[numeric_cols])
    X_num = scaler.transform(X_num)
    return X_num

def save_preprocessors(num_imputer, scaler):
    ARTIFACTS_SCALERS.mkdir(parents=True, exist_ok=True)
    joblib.dump(num_imputer, ARTIFACTS_SCALERS / "num_imputer.joblib")
    joblib.dump(scaler, ARTIFACTS_SCALERS / "scaler.joblib")
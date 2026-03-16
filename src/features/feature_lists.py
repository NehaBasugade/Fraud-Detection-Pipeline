def get_initial_feature_columns(df):
    exclude_cols = {
        "isFraud",
        "TransactionID",
        "TransactionDT",
    }

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
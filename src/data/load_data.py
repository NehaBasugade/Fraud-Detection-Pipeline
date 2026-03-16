import pandas as pd
from src.config import TRAIN_TRANSACTION_PATH, TRAIN_IDENTITY_PATH

def load_raw_data(use_identity: bool = False) -> pd.DataFrame:
    tx = pd.read_csv(TRAIN_TRANSACTION_PATH)

    if use_identity:
        identity = pd.read_csv(TRAIN_IDENTITY_PATH)
        df = tx.merge(identity, on="TransactionID", how="left")
    else:
        df = tx.copy()

    return df
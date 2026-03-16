from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS_SPLITS = ARTIFACTS / "splits"
ARTIFACTS_SCALERS = ARTIFACTS / "scalers"
ARTIFACTS_REPORTS = ARTIFACTS / "reports"

TRAIN_TRANSACTION_PATH = DATA_RAW / "train_transaction.csv"
TRAIN_IDENTITY_PATH = DATA_RAW / "train_identity.csv"

RANDOM_STATE = 42
TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# -----------------------------
# Phase 2 – Modeling Config
# -----------------------------

DROP_HIGH_MISSING = [
    "D7",
    "dist2",
]

CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]

PHASE2_ARTIFACTS = ARTIFACTS / "phase2"

LOGREG_ARTIFACTS = PHASE2_ARTIFACTS / "logreg"
LGBM_ARTIFACTS = PHASE2_ARTIFACTS / "lightgbm"
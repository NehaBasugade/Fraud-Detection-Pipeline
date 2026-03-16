from __future__ import annotations

from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PHASE4_DATA_DIR = ROOT / "artifacts" / "phase4" / "data"
PHASE4_PRED_DIR = ROOT / "artifacts" / "phase4" / "predictions" / "baseline_gnn_card_only"
PHASE5_DATA_DIR = ROOT / "artifacts" / "phase5" / "score_only_features"


def main() -> None:
    PHASE5_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train = np.load(PHASE4_DATA_DIR / "X_train.npy")
    X_val = np.load(PHASE4_DATA_DIR / "X_val.npy")
    X_test = np.load(PHASE4_DATA_DIR / "X_test.npy")

    y_train = np.load(PHASE4_DATA_DIR / "y_train.npy")
    y_val = np.load(PHASE4_DATA_DIR / "y_val.npy")
    y_test = np.load(PHASE4_DATA_DIR / "y_test.npy")

    gnn_train = np.load(PHASE4_PRED_DIR / "train_oof_pred.npy").reshape(-1, 1)
    gnn_val = np.load(PHASE4_PRED_DIR / "val_pred.npy").reshape(-1, 1)
    gnn_test = np.load(PHASE4_PRED_DIR / "test_pred.npy").reshape(-1, 1)

    # Keep only train rows with leakage-safe OOF predictions.
    mask = np.isfinite(gnn_train.ravel())
    X_train_control = X_train[mask]
    y_train_filtered = y_train[mask]
    gnn_train = gnn_train[mask]

    X_train_hybrid = np.concatenate([X_train_control, gnn_train], axis=1)
    X_val_hybrid = np.concatenate([X_val, gnn_val], axis=1)
    X_test_hybrid = np.concatenate([X_test, gnn_test], axis=1)

    np.save(PHASE5_DATA_DIR / "X_train_control.npy", X_train_control)
    np.save(PHASE5_DATA_DIR / "X_val_control.npy", X_val)
    np.save(PHASE5_DATA_DIR / "X_test_control.npy", X_test)

    np.save(PHASE5_DATA_DIR / "X_train_hybrid.npy", X_train_hybrid)
    np.save(PHASE5_DATA_DIR / "X_val_hybrid.npy", X_val_hybrid)
    np.save(PHASE5_DATA_DIR / "X_test_hybrid.npy", X_test_hybrid)

    np.save(PHASE5_DATA_DIR / "y_train.npy", y_train_filtered)
    np.save(PHASE5_DATA_DIR / "y_val.npy", y_val)
    np.save(PHASE5_DATA_DIR / "y_test.npy", y_test)

    print("Saved score-only Phase 5 features to:")
    print(PHASE5_DATA_DIR)
    print(f"Train rows kept: {X_train_control.shape[0]} / {X_train.shape[0]}")
    print(f"Hybrid train feature dim: {X_train_hybrid.shape[1]}")


if __name__ == "__main__":
    main()
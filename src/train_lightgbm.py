import json
import time
import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np

from src.config import ARTIFACTS
from src.data_prep import load_split_data, build_lgbm_preprocessor, fit_transform_splits
from src.metrics import compute_classification_metrics


def save_predictions(path, y_true, scores):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_score": scores,
    })
    df.to_csv(path, index=False)


def main():
    split_data = load_split_data()

    preprocessor = build_lgbm_preprocessor(
        split_data.numeric_cols,
        split_data.categorical_cols,
    )

    X_train_t, X_val_t, X_test_t = fit_transform_splits(preprocessor, split_data)

    y_train = split_data.y_train.values
    y_val = split_data.y_val.values
    y_test = split_data.y_test.values

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = neg / pos

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(
        X_train_t,
        y_train,
        eval_set=[(X_val_t, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
            lgb.log_evaluation(period=50),
        ],
    )
    elapsed = time.time() - start

    print(f"Training done in {elapsed:.2f} seconds")
    print("Best iteration:", model.best_iteration_)
    print("Best score dict:", model.best_score_)

    val_scores = model.predict_proba(X_val_t)[:, 1]
    test_scores = model.predict_proba(X_test_t)[:, 1]

    val_metrics = compute_classification_metrics(y_val, val_scores)
    test_metrics = compute_classification_metrics(y_test, test_scores)

    out_dir = ARTIFACTS / "phase2" / "lightgbm"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, out_dir / "lgbm_preprocessor.joblib")
    joblib.dump(model, out_dir / "lgbm_model.joblib")

    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    save_predictions(out_dir / "val_predictions.csv", y_val, val_scores)
    save_predictions(out_dir / "test_predictions.csv", y_test, test_scores)

    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()

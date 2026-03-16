import json
import time
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

from src.data_prep import load_split_data, build_logreg_preprocessor, fit_transform_splits
from src.metrics import compute_classification_metrics
from src.config import ARTIFACTS


def save_predictions(path, y_true, y_score):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_score": y_score,
    })
    df.to_csv(path, index=False)


def main():
    split_data = load_split_data()

    print("Building preprocessor...")
    preprocessor = build_logreg_preprocessor(
        split_data.numeric_cols,
        split_data.categorical_cols,
    )

    print("Transforming train/val/test...")
    t0 = time.time()
    X_train, X_val, X_test = fit_transform_splits(preprocessor, split_data)
    t1 = time.time()
    print(f"Preprocessing done in {t1 - t0:.2f} seconds")
    print("Transformed shapes:", X_train.shape, X_val.shape, X_test.shape)

    model = LogisticRegression(
        solver="saga",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        verbose=1,
    )

    print("Training logistic regression...")
    t2 = time.time()
    model.fit(X_train, split_data.y_train)
    t3 = time.time()
    print(f"Training done in {t3 - t2:.2f} seconds")

    print("Scoring validation/test...")
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    val_metrics = compute_classification_metrics(split_data.y_val, val_scores)
    test_metrics = compute_classification_metrics(split_data.y_test, test_scores)

    out_dir = ARTIFACTS / "phase2" / "logreg"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, out_dir / "logreg_preprocessor.joblib")
    joblib.dump(model, out_dir / "logreg_model.joblib")

    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    save_predictions(out_dir / "val_predictions.csv", split_data.y_val, val_scores)
    save_predictions(out_dir / "test_predictions.csv", split_data.y_test, test_scores)

    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()

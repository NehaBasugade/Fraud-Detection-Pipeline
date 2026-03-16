import json
from pathlib import Path

from src.config import ARTIFACT_DIR


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    model_dirs = {
        "logreg": ARTIFACT_DIR / "phase2_logreg",
        "lightgbm": ARTIFACT_DIR / "phase2_lightgbm",
    }

    for model_name, model_dir in model_dirs.items():
        val_metrics = load_json(model_dir / "val_metrics.json")
        test_metrics = load_json(model_dir / "test_metrics.json")

        print(f"\n=== {model_name.upper()} ===")
        print("Validation:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.6f}")

        print("Test:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()

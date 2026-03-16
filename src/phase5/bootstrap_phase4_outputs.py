from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase4-report-dir",
        type=Path,
        default=Path("reports/phase4/baseline_gnn_card_only"),
    )
    parser.add_argument(
        "--phase4-artifact-dir",
        type=Path,
        default=Path("artifacts/phase4/baseline_gnn_card_only"),
    )
    parser.add_argument(
        "--phase4-pred-dir",
        type=Path,
        default=Path("reports/phase4/baseline_gnn_card_only/predictions"),
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def has_any(paths: list[Path]) -> bool:
    return any(p.exists() for p in paths)


def ensure_predictions_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main() -> None:
    args = parse_args()

    checkpoint_candidates = [
        args.phase4_artifact_dir / "best_model.pt",
        args.phase4_artifact_dir / "model.pt",
        args.phase4_artifact_dir / "checkpoint.pt",
        args.phase4_artifact_dir / "best.pt",
        args.phase4_artifact_dir / "gnn_model.pt",
        args.phase4_report_dir / "best_model.pt",
        args.phase4_report_dir / "model.pt",
        args.phase4_report_dir / "checkpoint.pt",
        args.phase4_report_dir / "best.pt",
        args.phase4_report_dir / "gnn_model.pt",
    ]

    prediction_candidates = [
        args.phase4_p_pred_dir / "train_predictions.parquet" if False else args.phase4_pred_dir / "train_predictions.parquet",
        args.phase4_pred_dir / "val_predictions.parquet",
        args.phase4_pred_dir / "test_predictions.parquet",
    ]

    if has_any(checkpoint_candidates) and all(p.exists() for p in prediction_candidates):
        print("[ok] existing Phase 4 checkpoint and predictions found")
        return

    run([sys.executable, "-m", "src.phase4.run_phase4"])

    args.phase4_pred_dir.mkdir(parents=True, exist_ok=True)

    copied_any = False
    copied_any |= copy_if_exists(
        args.phase4_report_dir / "train_predictions.parquet",
        args.phase4_pred_dir / "train_predictions.parquet",
    )
    copied_any |= copy_if_exists(
        args.phase4_report_dir / "val_predictions.parquet",
        args.phase4_pred_dir / "val_predictions.parquet",
    )
    copied_any |= copy_if_exists(
        args.phase4_report_dir / "test_predictions.parquet",
        args.phase4_pred_dir / "test_predictions.parquet",
    )

    if not has_any(checkpoint_candidates):
        raise FileNotFoundError(
            "Phase 4 rerun finished but still no checkpoint was saved. "
            "Patch src.phase4.phase4_train_gnn.py to save best_model.pt."
        )

    if not all(p.exists() for p in prediction_candidates):
        raise FileNotFoundError(
            "Phase 4 rerun finished but prediction parquet files are still missing. "
            "Patch src.phase4.phase4_infer.py to save train/val/test predictions."
        )


if __name__ == "__main__":
    main()
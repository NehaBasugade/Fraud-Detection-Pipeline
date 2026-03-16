from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.phase5.phase5_utils import ensure_dir, load_split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
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
        "--output-dir",
        type=Path,
        default=Path("reports/phase4/baseline_gnn_card_only/predictions"),
    )
    return parser.parse_args()


def find_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def find_checkpoint(artifact_dir: Path, report_dir: Path) -> Path:
    candidates = [
        artifact_dir / "best_model.pt",
        artifact_dir / "model.pt",
        artifact_dir / "checkpoint.pt",
        artifact_dir / "best.pt",
        artifact_dir / "gnn_model.pt",
        report_dir / "best_model.pt",
        report_dir / "model.pt",
        report_dir / "checkpoint.pt",
        report_dir / "best.pt",
        report_dir / "gnn_model.pt",
    ]
    ckpt = find_existing(candidates)
    if ckpt is None:
        searched = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "No saved Phase 4 checkpoint found.\n"
            f"Tried:\n{searched}\n\n"
            "Your Phase 4 run did not save a reusable model checkpoint."
        )
    return ckpt


def find_phase4_config(report_dir: Path, artifact_dir: Path) -> Path | None:
    candidates = [
        report_dir / "train_config.json",
        report_dir / "run_config.json",
        artifact_dir / "train_config.json",
        artifact_dir / "run_config.json",
    ]
    return find_existing(candidates)


def load_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def load_checkpoint_payload(ckpt_path: Path):
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    return {"state_dict": payload}


def resolve_model_class():
    from src.phase4.phase4_models import CardOnlyGNN

    return CardOnlyGNN


def build_model_from_payload(payload: dict, cfg: dict):
    ModelClass = resolve_model_class()

    model_kwargs = payload.get("model_kwargs")
    if model_kwargs is None:
        model_kwargs = cfg.get("model_kwargs")

    if model_kwargs is None:
        model_kwargs = {
            "num_numeric_features": cfg.get("num_numeric_features"),
            "categorical_cardinalities": cfg.get("categorical_cardinalities"),
            "embedding_dim": cfg.get("embedding_dim", 32),
            "hidden_dim": cfg.get("hidden_dim", 64),
            "dropout": cfg.get("dropout", 0.1),
        }

    missing = [k for k, v in model_kwargs.items() if v is None]
    if missing:
        raise ValueError(
            f"Could not reconstruct model kwargs. Missing keys={missing}. "
            f"Available cfg keys={list(cfg.keys())}"
        )

    model = ModelClass(**model_kwargs)

    state_dict = payload.get("state_dict", payload)
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def resolve_dataset_builders():
    from src.phase4.phase4_dataset import build_phase4_inference_dataset

    return build_phase4_inference_dataset


@torch.no_grad()
def infer_scores(model, dataset):
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)

    scores = []
    ids = []

    for batch in loader:
        if isinstance(batch, dict):
            x = batch
            batch_ids = batch.get("TransactionID") or batch.get("transaction_node_id")
        else:
            raise ValueError("Expected phase4 dataset to return dict batches.")

        tensor_batch = {}
        for k, v in x.items():
            if torch.is_tensor(v):
                tensor_batch[k] = v.to(device)
            else:
                tensor_batch[k] = v

        logits = model(tensor_batch)
        if logits.ndim > 1:
            logits = logits.squeeze(-1)

        probs = torch.sigmoid(logits).cpu().numpy()
        scores.append(probs)

        if batch_ids is None:
            raise ValueError(
                "Phase 4 dataset batch does not contain TransactionID or transaction_node_id."
            )

        if torch.is_tensor(batch_ids):
            batch_ids = batch_ids.cpu().numpy()

        ids.append(np.asarray(batch_ids))

    return np.concatenate(ids), np.concatenate(scores)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    ckpt_path = find_checkpoint(args.phase4_artifact_dir, args.phase4_report_dir)
    cfg_path = find_phase4_config(args.phase4_report_dir, args.phase4_artifact_dir)

    payload = load_checkpoint_payload(ckpt_path)
    cfg = load_config(cfg_path)
    model = build_model_from_payload(payload, cfg)

    split_frames = load_split_data(args.processed_dir)
    build_dataset = resolve_dataset_builders()

    for split, df in split_frames.items():
        dataset = build_dataset(
            split=split,
            processed_dir=args.processed_dir,
            phase4_artifact_dir=args.phase4_artifact_dir,
        )
        row_ids, scores = infer_scores(model, dataset)

        if "TransactionID" in df.columns:
            join_key = "TransactionID"
        elif "transaction_node_id" in df.columns:
            join_key = "transaction_node_id"
        else:
            raise ValueError("Processed split missing TransactionID and transaction_node_id.")

        out = pd.DataFrame(
            {
                join_key: row_ids,
                "gnn_score": scores,
            }
        )
        out.to_parquet(args.output_dir / f"{split}_predictions.parquet", index=False)
        print(f"[ok] wrote {split} predictions -> {args.output_dir / f'{split}_predictions.parquet'}")


if __name__ == "__main__":
    main()
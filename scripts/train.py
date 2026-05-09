#!/usr/bin/env python3
"""Train RFML models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from _bootstrap import delegate_to_conda_if_needed, delegated_env_name


delegate_to_conda_if_needed(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfml.training.trainer import RFMLTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> TrainerConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model_cfg = data["model"]
    train_cfg = data["training"]
    runtime_cfg = data.get("runtime", {})
    task_cfg = data.get("task", {})
    return TrainerConfig(
        task=str(task_cfg.get("name", "amc")),
        model_name=model_cfg["name"],
        num_classes=int(model_cfg["num_classes"]),
        modulation_num_classes=int(model_cfg["modulation_num_classes"]) if model_cfg.get("modulation_num_classes") is not None else None,
        sensing_num_classes=int(model_cfg["sensing_num_classes"]) if model_cfg.get("sensing_num_classes") is not None else None,
        epochs=int(train_cfg["epochs"]),
        batch_size=int(train_cfg["batch_size"]),
        lr=float(train_cfg["lr"]),
        optimizer=str(train_cfg["optimizer"]),
        weight_decay=float(train_cfg["weight_decay"]),
        amp=bool(train_cfg["amp"]),
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg["pin_memory"]),
        grad_clip=float(train_cfg["grad_clip"]) if train_cfg.get("grad_clip") is not None else None,
        early_stopping_patience=int(train_cfg["early_stopping_patience"]),
        device=str(runtime_cfg.get("device", "cuda")),
        dropout=float(model_cfg.get("dropout", 0.3)),
        classifier_hidden_dim=int(model_cfg.get("classifier_hidden_dim", 256)),
        channels=tuple(model_cfg.get("channels", [64, 128, 256])),
        kernel_sizes=tuple(model_cfg.get("kernel_sizes", [7, 5, 3])),
        save_every=int(train_cfg.get("save_every", 5)),
        scan_chunk_size=int(runtime_cfg.get("scan_chunk_size", 8192)),
        stft_n_fft=model_cfg.get("stft_n_fft"),
        stft_hop_length=model_cfg.get("stft_hop_length"),
        stft_window=model_cfg.get("stft_window"),
        stft_output=model_cfg.get("stft_output"),
        stft_backend=model_cfg.get("stft_backend"),
        stft_backbone=str(model_cfg.get("stft_backbone", "basic")),
        sensing_positive_ratio=task_cfg.get("positive_ratio"),
        sensing_noise_power=task_cfg.get("noise_power"),
        sensing_seed=int(task_cfg.get("seed", 42)),
        lambda_sensing=float(task_cfg.get("lambda_sensing", 1.0)),
        best_metric=str(train_cfg.get("best_metric", "val_loss")),
        low_snr_threshold=float(train_cfg["low_snr_threshold"]) if train_cfg.get("low_snr_threshold") is not None else None,
        low_snr_weight=float(train_cfg.get("low_snr_weight", 1.0)),
        low_snr_oversample_factor=float(train_cfg.get("low_snr_oversample_factor", 1.0)),
    )


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5).expanduser().resolve()
    split_path = Path(args.split).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    if not h5_path.exists():
        print(f"error: HDF5 file not found: {h5_path}")
        return 1
    if not split_path.exists():
        print(f"error: split file not found: {split_path}")
        return 1

    config = load_config(args.config)
    try:
        trainer = RFMLTrainer(
            config,
            h5_path=h5_path,
            split_path=split_path,
            out_dir=out_dir,
            resume_ckpt=args.resume,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    result = trainer.fit()
    print(f"delegated_conda_env: {delegated_env_name() or '<none>'}")
    print(f"out_dir: {out_dir}")
    print(f"best_ckpt: {result['best_ckpt']}")
    print(f"last_ckpt: {result['last_ckpt']}")
    print(f"num_epochs_run: {len(result['history'])}")
    if result["history"]:
        first = result["history"][0]
        last = result["history"][-1]
        print(f"first_train_loss: {first['train_loss']:.6f}")
        print(f"last_train_loss: {last['train_loss']:.6f}")
        print(f"best_val_loss: {min(row['val_loss'] for row in result['history']):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

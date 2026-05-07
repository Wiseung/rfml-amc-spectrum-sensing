#!/usr/bin/env python3
"""Evaluate trained RFML models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from _bootstrap import delegate_to_conda_if_needed, delegated_env_name


delegate_to_conda_if_needed(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfml.data.radioml2018 import RadioML2018Dataset
from rfml.data.splits import load_split_bundle, resolve_split_indices
from rfml.data.transforms import STFTTransform
from rfml.models.cnn1d import CNN1D
from rfml.models.resnet1d import build_resnet1d
from rfml.models.stft_cnn import STFTCNN
from rfml.training.metrics import (
    compute_accuracy,
    compute_accuracy_vs_snr,
    compute_classification_report,
    compute_confusion_matrix,
    plot_accuracy_vs_snr,
    plot_confusion_matrix,
)
from rfml.training.trainer import TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> TrainerConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model_cfg = data["model"]
    train_cfg = data["training"]
    runtime_cfg = data.get("runtime", {})
    return TrainerConfig(
        model_name=model_cfg["name"],
        num_classes=int(model_cfg["num_classes"]),
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
    )


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5).expanduser().resolve()
    split_path = Path(args.split).expanduser().resolve()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not h5_path.exists():
        print(f"error: HDF5 file not found: {h5_path}")
        return 1
    if not split_path.exists():
        print(f"error: split file not found: {split_path}")
        return 1
    if not ckpt_path.exists():
        print(f"error: checkpoint file not found: {ckpt_path}")
        return 1

    config = load_config(args.config)
    split_bundle = load_split_bundle(split_path)
    split_indices = resolve_split_indices(split_bundle, args.split_name)
    transform = None
    if config.model_name == "stft_cnn":
        transform = STFTTransform(
            n_fft=int(config.stft_n_fft or 128),
            hop_length=int(config.stft_hop_length or 32),
            window=str(config.stft_window or "hann"),
            output=str(config.stft_output or "log_power"),
            backend=str(config.stft_backend or "torch"),
        )
    dataset = RadioML2018Dataset(
        h5_path,
        split_indices=split_indices,
        class_names=split_bundle.class_names,
        scan_chunk_size=config.scan_chunk_size,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.model_name == "cnn1d":
        model = CNN1D(
            num_classes=config.num_classes,
            dropout=config.dropout,
            classifier_hidden_dim=config.classifier_hidden_dim,
            channels=config.channels,
            kernel_sizes=config.kernel_sizes,
        )
    elif config.model_name in {"resnet1d-small", "resnet1d-medium"}:
        model = build_resnet1d(
            config.model_name,
            num_classes=config.num_classes,
            dropout=config.dropout,
            classifier_hidden_dim=config.classifier_hidden_dim,
        )
    elif config.model_name == "stft_cnn":
        model = STFTCNN(
            num_classes=config.num_classes,
            channels=config.channels,
            dropout=config.dropout,
            classifier_hidden_dim=config.classifier_hidden_dim,
        )
    else:
        raise ValueError(f"Unsupported model_name: {config.model_name}")
    payload = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    ys = []
    preds = []
    snrs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["iq"].to(device, non_blocking=config.pin_memory)
            logits = model(x)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            ys.extend(batch["label"].cpu().numpy().tolist())
            preds.extend(batch_preds.tolist())
            snrs.extend(batch["snr"].cpu().numpy().astype(np.float32).tolist())

    y_true = np.asarray(ys, dtype=np.int64)
    y_pred = np.asarray(preds, dtype=np.int64)
    snr_array = np.asarray(snrs, dtype=np.float32)

    accuracy = compute_accuracy(y_true, y_pred)
    acc_vs_snr_df = compute_accuracy_vs_snr(y_true, y_pred, snr_array)
    report_text, report_dict = compute_classification_report(
        y_true,
        y_pred,
        class_names=split_bundle.class_names,
    )
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=config.num_classes)

    acc_vs_snr_path = plot_accuracy_vs_snr(acc_vs_snr_df, out_dir / "acc_vs_snr.png")
    confusion_path = plot_confusion_matrix(cm, out_dir / "confusion_matrix.png", class_names=split_bundle.class_names)
    acc_vs_snr_df.to_csv(out_dir / "accuracy_vs_snr.csv", index=False)
    pd.DataFrame(cm).to_csv(out_dir / "confusion_matrix.csv", index=False)
    (out_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (out_dir / "classification_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "overall_accuracy": accuracy,
                "num_samples": int(len(dataset)),
                "split": args.split_name,
                "ckpt": str(ckpt_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"delegated_conda_env: {delegated_env_name() or '<none>'}")
    print(f"overall_accuracy: {accuracy:.6f}")
    print(f"num_samples: {len(dataset)}")
    print(f"acc_vs_snr_png: {acc_vs_snr_path}")
    print(f"confusion_matrix_png: {confusion_path}")
    print(f"classification_report_txt: {out_dir / 'classification_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

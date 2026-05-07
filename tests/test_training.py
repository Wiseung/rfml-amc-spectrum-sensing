from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from rfml.data.splits import create_stratified_splits_from_h5, save_split_bundle
from rfml.training.metrics import compute_accuracy_vs_snr
from rfml.training.trainer import RFMLTrainer, TrainerConfig


def test_trainer_runs_and_saves_checkpoints(tmp_path: Path) -> None:
    h5_path = _build_training_h5(tmp_path / "train_radioml.h5")
    split_bundle = create_stratified_splits_from_h5(h5_path, seed=42)
    split_path = save_split_bundle(split_bundle, tmp_path / "splits.npz")
    out_dir = tmp_path / "run"

    config = TrainerConfig(
        task="amc",
        model_name="cnn1d",
        num_classes=4,
        epochs=5,
        batch_size=16,
        lr=1e-3,
        optimizer="adamw",
        weight_decay=1e-4,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_clip=1.0,
        early_stopping_patience=8,
        device="cpu",
        dropout=0.1,
        classifier_hidden_dim=64,
        channels=(16, 32, 64),
        kernel_sizes=(7, 5, 3),
        save_every=5,
        scan_chunk_size=256,
    )
    trainer = RFMLTrainer(
        config,
        h5_path=h5_path,
        split_path=split_path,
        out_dir=out_dir,
    )
    result = trainer.fit()

    history = result["history"]
    assert len(history) == 5
    assert history[-1]["train_loss"] <= history[0]["train_loss"]
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "last.pt").exists()
    assert (out_dir / "train_log.csv").exists()
    assert (out_dir / "history.json").exists()
    parsed = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    assert len(parsed) == 5


def test_accuracy_vs_snr_trend_on_synthetic_predictions() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1], dtype=np.int64)
    snrs = np.array([-20.0, -20.0, -20.0, -20.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    df = compute_accuracy_vs_snr(y_true, y_pred, snrs)
    low = float(df.loc[df["snr"] == -20.0, "accuracy"].iloc[0])
    high = float(df.loc[df["snr"] == 10.0, "accuracy"].iloc[0])
    assert high >= low


def _build_training_h5(path: Path) -> Path:
    class_defs = [
        ("BPSK", 1.0),
        ("QPSK", 2.0),
        ("QAM16", 3.0),
        ("QAM64", 4.0),
    ]
    snrs = [-20.0, -10.0, 0.0, 10.0]
    repeats = 6
    seq_len = 1024
    num_samples = len(class_defs) * len(snrs) * repeats

    x = np.zeros((num_samples, seq_len, 2), dtype=np.float32)
    y = np.zeros((num_samples, len(class_defs)), dtype=np.float32)
    z = np.zeros((num_samples, 1), dtype=np.float32)

    idx = 0
    for label, (_, freq_scale) in enumerate(class_defs):
        for snr in snrs:
            amp = 0.3 + (snr + 20.0) / 20.0
            noise_std = max(0.05, 0.35 - (snr + 20.0) / 120.0)
            for repeat in range(repeats):
                t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
                phase = repeat * 0.3
                i = amp * np.sin(2.0 * np.pi * freq_scale * t + phase)
                q = amp * np.cos(2.0 * np.pi * freq_scale * t + phase)
                rng = np.random.default_rng(1000 + idx)
                x[idx, :, 0] = i + rng.normal(0.0, noise_std, size=seq_len)
                x[idx, :, 1] = q + rng.normal(0.0, noise_std, size=seq_len)
                y[idx, label] = 1.0
                z[idx, 0] = snr
                idx += 1

    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("X", data=x)
        h5f.create_dataset("Y", data=y)
        h5f.create_dataset("Z", data=z)
        h5f.create_dataset(
            "classes",
            data=np.array([name.encode("utf-8") for name, _ in class_defs]),
        )
    return path

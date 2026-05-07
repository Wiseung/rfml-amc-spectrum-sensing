from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from rfml.data.spectrum_sensing import SpectrumSensingDataset
from rfml.data.splits import create_stratified_splits_from_h5, save_split_bundle
from rfml.eval.sensing_metrics import evaluate_sensing_predictions
from rfml.training.trainer import RFMLTrainer, TrainerConfig


def test_spectrum_sensing_dataset_balances_positive_negative(tmp_path: Path) -> None:
    h5_path = _build_sensing_h5(tmp_path / "sensing.h5")
    dataset = SpectrumSensingDataset(h5_path, positive_ratio=0.5, seed=42, max_samples=32)
    labels = [int(dataset[idx]["label"].item()) for idx in range(len(dataset))]
    assert len(dataset) == 32
    assert sum(labels) == 16
    assert labels.count(0) == 16
    noise_idx = next(idx for idx in range(len(dataset)) if int(dataset[idx]["label"].item()) == 0)
    noise_sample = dataset[noise_idx]
    assert tuple(noise_sample["iq"].shape) == (2, 1024)
    reference_idx = int(dataset.reference_positions[noise_idx])
    reference = dataset.signal_dataset[reference_idx]
    reference_power = float((reference["iq"][0].square() + reference["iq"][1].square()).mean().item())
    noise_power = float((noise_sample["iq"][0].square() + noise_sample["iq"][1].square()).mean().item())
    assert noise_power <= reference_power + 1e-4


def test_sensing_metrics_return_expected_columns() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    y_score = np.array([0.9, 0.8, 0.55, 0.45, 0.2, 0.1], dtype=np.float32)
    snrs = np.array([-10.0, 0.0, 10.0, -10.0, 0.0, 10.0], dtype=np.float32)
    result = evaluate_sensing_predictions(y_true, y_score, snrs)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.auc_value <= 1.0
    assert "pd_at_pfa_0p10" in result.metrics.columns
    assert "pd_at_pfa_0p05" in result.metrics.columns
    assert "pd_at_pfa_0p10" in result.pd_vs_snr.columns


def test_spectrum_sensing_trainer_runs(tmp_path: Path) -> None:
    h5_path = _build_sensing_h5(tmp_path / "sensing_train.h5")
    split_bundle = create_stratified_splits_from_h5(h5_path, seed=42)
    split_path = save_split_bundle(split_bundle, tmp_path / "splits.npz")
    config = TrainerConfig(
        task="spectrum_sensing",
        model_name="cnn1d",
        num_classes=2,
        epochs=2,
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
        save_every=2,
        scan_chunk_size=256,
        sensing_positive_ratio=0.5,
        sensing_noise_power=None,
        sensing_seed=42,
    )
    trainer = RFMLTrainer(
        config,
        h5_path=h5_path,
        split_path=split_path,
        out_dir=tmp_path / "run",
    )
    result = trainer.fit()
    assert len(result["history"]) == 2


def _build_sensing_h5(path: Path) -> Path:
    class_defs = [("BPSK", 1.0), ("QPSK", 2.0), ("QAM16", 3.0), ("QAM64", 4.0)]
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
            amp = 0.5 + (snr + 20.0) / 20.0
            noise_std = max(0.03, 0.30 - (snr + 20.0) / 120.0)
            for repeat in range(repeats):
                t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
                phase = repeat * 0.2
                rng = np.random.default_rng(9000 + idx)
                x[idx, :, 0] = amp * np.sin(2.0 * np.pi * freq_scale * t + phase) + rng.normal(0.0, noise_std, size=seq_len)
                x[idx, :, 1] = amp * np.cos(2.0 * np.pi * freq_scale * t + phase) + rng.normal(0.0, noise_std, size=seq_len)
                y[idx, label] = 1.0
                z[idx, 0] = snr
                idx += 1

    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("X", data=x)
        h5f.create_dataset("Y", data=y)
        h5f.create_dataset("Z", data=z)
        h5f.create_dataset("classes", data=np.array([name.encode("utf-8") for name, _ in class_defs]))
    return path

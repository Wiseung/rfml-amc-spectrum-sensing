from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from rfml.data.multitask import MultiTaskRadioMLDataset
from rfml.data.splits import create_stratified_splits_from_h5, save_split_bundle
from rfml.models.multitask import MultiTaskRFModel
from rfml.training.losses import compute_multitask_loss
from rfml.training.trainer import RFMLTrainer, TrainerConfig


def test_multitask_dataset_returns_expected_fields(tmp_path: Path) -> None:
    h5_path = _build_multitask_h5(tmp_path / "multitask.h5")
    dataset = MultiTaskRadioMLDataset(h5_path, positive_ratio=0.5, seed=42, max_samples=24)
    sample = dataset[0]
    assert "iq" in sample
    assert "modulation_label" in sample
    assert "sensing_label" in sample
    assert "mod_mask" in sample
    assert tuple(sample["iq"].shape) == (2, 1024)


def test_multitask_model_forward_shapes() -> None:
    model = MultiTaskRFModel(
        backbone="cnn1d",
        modulation_num_classes=24,
        sensing_num_classes=2,
        channels=(16, 32, 64),
        classifier_hidden_dim=64,
        dropout=0.1,
    )
    x = torch.randn(4, 2, 1024)
    outputs = model(x)
    assert tuple(outputs["modulation_logits"].shape) == (4, 24)
    assert tuple(outputs["sensing_logits"].shape) == (4, 2)


def test_multitask_loss_masks_noise_samples() -> None:
    modulation_logits = torch.randn(4, 4)
    modulation_targets = torch.tensor([0, 1, -1, -1])
    modulation_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    sensing_logits = torch.randn(4, 2)
    sensing_targets = torch.tensor([1, 1, 0, 0])
    loss_output = compute_multitask_loss(
        modulation_logits,
        modulation_targets,
        modulation_mask,
        sensing_logits,
        sensing_targets,
        lambda_sensing=0.5,
    )
    assert float(loss_output.total_loss.item()) > 0.0
    assert float(loss_output.modulation_loss.item()) > 0.0
    assert float(loss_output.sensing_loss.item()) > 0.0


def test_multitask_trainer_runs(tmp_path: Path) -> None:
    h5_path = _build_multitask_h5(tmp_path / "multitask_train.h5")
    split_bundle = create_stratified_splits_from_h5(h5_path, seed=42)
    split_path = save_split_bundle(split_bundle, tmp_path / "splits.npz")
    config = TrainerConfig(
        task="multitask",
        model_name="cnn1d",
        num_classes=4,
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
        modulation_num_classes=4,
        sensing_num_classes=2,
        sensing_positive_ratio=0.5,
        sensing_noise_power=None,
        sensing_seed=42,
        lambda_sensing=1.0,
        best_metric="val_acc",
    )
    trainer = RFMLTrainer(
        config,
        h5_path=h5_path,
        split_path=split_path,
        out_dir=tmp_path / "run",
    )
    result = trainer.fit()
    assert len(result["history"]) == 2
    assert (tmp_path / "run" / "best.pt").exists()


def _build_multitask_h5(path: Path) -> Path:
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
                rng = np.random.default_rng(9100 + idx)
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

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from rfml.data.radioml2018 import RadioML2018Dataset
from rfml.data.splits import create_stratified_splits_from_h5, save_split_bundle
from rfml.data.transforms import STFTTransform
from rfml.models.stft_cnn import STFTCNN
from rfml.training.trainer import RFMLTrainer, TrainerConfig


def test_stft_transform_returns_expected_shape() -> None:
    t = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    iq = np.stack([np.sin(2 * np.pi * 3 * t), np.cos(2 * np.pi * 3 * t)], axis=0)
    transform = STFTTransform(n_fft=128, hop_length=32, output="log_power", backend="torch")
    spec = transform(iq)
    assert spec.ndim == 3
    assert spec.shape[0] == 1
    assert torch.isfinite(spec).all()


def test_stft_transform_scipy_backend_returns_expected_shape() -> None:
    t = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    iq = np.stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 5 * t)], axis=0)
    transform = STFTTransform(n_fft=64, hop_length=16, output="power", backend="scipy")
    spec = transform(iq)
    assert spec.ndim == 3
    assert spec.shape[0] == 1
    assert torch.isfinite(spec).all()


def test_stft_transform_log_power_phase_returns_two_channels() -> None:
    t = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    iq = np.stack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 5 * t)], axis=0)
    transform = STFTTransform(n_fft=64, hop_length=16, output="log_power_phase", backend="torch")
    spec = transform(iq)
    assert spec.ndim == 3
    assert spec.shape[0] == 2
    assert transform.num_channels == 2
    assert torch.isfinite(spec).all()


def test_stft_transform_real_imag_returns_two_channels() -> None:
    t = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    iq = np.stack([np.sin(2 * np.pi * 7 * t), np.cos(2 * np.pi * 7 * t)], axis=0)
    transform = STFTTransform(n_fft=64, hop_length=16, output="real_imag", backend="torch")
    spec = transform(iq)
    assert spec.ndim == 3
    assert spec.shape[0] == 2
    assert transform.num_channels == 2
    assert torch.isfinite(spec).all()


def test_dataset_applies_stft_transform(tmp_path: Path) -> None:
    h5_path = _build_stft_h5(tmp_path / "stft_dataset.h5")
    dataset = RadioML2018Dataset(h5_path, transform=STFTTransform(n_fft=64, hop_length=16))
    sample = dataset[0]
    assert sample["iq"].ndim == 3
    assert sample["iq"].shape[0] == 1


def test_stft_cnn_forward_shape() -> None:
    model = STFTCNN(num_classes=24, channels=(16, 32, 64), classifier_hidden_dim=64, dropout=0.1)
    x = torch.randn(2, 1, 128, 33)
    y = model(x)
    assert tuple(y.shape) == (2, 24)


def test_stft_cnn_deep_forward_shape() -> None:
    model = STFTCNN(
        num_classes=24,
        channels=(16, 32, 64),
        classifier_hidden_dim=64,
        dropout=0.1,
        backbone="deep",
    )
    x = torch.randn(2, 1, 128, 33)
    y = model(x)
    assert tuple(y.shape) == (2, 24)


def test_stft_cnn_deeper_forward_shape() -> None:
    model = STFTCNN(
        num_classes=24,
        channels=(16, 32, 64),
        classifier_hidden_dim=64,
        dropout=0.1,
        backbone="deeper",
    )
    x = torch.randn(2, 1, 128, 65)
    y = model(x)
    assert tuple(y.shape) == (2, 24)


def test_stft_cnn_deeper_two_channel_forward_shape() -> None:
    model = STFTCNN(
        in_channels=2,
        num_classes=24,
        channels=(16, 32, 64),
        classifier_hidden_dim=64,
        dropout=0.1,
        backbone="deeper",
    )
    x = torch.randn(2, 2, 128, 65)
    y = model(x)
    assert tuple(y.shape) == (2, 24)


def test_stft_cnn_resnetplus_forward_shape() -> None:
    model = STFTCNN(
        in_channels=2,
        num_classes=24,
        channels=(24, 48, 96),
        classifier_hidden_dim=64,
        dropout=0.1,
        backbone="resnetplus",
    )
    x = torch.randn(2, 2, 128, 65)
    y = model(x)
    assert tuple(y.shape) == (2, 24)


def test_stft_trainer_runs(tmp_path: Path) -> None:
    h5_path = _build_stft_h5(tmp_path / "stft_train.h5")
    split_bundle = create_stratified_splits_from_h5(h5_path, seed=42)
    split_path = save_split_bundle(split_bundle, tmp_path / "splits.npz")
    config = TrainerConfig(
        task="amc",
        model_name="stft_cnn",
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
        stft_n_fft=64,
        stft_hop_length=16,
        stft_window="hann",
        stft_output="log_power_phase",
        stft_backend="torch",
        stft_backbone="deep",
    )
    trainer = RFMLTrainer(
        config,
        h5_path=h5_path,
        split_path=split_path,
        out_dir=tmp_path / "run",
    )
    result = trainer.fit()
    assert len(result["history"]) == 2


def test_stft_trainer_builds_low_snr_sampler(tmp_path: Path) -> None:
    h5_path = _build_stft_h5(tmp_path / "stft_sampler.h5")
    split_bundle = create_stratified_splits_from_h5(h5_path, seed=42)
    split_path = save_split_bundle(split_bundle, tmp_path / "splits_sampler.npz")
    config = TrainerConfig(
        task="amc",
        model_name="stft_cnn",
        num_classes=4,
        epochs=1,
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
        scan_chunk_size=64,
        stft_n_fft=64,
        stft_hop_length=16,
        stft_window="hann",
        stft_output="log_power_phase",
        stft_backend="torch",
        stft_backbone="deep",
        low_snr_threshold=0.0,
        low_snr_weight=2.0,
        low_snr_oversample_factor=3.0,
    )
    trainer = RFMLTrainer(
        config,
        h5_path=h5_path,
        split_path=split_path,
        out_dir=tmp_path / "sampler_run",
    )
    assert trainer.train_loader.sampler is not None


def _build_stft_h5(path: Path) -> Path:
    class_defs = [("BPSK", 1.0), ("QPSK", 2.0), ("QAM16", 3.0), ("QAM64", 4.0)]
    snrs = [-20.0, -10.0, 0.0, 10.0]
    repeats = 4
    seq_len = 1024
    num_samples = len(class_defs) * len(snrs) * repeats

    x = np.zeros((num_samples, seq_len, 2), dtype=np.float32)
    y = np.zeros((num_samples, len(class_defs)), dtype=np.float32)
    z = np.zeros((num_samples, 1), dtype=np.float32)

    idx = 0
    for label, (_, freq_scale) in enumerate(class_defs):
        for snr in snrs:
            amp = 0.8 + 0.2 * label
            noise_std = { -20.0: 0.9, -10.0: 0.5, 0.0: 0.2, 10.0: 0.05 }[snr]
            for repeat in range(repeats):
                t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
                phase = repeat * 0.3
                rng = np.random.default_rng(7000 + idx)
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

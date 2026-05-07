from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from rfml.baselines.common import extract_statistical_features
from rfml.baselines.energy_detection import evaluate_energy_detection, run_energy_detection_from_split
from rfml.baselines.sklearn_baselines import run_sklearn_baseline
from rfml.data.noise import estimate_noise_power_from_observation
from rfml.data.splits import create_stratified_splits_from_h5, save_split_bundle


def test_extract_statistical_features_has_expected_shape() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 1024, dtype=np.float32)
    iq = np.stack([np.sin(t), np.cos(t)], axis=0)
    features = extract_statistical_features(iq)
    assert features.shape == (20,)
    assert np.isfinite(features).all()


def test_sklearn_baseline_runs_on_tiny_dataset(tmp_path: Path) -> None:
    h5_path = _build_baseline_h5(tmp_path / "baseline_radioml.h5")
    split_bundle = create_stratified_splits_from_h5(h5_path, seed=42)
    split_path = save_split_bundle(split_bundle, tmp_path / "splits.npz")

    result = run_sklearn_baseline(
        h5_path,
        split_path,
        classifier_name="rf",
        max_train_samples=24,
        max_eval_samples=12,
        random_state=42,
    )

    assert result.feature_dim == 20
    assert result.train_size > 0
    assert result.eval_size > 0
    assert 0.0 <= result.train_accuracy <= 1.0
    assert 0.0 <= result.eval_accuracy <= 1.0
    assert not result.accuracy_vs_snr.empty


def test_energy_detection_runs_on_tiny_dataset(tmp_path: Path) -> None:
    h5_path = _build_baseline_h5(tmp_path / "baseline_radioml.h5")
    split_bundle = create_stratified_splits_from_h5(h5_path, seed=42)
    split_path = save_split_bundle(split_bundle, tmp_path / "splits.npz")

    result = run_energy_detection_from_split(
        h5_path,
        split_path,
        split_name="test",
        max_samples=12,
        seed=42,
        num_thresholds=64,
    )

    assert not result.metrics.empty
    assert not result.roc_curve.empty
    assert not result.pd_vs_snr.empty
    assert 0.0 <= result.auc_value <= 1.0


def test_evaluate_energy_detection_outputs_valid_tables() -> None:
    signal_energies = np.array([2.0, 2.1, 2.2, 2.3], dtype=np.float32)
    noise_energies = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    snrs = np.array([-10.0, 0.0, 10.0, 20.0], dtype=np.float32)

    result = evaluate_energy_detection(signal_energies, noise_energies, snrs, num_thresholds=16)
    assert result.metrics.loc[0, "best_pd"] >= result.metrics.loc[0, "best_pfa"]
    assert set(result.pd_vs_snr.columns) == {"snr", "num_samples", "pd", "pfa"}


def test_estimate_noise_power_from_observation_matches_total_power_model() -> None:
    total_power = 1.2
    snr_db = 0.0
    noise_power = estimate_noise_power_from_observation(total_power, snr_db)
    assert np.isclose(noise_power, 0.6)


def _build_baseline_h5(path: Path) -> Path:
    class_defs = [
        ("BPSK", 1.0),
        ("QPSK", 2.0),
        ("QAM16", 3.0),
        ("QAM64", 4.0),
    ]
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
            amp = 1.0 + (snr + 20.0) / 30.0
            for repeat in range(repeats):
                t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
                phase = repeat * 0.25
                x[idx, :, 0] = amp * np.sin(2.0 * np.pi * freq_scale * t + phase)
                x[idx, :, 1] = amp * np.cos(2.0 * np.pi * freq_scale * t + phase)
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

"""Common helpers for traditional RFML baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats

from rfml.data.radioml2018 import RadioML2018Dataset
from rfml.data.splits import SplitBundle, load_split_bundle, resolve_split_indices


@dataclass(frozen=True)
class FeatureBatch:
    features: np.ndarray
    labels: np.ndarray
    snrs: np.ndarray
    indices: np.ndarray

def load_split(split_path: str | Path) -> SplitBundle:
    return load_split_bundle(split_path)


def iter_dataset_samples(
    h5_path: str | Path,
    split_indices: np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
    snr_filter: Sequence[int | float] | None = None,
    max_samples: int | None = None,
    scan_chunk_size: int = 8192,
):
    dataset = RadioML2018Dataset(
        h5_path,
        split_indices=split_indices,
        class_names=class_names,
        snr_filter=snr_filter,
        max_samples=max_samples,
        scan_chunk_size=scan_chunk_size,
    )
    for local_index in range(len(dataset)):
        yield dataset[local_index]


def extract_statistical_features(iq: np.ndarray) -> np.ndarray:
    """Extract lightweight handcrafted features from a 2xT IQ sample."""

    i = iq[0]
    q = iq[1]
    complex_signal = i + 1j * q
    amplitude = np.abs(complex_signal)
    phase = np.unwrap(np.angle(complex_signal))
    phase_diff = np.diff(phase, prepend=phase[0])
    inst_frequency = phase_diff / (2.0 * np.pi)

    amplitude_skew, amplitude_kurt = _stable_skew_kurtosis(amplitude)
    phase_diff_skew, phase_diff_kurt = _stable_skew_kurtosis(phase_diff)
    inst_freq_skew, inst_freq_kurt = _stable_skew_kurtosis(inst_frequency)

    features = np.array(
        [
            float(np.mean(i)),
            float(np.std(i)),
            float(np.mean(q)),
            float(np.std(q)),
            float(np.mean(amplitude)),
            float(np.std(amplitude)),
            amplitude_skew,
            amplitude_kurt,
            float(np.mean(phase_diff)),
            float(np.std(phase_diff)),
            phase_diff_skew,
            phase_diff_kurt,
            float(np.mean(inst_frequency)),
            float(np.std(inst_frequency)),
            inst_freq_skew,
            inst_freq_kurt,
            float(np.mean(i * q)),
            float(np.mean(i**2 + q**2)),
            float(np.var(i)),
            float(np.var(q)),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _stable_skew_kurtosis(values: np.ndarray) -> tuple[float, float]:
    centered = values - np.mean(values)
    if float(np.max(np.abs(centered))) < 1e-10:
        return 0.0, 0.0
    skew_value = float(stats.skew(centered, bias=False))
    kurtosis_value = float(stats.kurtosis(centered, fisher=True, bias=False))
    return (
        0.0 if not np.isfinite(skew_value) else skew_value,
        0.0 if not np.isfinite(kurtosis_value) else kurtosis_value,
    )


def build_feature_batch(
    h5_path: str | Path,
    split_indices: np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
    snr_filter: Sequence[int | float] | None = None,
    max_samples: int | None = None,
    scan_chunk_size: int = 8192,
) -> FeatureBatch:
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    snrs: list[float] = []
    indices: list[int] = []

    for sample in iter_dataset_samples(
        h5_path,
        split_indices,
        class_names=class_names,
        snr_filter=snr_filter,
        max_samples=max_samples,
        scan_chunk_size=scan_chunk_size,
    ):
        iq = sample["iq"].numpy()
        feature_rows.append(extract_statistical_features(iq))
        labels.append(int(sample["label"].item()))
        snrs.append(float(sample["snr"].item()))
        indices.append(int(sample["index"]))

    if not feature_rows:
        raise ValueError("No samples available for feature extraction")

    return FeatureBatch(
        features=np.vstack(feature_rows).astype(np.float32, copy=False),
        labels=np.asarray(labels, dtype=np.int64),
        snrs=np.asarray(snrs, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.int64),
    )


def sample_matched_noise(
    signal_iq: np.ndarray,
    snr_db: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate complex AWGN with power matched to the requested SNR."""

    signal_power = float(np.mean(signal_iq[0] ** 2 + signal_iq[1] ** 2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-8)
    component_std = np.sqrt(noise_power / 2.0)
    noise = rng.normal(0.0, component_std, size=signal_iq.shape).astype(np.float32)
    return noise

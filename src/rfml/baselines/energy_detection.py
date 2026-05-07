"""Energy detection baseline for spectrum sensing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import auc

from rfml.baselines.common import iter_dataset_samples, load_split, resolve_split_indices, sample_matched_noise


@dataclass(frozen=True)
class EnergyDetectionResult:
    metrics: pd.DataFrame
    roc_curve: pd.DataFrame
    pd_vs_snr: pd.DataFrame
    auc_value: float
    num_signal_samples: int
    num_noise_samples: int


def compute_sample_energy(iq: np.ndarray) -> float:
    return float(np.mean(iq[0] ** 2 + iq[1] ** 2))


def build_sensing_arrays(
    h5_path: str | Path,
    split_path: str | Path,
    *,
    split_name: str = "test",
    snr_filter: Sequence[int | float] | None = None,
    max_samples: int | None = None,
    scan_chunk_size: int = 8192,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = load_split(split_path)
    split_indices = resolve_split_indices(bundle, split_name)
    rng = np.random.default_rng(seed)

    signal_energies: list[float] = []
    noise_energies: list[float] = []
    signal_snrs: list[float] = []

    for sample in iter_dataset_samples(
        h5_path,
        split_indices,
        class_names=bundle.class_names,
        snr_filter=snr_filter,
        max_samples=max_samples,
        scan_chunk_size=scan_chunk_size,
    ):
        iq = sample["iq"].numpy()
        snr = float(sample["snr"].item())
        signal_energies.append(compute_sample_energy(iq))
        noise_iq = sample_matched_noise(iq, snr, rng=rng)
        noise_energies.append(compute_sample_energy(noise_iq))
        signal_snrs.append(snr)

    if not signal_energies:
        raise ValueError("No sensing samples available")

    return (
        np.asarray(signal_energies, dtype=np.float32),
        np.asarray(noise_energies, dtype=np.float32),
        np.asarray(signal_snrs, dtype=np.float32),
    )


def evaluate_energy_detection(
    signal_energies: np.ndarray,
    noise_energies: np.ndarray,
    signal_snrs: np.ndarray,
    *,
    num_thresholds: int = 256,
) -> EnergyDetectionResult:
    if signal_energies.ndim != 1 or noise_energies.ndim != 1:
        raise ValueError("signal_energies and noise_energies must be 1D")
    if signal_energies.shape[0] != signal_snrs.shape[0]:
        raise ValueError("signal_snrs must align with signal_energies")

    all_energies = np.concatenate([signal_energies, noise_energies])
    thresholds = np.linspace(float(all_energies.min()), float(all_energies.max()), num=num_thresholds)

    roc_rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pd_value = float(np.mean(signal_energies >= threshold))
        pfa_value = float(np.mean(noise_energies >= threshold))
        roc_rows.append(
            {
                "threshold": float(threshold),
                "pd": pd_value,
                "pfa": pfa_value,
            }
        )

    roc_df = pd.DataFrame(roc_rows).sort_values("pfa")
    auc_value = float(auc(roc_df["pfa"].to_numpy(), roc_df["pd"].to_numpy()))

    best_row = roc_df.iloc[(roc_df["pd"] - (1.0 - roc_df["pfa"])).abs().argmin()]
    metrics_df = pd.DataFrame(
        [
            {
                "auc": auc_value,
                "best_threshold": float(best_row["threshold"]),
                "best_pd": float(best_row["pd"]),
                "best_pfa": float(best_row["pfa"]),
                "num_signal_samples": int(signal_energies.shape[0]),
                "num_noise_samples": int(noise_energies.shape[0]),
            }
        ]
    )

    pd_snr_rows: list[dict[str, float | int]] = []
    for snr in sorted(np.unique(signal_snrs).tolist()):
        mask = signal_snrs == snr
        threshold = float(best_row["threshold"])
        pd_value = float(np.mean(signal_energies[mask] >= threshold))
        pfa_value = float(np.mean(noise_energies[mask] >= threshold))
        pd_snr_rows.append(
            {
                "snr": float(snr),
                "num_samples": int(np.sum(mask)),
                "pd": pd_value,
                "pfa": pfa_value,
            }
        )

    return EnergyDetectionResult(
        metrics=metrics_df,
        roc_curve=roc_df,
        pd_vs_snr=pd.DataFrame(pd_snr_rows),
        auc_value=auc_value,
        num_signal_samples=int(signal_energies.shape[0]),
        num_noise_samples=int(noise_energies.shape[0]),
    )


def run_energy_detection_from_split(
    h5_path: str | Path,
    split_path: str | Path,
    *,
    split_name: str = "test",
    snr_filter: Sequence[int | float] | None = None,
    max_samples: int | None = None,
    scan_chunk_size: int = 8192,
    seed: int = 42,
    num_thresholds: int = 256,
) -> EnergyDetectionResult:
    signal_energies, noise_energies, signal_snrs = build_sensing_arrays(
        h5_path,
        split_path,
        split_name=split_name,
        snr_filter=snr_filter,
        max_samples=max_samples,
        scan_chunk_size=scan_chunk_size,
        seed=seed,
    )
    return evaluate_energy_detection(
        signal_energies,
        noise_energies,
        signal_snrs,
        num_thresholds=num_thresholds,
    )

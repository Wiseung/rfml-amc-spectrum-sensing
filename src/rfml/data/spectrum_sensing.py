"""Spectrum sensing dataset built on top of RadioML 2018.01A samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from rfml.data.noise import estimate_noise_power_from_observation
from rfml.data.radioml2018 import FilterLike, IndexLike, RadioML2018Dataset


@dataclass(frozen=True)
class SpectrumSensingDatasetInfo:
    num_signal_candidates: int
    num_positive_samples: int
    num_negative_samples: int
    positive_ratio: float
    noise_power: float | None


class SpectrumSensingDataset(Dataset[dict[str, Any]]):
    """Binary detection dataset for spectrum sensing.

    Positive samples are RadioML signal examples. Negative samples are synthetic
    AWGN noise-only sequences generated lazily from matched signal references.
    """

    def __init__(
        self,
        h5_path: str | Path,
        *,
        snr_filter: FilterLike | None = None,
        class_filter: FilterLike | None = None,
        max_samples: int | None = None,
        split_indices: IndexLike | None = None,
        class_names: Sequence[str] | None = None,
        scan_chunk_size: int = 8192,
        transform: Any | None = None,
        positive_ratio: float = 0.5,
        noise_power: float | None = None,
        seed: int = 42,
    ) -> None:
        if not (0.0 < positive_ratio < 1.0):
            raise ValueError("positive_ratio must lie strictly between 0 and 1")
        if noise_power is not None and noise_power <= 0.0:
            raise ValueError("noise_power must be positive when provided")
        if max_samples is not None and max_samples <= 1:
            raise ValueError("max_samples must be greater than 1 for binary sensing")

        self.h5_path = Path(h5_path).expanduser().resolve()
        self.positive_ratio = float(positive_ratio)
        self.noise_power = float(noise_power) if noise_power is not None else None
        self.seed = int(seed)
        self.transform = transform

        self.signal_dataset = RadioML2018Dataset(
            self.h5_path,
            snr_filter=snr_filter,
            class_filter=class_filter,
            max_samples=None,
            split_indices=split_indices,
            class_names=class_names,
            scan_chunk_size=scan_chunk_size,
            transform=None,
        )
        self.num_classes = 2
        self.class_names = ("noise", "signal")

        self._build_schedule(max_samples=max_samples)
        self.info = SpectrumSensingDatasetInfo(
            num_signal_candidates=len(self.signal_dataset),
            num_positive_samples=int(np.sum(self.labels == 1)),
            num_negative_samples=int(np.sum(self.labels == 0)),
            positive_ratio=self.positive_ratio,
            noise_power=self.noise_power,
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, item: int) -> dict[str, Any]:
        local_index = int(item)
        ref_position = int(self.reference_positions[local_index])
        target = int(self.labels[local_index])

        reference = self.signal_dataset[ref_position]
        reference_iq = reference["iq"]
        reference_snr = float(reference["snr"].item())
        reference_index = int(reference["index"])

        if target == 1:
            iq_tensor = reference_iq
        else:
            iq_tensor = self._generate_noise_sample(
                reference_iq,
                reference_snr,
                reference_index=reference_index,
                item_index=local_index,
            )

        if self.transform is not None:
            iq_tensor = self.transform(iq_tensor)

        return {
            "iq": iq_tensor,
            "label": torch.tensor(target, dtype=torch.long),
            "snr": torch.tensor(reference_snr, dtype=torch.float32),
            "index": reference_index,
        }

    def close(self) -> None:
        self.signal_dataset.close()

    def __del__(self) -> None:
        self.close()

    def describe(self) -> dict[str, Any]:
        return {
            "h5_path": str(self.h5_path),
            "num_signal_candidates": self.info.num_signal_candidates,
            "num_positive_samples": self.info.num_positive_samples,
            "num_negative_samples": self.info.num_negative_samples,
            "positive_ratio": self.info.positive_ratio,
            "noise_power": self.info.noise_power,
        }

    def _build_schedule(self, *, max_samples: int | None) -> None:
        num_candidates = len(self.signal_dataset)
        if num_candidates == 0:
            raise ValueError("No RadioML signal samples available for spectrum sensing")

        total_required = int(np.ceil(num_candidates / self.positive_ratio))
        total_samples = min(total_required, max_samples) if max_samples is not None else total_required
        total_samples = max(2, total_samples)

        num_positive = min(num_candidates, max(1, int(np.floor(total_samples * self.positive_ratio))))
        num_negative = total_samples - num_positive
        if num_negative == 0:
            num_negative = 1
            num_positive = max(1, total_samples - 1)

        rng = np.random.default_rng(self.seed)
        permutation = rng.permutation(num_candidates)

        positive_positions = permutation[:num_positive]
        negative_positions = self._repeat_positions(permutation, num_negative)

        labels = np.concatenate(
            [
                np.ones((num_positive,), dtype=np.int64),
                np.zeros((num_negative,), dtype=np.int64),
            ]
        )
        reference_positions = np.concatenate([positive_positions, negative_positions]).astype(np.int64, copy=False)
        order = rng.permutation(labels.shape[0])

        self.labels = labels[order]
        self.reference_positions = reference_positions[order]

    def _generate_noise_sample(
        self,
        reference_iq: torch.Tensor,
        reference_snr: float,
        *,
        reference_index: int,
        item_index: int,
    ) -> torch.Tensor:
        total_power = float(torch.mean(reference_iq[0].square() + reference_iq[1].square()).item())
        noise_power = self.noise_power
        if noise_power is None:
            noise_power = estimate_noise_power_from_observation(total_power, reference_snr)

        component_std = float(np.sqrt(noise_power / 2.0))
        rng_seed = self.seed * 1_000_003 + reference_index * 97 + item_index
        rng = np.random.default_rng(rng_seed)
        noise = rng.normal(0.0, component_std, size=tuple(reference_iq.shape)).astype(np.float32)
        return torch.from_numpy(noise)

    @staticmethod
    def _repeat_positions(positions: np.ndarray, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=np.int64)
        if count <= positions.shape[0]:
            return positions[:count].astype(np.int64, copy=False)
        repeats = int(np.ceil(count / positions.shape[0]))
        tiled = np.tile(positions, repeats)
        return tiled[:count].astype(np.int64, copy=False)

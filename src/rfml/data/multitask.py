"""Multi-task dataset for AMC and spectrum sensing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from rfml.data.radioml2018 import FilterLike, IndexLike, RadioML2018Dataset


@dataclass(frozen=True)
class MultiTaskDatasetInfo:
    num_signal_candidates: int
    num_signal_samples: int
    num_noise_samples: int
    positive_ratio: float
    noise_power: float | None


class MultiTaskRadioMLDataset(Dataset[dict[str, Any]]):
    """Joint dataset for modulation classification and signal detection.

    Signal samples contribute to both modulation and sensing tasks.
    Noise-only samples contribute only to the sensing task via `mod_mask=0`.
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
        self.modulation_num_classes = self.signal_dataset.num_classes
        self.modulation_class_names = tuple(self.signal_dataset.class_names or [str(i) for i in range(self.modulation_num_classes)])
        self.sensing_num_classes = 2

        self._build_schedule(max_samples=max_samples)
        self.info = MultiTaskDatasetInfo(
            num_signal_candidates=len(self.signal_dataset),
            num_signal_samples=int(np.sum(self.is_signal == 1)),
            num_noise_samples=int(np.sum(self.is_signal == 0)),
            positive_ratio=self.positive_ratio,
            noise_power=self.noise_power,
        )

    def __len__(self) -> int:
        return int(self.is_signal.shape[0])

    def __getitem__(self, item: int) -> dict[str, Any]:
        local_index = int(item)
        ref_position = int(self.reference_positions[local_index])
        signal_flag = int(self.is_signal[local_index])

        reference = self.signal_dataset[ref_position]
        reference_iq = reference["iq"]
        reference_label = int(reference["label"].item())
        reference_snr = float(reference["snr"].item())
        reference_index = int(reference["index"])

        if signal_flag == 1:
            iq_tensor = reference_iq
            modulation_label = reference_label
            modulation_mask = 1.0
        else:
            iq_tensor = self._generate_noise_sample(
                reference_iq,
                reference_snr,
                reference_index=reference_index,
                item_index=local_index,
            )
            modulation_label = -1
            modulation_mask = 0.0

        if self.transform is not None:
            iq_tensor = self.transform(iq_tensor)

        return {
            "iq": iq_tensor,
            "modulation_label": torch.tensor(modulation_label, dtype=torch.long),
            "sensing_label": torch.tensor(signal_flag, dtype=torch.long),
            "mod_mask": torch.tensor(modulation_mask, dtype=torch.float32),
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
            "num_signal_samples": self.info.num_signal_samples,
            "num_noise_samples": self.info.num_noise_samples,
            "positive_ratio": self.info.positive_ratio,
            "noise_power": self.info.noise_power,
            "modulation_num_classes": self.modulation_num_classes,
        }

    def _build_schedule(self, *, max_samples: int | None) -> None:
        num_candidates = len(self.signal_dataset)
        if num_candidates == 0:
            raise ValueError("No RadioML signal samples available for multitask training")

        total_required = int(np.ceil(num_candidates / self.positive_ratio))
        total_samples = min(total_required, max_samples) if max_samples is not None else total_required
        total_samples = max(2, total_samples)

        num_signal_samples = min(num_candidates, max(1, int(np.floor(total_samples * self.positive_ratio))))
        num_noise_samples = total_samples - num_signal_samples
        if num_noise_samples == 0:
            num_noise_samples = 1
            num_signal_samples = max(1, total_samples - 1)

        rng = np.random.default_rng(self.seed)
        permutation = rng.permutation(num_candidates)
        signal_positions = permutation[:num_signal_samples]
        noise_positions = self._repeat_positions(permutation, num_noise_samples)

        is_signal = np.concatenate(
            [
                np.ones((num_signal_samples,), dtype=np.int64),
                np.zeros((num_noise_samples,), dtype=np.int64),
            ]
        )
        reference_positions = np.concatenate([signal_positions, noise_positions]).astype(np.int64, copy=False)
        order = rng.permutation(is_signal.shape[0])

        self.is_signal = is_signal[order]
        self.reference_positions = reference_positions[order]

    def _generate_noise_sample(
        self,
        reference_iq: torch.Tensor,
        reference_snr: float,
        *,
        reference_index: int,
        item_index: int,
    ) -> torch.Tensor:
        signal_power = float(torch.mean(reference_iq[0].square() + reference_iq[1].square()).item())
        noise_power = self.noise_power
        if noise_power is None:
            snr_linear = 10.0 ** (reference_snr / 10.0)
            noise_power = signal_power / max(snr_linear, 1e-8)

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

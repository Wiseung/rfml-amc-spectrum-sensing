"""Lazy-loading dataset utilities for RadioML 2018.01A."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import ast
from contextlib import contextmanager
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


IndexLike = Sequence[int] | np.ndarray | Tensor
FilterLike = Sequence[int | float | str] | np.ndarray | set[int | float | str]


@dataclass(frozen=True)
class RadioMLDatasetInfo:
    """Summary metadata scanned from the HDF5 file."""

    num_examples: int
    num_classes: int
    sample_shape: tuple[int, ...]
    snr_values: tuple[float, ...]


def _normalize_numeric_filter(values: FilterLike | None) -> set[float] | None:
    if values is None:
        return None
    return {float(value) for value in values}


def _normalize_class_filter(
    class_filter: FilterLike | None,
    class_names: Sequence[str] | None,
) -> set[int] | None:
    if class_filter is None:
        return None

    normalized: set[int] = set()
    name_to_index = {name: idx for idx, name in enumerate(class_names or [])}
    for item in class_filter:
        if isinstance(item, str):
            if item not in name_to_index:
                raise ValueError(f"Unknown class name in class_filter: {item!r}")
            normalized.add(name_to_index[item])
        else:
            normalized.add(int(item))
    return normalized


def _as_numpy_indices(indices: IndexLike | None) -> np.ndarray | None:
    if indices is None:
        return None
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    array = np.asarray(indices, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError("split_indices must be a 1D sequence of integers")
    return array


def _decode_if_bytes(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


class RadioML2018Dataset(Dataset[dict[str, Any]]):
    """PyTorch dataset for RadioML 2018.01A with HDF5 lazy loading.

    This dataset intentionally avoids materializing the full `X` array in memory.
    It scans `Y` and `Z` once to build a filtered index list, and reads individual
    IQ samples on demand in ``__getitem__``.
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
        label_axis: int = -1,
        scan_chunk_size: int = 8192,
        transform: Any | None = None,
    ) -> None:
        self.h5_path = Path(h5_path).expanduser().resolve()
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self._h5_file: h5py.File | None = None
        self._x_ds = None
        self._y_ds = None
        self._z_ds = None
        self._h5_pid: int | None = None
        self.label_axis = label_axis
        self.transform = transform
        if scan_chunk_size <= 0:
            raise ValueError("scan_chunk_size must be a positive integer")
        self.scan_chunk_size = int(scan_chunk_size)

        self.class_names = list(class_names) if class_names is not None else None
        self.snr_filter = _normalize_numeric_filter(snr_filter)
        self._split_indices = _as_numpy_indices(split_indices)

        info = self.scan_info(self.h5_path, scan_chunk_size=self.scan_chunk_size)
        self.info = info
        if self.class_names is not None and len(self.class_names) != info.num_classes:
            raise ValueError(
                "class_names length does not match the dataset class count: "
                f"{len(self.class_names)} != {info.num_classes}"
            )

        self.class_filter = _normalize_class_filter(class_filter, self.class_names)
        self.indices = self._build_indices(max_samples=max_samples)
        self.num_classes = info.num_classes

    @staticmethod
    def scan_info(
        h5_path: str | Path,
        *,
        scan_chunk_size: int = 8192,
    ) -> RadioMLDatasetInfo:
        path = Path(h5_path).expanduser().resolve()
        with h5py.File(path, "r") as h5f:
            x_ds = h5f["X"]
            y_ds = h5f["Y"]
            z_ds = h5f["Z"]

            num_examples = int(x_ds.shape[0])
            num_classes = int(y_ds.shape[-1])
            sample_shape = tuple(int(dim) for dim in x_ds.shape[1:])

            snr_set: set[float] = set()
            for start in range(0, num_examples, scan_chunk_size):
                stop = min(start + scan_chunk_size, num_examples)
                snr_chunk = _flatten_z(np.asarray(z_ds[start:stop]))
                snr_set.update(float(value) for value in snr_chunk)

        return RadioMLDatasetInfo(
            num_examples=num_examples,
            num_classes=num_classes,
            sample_shape=sample_shape,
            snr_values=tuple(sorted(snr_set)),
        )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, Any]:
        real_index = int(self.indices[item])
        h5f = self._ensure_open()

        iq = np.asarray(h5f["X"][real_index], dtype=np.float32)
        if iq.ndim != 2:
            raise ValueError(f"Expected X sample to be 2D, got shape {iq.shape}")
        if iq.shape[-1] == 2:
            iq = np.transpose(iq, (1, 0))
        elif iq.shape[0] != 2:
            raise ValueError(f"Expected IQ sample with 2 channels, got shape {iq.shape}")

        y_row = np.asarray(h5f["Y"][real_index])
        label = int(np.argmax(y_row, axis=self.label_axis))

        z_row = h5f["Z"][real_index]
        snr = float(_flatten_scalar_z(z_row))

        iq_tensor = torch.from_numpy(np.ascontiguousarray(iq))
        if self.transform is not None:
            iq_tensor = self.transform(iq_tensor)

        return {
            "iq": iq_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "snr": torch.tensor(snr, dtype=torch.float32),
            "index": real_index,
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_x_ds"] = None
        state["_y_ds"] = None
        state["_z_ds"] = None
        state["_h5_pid"] = None
        return state

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._x_ds = None
            self._y_ds = None
            self._z_ds = None
            self._h5_pid = None

    def __del__(self) -> None:
        self.close()

    def snr_values(self) -> list[float]:
        return list(self.info.snr_values)

    def label_histogram(self) -> dict[int, int]:
        return self._histogram(key="label")

    def snr_histogram(self) -> dict[float, int]:
        return self._histogram(key="snr")

    def describe(self) -> dict[str, Any]:
        class_hist = self.label_histogram()
        if self.class_names is None:
            class_counts = {str(label): count for label, count in class_hist.items()}
        else:
            class_counts = {
                self.class_names[label]: count for label, count in class_hist.items()
            }

        return {
            "h5_path": str(self.h5_path),
            "num_selected_samples": len(self),
            "num_total_samples": self.info.num_examples,
            "num_classes": self.info.num_classes,
            "sample_shape": self.info.sample_shape,
            "snr_values": self.snr_values(),
            "class_counts": class_counts,
            "snr_counts": self.snr_histogram(),
        }

    def _build_indices(self, *, max_samples: int | None) -> np.ndarray:
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")

        with h5py.File(self.h5_path, "r") as h5f:
            total_examples = int(h5f["X"].shape[0])
            candidate_indices = (
                self._split_indices.copy()
                if self._split_indices is not None
                else np.arange(total_examples, dtype=np.int64)
            )

            if candidate_indices.size == 0:
                return candidate_indices

            label_filter = (
                np.fromiter(sorted(self.class_filter), dtype=np.int64)
                if self.class_filter is not None
                else None
            )
            snr_filter = (
                np.fromiter(sorted(self.snr_filter), dtype=np.float32)
                if self.snr_filter is not None
                else None
            )

            selected_chunks: list[np.ndarray] = []
            selected_total = 0
            for chunk_indices in self._iter_index_chunks(candidate_indices):
                mask = np.ones(chunk_indices.shape[0], dtype=bool)

                if label_filter is not None:
                    y_rows = np.asarray(h5f["Y"][chunk_indices])
                    labels = np.argmax(y_rows, axis=1).astype(np.int64, copy=False)
                    mask &= np.isin(labels, label_filter)

                if snr_filter is not None:
                    z_rows = np.asarray(h5f["Z"][chunk_indices])
                    snr_values = _flatten_z(z_rows)
                    mask &= np.isin(snr_values, snr_filter)

                selected_chunk = chunk_indices[mask]
                if selected_chunk.size == 0:
                    continue

                if max_samples is not None:
                    remaining = max_samples - selected_total
                    if remaining <= 0:
                        break
                    selected_chunk = selected_chunk[:remaining]

                selected_chunks.append(selected_chunk.astype(np.int64, copy=False))
                selected_total += int(selected_chunk.shape[0])

                if max_samples is not None and selected_total >= max_samples:
                    break

            if not selected_chunks:
                return np.empty((0,), dtype=np.int64)
            return np.concatenate(selected_chunks, axis=0)

    def _ensure_open(self) -> h5py.File:
        current_pid = os.getpid()
        if self._h5_file is not None and self._h5_pid != current_pid:
            self.close()

        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._x_ds = self._h5_file["X"]
            self._y_ds = self._h5_file["Y"]
            self._z_ds = self._h5_file["Z"]
            self._h5_pid = current_pid
        return self._h5_file

    @contextmanager
    def open_h5(self) -> Any:
        h5f = h5py.File(self.h5_path, "r")
        try:
            yield h5f
        finally:
            h5f.close()

    def _histogram(self, *, key: str) -> dict[int | float, int]:
        if len(self.indices) == 0:
            return {}

        histogram: dict[int | float, int] = {}
        with h5py.File(self.h5_path, "r") as h5f:
            for chunk_indices in self._iter_index_chunks(self.indices):
                if key == "label":
                    values = np.argmax(np.asarray(h5f["Y"][chunk_indices]), axis=1)
                elif key == "snr":
                    values = _flatten_z(np.asarray(h5f["Z"][chunk_indices]))
                else:
                    raise ValueError(f"Unsupported histogram key: {key}")

                uniques, counts = np.unique(values, return_counts=True)
                for unique, count in zip(uniques.tolist(), counts.tolist(), strict=True):
                    scalar = _maybe_python_scalar(unique)
                    histogram[scalar] = histogram.get(scalar, 0) + int(count)

        return dict(sorted(histogram.items(), key=lambda item: item[0]))

    def _iter_index_chunks(self, indices: np.ndarray) -> list[np.ndarray]:
        return [
            indices[start : start + self.scan_chunk_size]
            for start in range(0, len(indices), self.scan_chunk_size)
        ]


def _flatten_z(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    return array.reshape(array.shape[0], -1)[:, 0]


def _flatten_scalar_z(value: Any) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(array[0])


def _maybe_python_scalar(value: Any) -> int | float:
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_class_names(class_names_path: str | Path) -> list[str]:
    """Load class names from a plain-text file with one label per line."""

    path = Path(class_names_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if "classes =" in text:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("classes ="):
                _, rhs = stripped.split("=", 1)
                try:
                    parsed = ast.literal_eval(rhs.strip())
                except Exception:
                    continue
                if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                    return parsed

        start = text.find("classes =")
        if start != -1:
            bracket_start = text.find("[", start)
            bracket_end = text.find("]", bracket_start)
            if bracket_start != -1 and bracket_end != -1:
                try:
                    parsed = ast.literal_eval(text[bracket_start : bracket_end + 1])
                except Exception:
                    parsed = None
                if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                    return parsed

    return [line.strip() for line in text.splitlines() if line.strip()]


def infer_class_names_from_h5(h5_path: str | Path) -> list[str] | None:
    """Try to recover class names from common HDF5 metadata layouts."""

    path = Path(h5_path).expanduser().resolve()
    # RadioML 2018.01A mirrors often ship sidecar class-name files instead of
    # embedding names inside the HDF5. Prefer the known fixed-order file when
    # present, then fall back to the original classes.txt.
    for candidate_name in ("classes-fixed.txt", "classes.txt"):
        candidate = path.parent / candidate_name
        if candidate.exists():
            try:
                return load_class_names(candidate)
            except Exception:
                pass

    with h5py.File(path, "r") as h5f:
        for key in ("classes", "class_names", "mods", "modulations"):
            if key in h5f:
                dataset = h5f[key][:]
                return [_decode_if_bytes(item) for item in dataset.tolist()]

            if key in h5f.attrs:
                attr = h5f.attrs[key]
                if np.ndim(attr) == 0:
                    return [_decode_if_bytes(attr)]
                return [_decode_if_bytes(item) for item in np.asarray(attr).tolist()]
    return None


def build_label_name_map(
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict[int, str]:
    if class_names is None:
        return {index: f"class_{index}" for index in range(num_classes)}
    return {index: name for index, name in enumerate(class_names)}

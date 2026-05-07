"""Stratified split helpers for RadioML 2018.01A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from rfml.data.radioml2018 import (
    RadioML2018Dataset,
    build_label_name_map,
    infer_class_names_from_h5,
)


@dataclass(frozen=True)
class SplitBundle:
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    class_names: tuple[str, ...] | None
    h5_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "class_names": np.array(self.class_names if self.class_names is not None else [], dtype=object),
            "h5_path": self.h5_path,
        }


def scan_labels_and_snrs(
    h5_path: str | Path,
    *,
    scan_chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(h5_path).expanduser().resolve()
    info = RadioML2018Dataset.scan_info(path, scan_chunk_size=scan_chunk_size)

    labels = np.empty((info.num_examples,), dtype=np.int64)
    snrs = np.empty((info.num_examples,), dtype=np.float32)

    with h5py.File(path, "r") as h5f:
        for start in range(0, info.num_examples, scan_chunk_size):
            stop = min(start + scan_chunk_size, info.num_examples)
            labels[start:stop] = np.argmax(np.asarray(h5f["Y"][start:stop]), axis=1)
            z_chunk = np.asarray(h5f["Z"][start:stop], dtype=np.float32)
            snrs[start:stop] = z_chunk.reshape(z_chunk.shape[0], -1)[:, 0]

    return labels, snrs


def build_stratify_groups(labels: np.ndarray, snrs: np.ndarray) -> np.ndarray:
    if labels.shape != snrs.shape:
        raise ValueError("labels and snrs must have the same shape")
    snr_tokens = np.rint(snrs * 1000).astype(np.int64)
    return np.char.add(labels.astype(str), np.char.add("::", snr_tokens.astype(str)))


def stratified_split_indices(
    labels: np.ndarray,
    snrs: np.ndarray,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_group_size_for_strict_split: int = 3,
) -> SplitBundle:
    if labels.ndim != 1 or snrs.ndim != 1:
        raise ValueError("labels and snrs must be 1D arrays")
    if len(labels) != len(snrs):
        raise ValueError("labels and snrs must have the same length")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = np.random.default_rng(seed)
    groups = build_stratify_groups(labels, snrs)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    unique_groups = np.unique(groups)
    for group in unique_groups:
        group_indices = np.flatnonzero(groups == group)
        group_indices = rng.permutation(group_indices)
        n = int(group_indices.shape[0])

        if n < min_group_size_for_strict_split:
            raise ValueError(
                "Encountered a stratification group with fewer than "
                f"{min_group_size_for_strict_split} samples: {group!r} has {n}. "
                "A strict 70/15/15 split cannot keep this group in all splits."
            )

        train_count = int(np.floor(n * train_ratio))
        val_count = int(np.floor(n * val_ratio))
        test_count = n - train_count - val_count

        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        test_count = n - train_count - val_count

        if test_count == 0:
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1
            test_count = 1

        train_parts.append(group_indices[:train_count])
        val_parts.append(group_indices[train_count : train_count + val_count])
        test_parts.append(group_indices[train_count + val_count : train_count + val_count + test_count])

    train_indices = np.sort(np.concatenate(train_parts).astype(np.int64, copy=False))
    val_indices = np.sort(np.concatenate(val_parts).astype(np.int64, copy=False))
    test_indices = np.sort(np.concatenate(test_parts).astype(np.int64, copy=False))

    if len(np.intersect1d(train_indices, val_indices)) > 0:
        raise RuntimeError("Train and val splits overlap")
    if len(np.intersect1d(train_indices, test_indices)) > 0:
        raise RuntimeError("Train and test splits overlap")
    if len(np.intersect1d(val_indices, test_indices)) > 0:
        raise RuntimeError("Val and test splits overlap")

    if train_indices.size + val_indices.size + test_indices.size != labels.size:
        raise RuntimeError("Split sizes do not sum to the total number of samples")

    return SplitBundle(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        class_names=None,
        h5_path="",
    )


def create_stratified_splits_from_h5(
    h5_path: str | Path,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    scan_chunk_size: int = 8192,
    class_names: Sequence[str] | None = None,
) -> SplitBundle:
    path = Path(h5_path).expanduser().resolve()
    labels, snrs = scan_labels_and_snrs(path, scan_chunk_size=scan_chunk_size)
    bundle = stratified_split_indices(
        labels,
        snrs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    resolved_class_names = tuple(class_names) if class_names is not None else None
    if resolved_class_names is None:
        inferred = infer_class_names_from_h5(path)
        if inferred is not None:
            resolved_class_names = tuple(inferred)

    return SplitBundle(
        train_indices=bundle.train_indices,
        val_indices=bundle.val_indices,
        test_indices=bundle.test_indices,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        class_names=resolved_class_names,
        h5_path=str(path),
    )


def save_split_bundle(bundle: SplitBundle, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle.as_dict())
    return path


def load_split_bundle(split_path: str | Path) -> SplitBundle:
    path = Path(split_path).expanduser().resolve()
    with np.load(path, allow_pickle=True) as data:
        class_names_array = data["class_names"]
        class_names = tuple(class_names_array.tolist()) if class_names_array.size > 0 else None
        return SplitBundle(
            train_indices=data["train_indices"].astype(np.int64, copy=False),
            val_indices=data["val_indices"].astype(np.int64, copy=False),
            test_indices=data["test_indices"].astype(np.int64, copy=False),
            seed=int(data["seed"]),
            train_ratio=float(data["train_ratio"]),
            val_ratio=float(data["val_ratio"]),
            test_ratio=float(data["test_ratio"]),
            class_names=class_names,
            h5_path=str(data["h5_path"]),
        )


def resolve_split_indices(bundle: SplitBundle, split_name: str) -> np.ndarray:
    normalized = split_name.lower()
    if normalized == "train":
        return bundle.train_indices
    if normalized == "val":
        return bundle.val_indices
    if normalized == "test":
        return bundle.test_indices
    raise ValueError(f"Unsupported split name: {split_name}")


def summarize_split_distribution(
    labels: np.ndarray,
    snrs: np.ndarray,
    split_indices: np.ndarray,
) -> dict[str, dict[int | float, int]]:
    split_labels = labels[split_indices]
    split_snrs = snrs[split_indices]
    label_unique, label_counts = np.unique(split_labels, return_counts=True)
    snr_unique, snr_counts = np.unique(split_snrs, return_counts=True)
    return {
        "label_counts": {
            int(label): int(count)
            for label, count in zip(label_unique.tolist(), label_counts.tolist(), strict=True)
        },
        "snr_counts": {
            float(snr): int(count)
            for snr, count in zip(snr_unique.tolist(), snr_counts.tolist(), strict=True)
        },
    }


def build_split_report(
    labels: np.ndarray,
    snrs: np.ndarray,
    bundle: SplitBundle,
) -> dict[str, Any]:
    class_names = bundle.class_names
    label_name_map = build_label_name_map(int(labels.max()) + 1, class_names)

    def convert_label_counts(raw_counts: dict[int, int]) -> dict[str, int]:
        return {label_name_map[label]: count for label, count in raw_counts.items()}

    train_summary = summarize_split_distribution(labels, snrs, bundle.train_indices)
    val_summary = summarize_split_distribution(labels, snrs, bundle.val_indices)
    test_summary = summarize_split_distribution(labels, snrs, bundle.test_indices)

    return {
        "train": {
            "size": int(bundle.train_indices.size),
            "label_counts": convert_label_counts(train_summary["label_counts"]),
            "snr_counts": train_summary["snr_counts"],
        },
        "val": {
            "size": int(bundle.val_indices.size),
            "label_counts": convert_label_counts(val_summary["label_counts"]),
            "snr_counts": val_summary["snr_counts"],
        },
        "test": {
            "size": int(bundle.test_indices.size),
            "label_counts": convert_label_counts(test_summary["label_counts"]),
            "snr_counts": test_summary["snr_counts"],
        },
    }

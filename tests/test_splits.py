from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from rfml.data.splits import (
    build_split_report,
    create_stratified_splits_from_h5,
    load_split_bundle,
    save_split_bundle,
    scan_labels_and_snrs,
    stratified_split_indices,
)


def test_scan_labels_and_snrs_returns_expected_arrays(tmp_path: Path) -> None:
    h5_path = _build_stratified_h5(tmp_path / "tiny_radioml.h5")
    labels, snrs = scan_labels_and_snrs(h5_path, scan_chunk_size=4)

    assert labels.shape == (48,)
    assert snrs.shape == (48,)
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2, 3]
    assert sorted(np.unique(snrs).tolist()) == [-20.0, -10.0, 0.0, 10.0]


def test_stratified_split_indices_preserves_group_coverage() -> None:
    labels = np.repeat(np.array([0, 1, 2], dtype=np.int64), 12)
    snrs = np.tile(np.repeat(np.array([-20.0, 0.0], dtype=np.float32), 6), 3)

    bundle = stratified_split_indices(labels, snrs, seed=42)
    assert bundle.train_indices.size + bundle.val_indices.size + bundle.test_indices.size == labels.size

    groups = set(zip(labels.tolist(), snrs.tolist(), strict=True))
    for split_indices in (bundle.train_indices, bundle.val_indices, bundle.test_indices):
        split_groups = set(zip(labels[split_indices].tolist(), snrs[split_indices].tolist(), strict=True))
        assert split_groups == groups


def test_create_save_and_load_split_bundle(tmp_path: Path) -> None:
    h5_path = _build_stratified_h5(tmp_path / "tiny_radioml.h5")
    bundle = create_stratified_splits_from_h5(h5_path, seed=42, scan_chunk_size=4)

    out_path = save_split_bundle(bundle, tmp_path / "splits.npz")
    loaded = load_split_bundle(out_path)

    assert loaded.seed == 42
    assert np.array_equal(loaded.train_indices, bundle.train_indices)
    assert np.array_equal(loaded.val_indices, bundle.val_indices)
    assert np.array_equal(loaded.test_indices, bundle.test_indices)
    assert loaded.class_names == bundle.class_names


def test_build_split_report_uses_label_names(tmp_path: Path) -> None:
    h5_path = _build_stratified_h5(tmp_path / "tiny_radioml.h5")
    labels, snrs = scan_labels_and_snrs(h5_path)
    bundle = create_stratified_splits_from_h5(h5_path, seed=42)

    report = build_split_report(labels, snrs, bundle)
    assert set(report) == {"train", "val", "test"}
    assert report["train"]["size"] > 0
    assert "BPSK" in report["train"]["label_counts"]


def test_stratified_split_raises_for_too_small_groups() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    snrs = np.array([-20.0, 0.0, -20.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="fewer than 3 samples"):
        stratified_split_indices(labels, snrs, seed=42)


def _build_stratified_h5(path: Path) -> Path:
    num_classes = 4
    snrs = [-20.0, -10.0, 0.0, 10.0]
    repeats_per_group = 3
    seq_len = 1024
    num_samples = num_classes * len(snrs) * repeats_per_group

    x = np.zeros((num_samples, seq_len, 2), dtype=np.float32)
    y = np.zeros((num_samples, num_classes), dtype=np.float32)
    z = np.zeros((num_samples, 1), dtype=np.float32)

    index = 0
    for label in range(num_classes):
        for snr in snrs:
            for repeat in range(repeats_per_group):
                phase = label + repeat / 10.0
                timeline = np.linspace(0, 6.28, seq_len, dtype=np.float32)
                x[index, :, 0] = np.sin(timeline + phase)
                x[index, :, 1] = np.cos(timeline + phase)
                y[index, label] = 1.0
                z[index, 0] = snr
                index += 1

    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("X", data=x)
        h5f.create_dataset("Y", data=y)
        h5f.create_dataset("Z", data=z)
        h5f.create_dataset(
            "classes",
            data=np.array([b"BPSK", b"QPSK", b"QAM16", b"QAM64"]),
        )
    return path

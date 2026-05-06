from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from rfml.data.radioml2018 import RadioML2018Dataset


def _build_tiny_h5(path: Path) -> Path:
    num_samples = 12
    seq_len = 1024
    num_classes = 4

    x = np.zeros((num_samples, seq_len, 2), dtype=np.float32)
    y = np.zeros((num_samples, num_classes), dtype=np.float32)
    z = np.zeros((num_samples, 1), dtype=np.float32)

    snrs = [-20.0, -10.0, 0.0, 10.0]
    for index in range(num_samples):
        x[index, :, 0] = index
        x[index, :, 1] = -index
        label = index % num_classes
        y[index, label] = 1.0
        z[index, 0] = snrs[index % len(snrs)]

    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("X", data=x)
        h5f.create_dataset("Y", data=y)
        h5f.create_dataset("Z", data=z)
        h5f.create_dataset(
            "classes",
            data=np.array([b"BPSK", b"QPSK", b"QAM16", b"QAM64"]),
        )
    return path


@pytest.fixture()
def tiny_h5(tmp_path: Path) -> Path:
    return _build_tiny_h5(tmp_path / "tiny_radioml.h5")


def test_dataset_returns_expected_sample_structure(tiny_h5: Path) -> None:
    dataset = RadioML2018Dataset(tiny_h5)
    sample = dataset[3]

    assert set(sample) == {"iq", "label", "snr", "index"}
    assert isinstance(sample["iq"], torch.Tensor)
    assert sample["iq"].shape == (2, 1024)
    assert sample["iq"].dtype == torch.float32
    assert sample["label"].dtype == torch.long
    assert int(sample["label"].item()) == 3
    assert sample["snr"].dtype == torch.float32
    assert float(sample["snr"].item()) == pytest.approx(10.0)
    assert sample["index"] == 3


def test_dataset_supports_filters_and_limits(tiny_h5: Path) -> None:
    dataset = RadioML2018Dataset(
        tiny_h5,
        snr_filter=[-20.0, 0.0],
        class_filter=[0, 2],
        max_samples=3,
    )

    assert len(dataset) == 3
    for sample in dataset:
        assert int(sample["label"].item()) in {0, 2}
        assert float(sample["snr"].item()) in {-20.0, 0.0}


def test_dataset_supports_split_indices(tiny_h5: Path) -> None:
    split_indices = [1, 4, 7, 10]
    dataset = RadioML2018Dataset(tiny_h5, split_indices=split_indices)

    assert len(dataset) == len(split_indices)
    returned_indices = [dataset[idx]["index"] for idx in range(len(dataset))]
    assert returned_indices == split_indices


def test_dataset_describe_returns_histograms(tiny_h5: Path) -> None:
    dataset = RadioML2018Dataset(tiny_h5)
    summary = dataset.describe()

    assert summary["num_selected_samples"] == 12
    assert summary["num_total_samples"] == 12
    assert summary["num_classes"] == 4
    assert summary["sample_shape"] == (1024, 2)
    assert summary["snr_values"] == [-20.0, -10.0, 0.0, 10.0]
    assert summary["class_counts"] == {"0": 3, "1": 3, "2": 3, "3": 3}
    assert summary["snr_counts"] == {-20.0: 3, -10.0: 3, 0.0: 3, 10.0: 3}


def test_dataloader_multiworker_access(tiny_h5: Path) -> None:
    dataset = RadioML2018Dataset(tiny_h5)
    _ = dataset[0]
    loader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)

    batch = next(iter(loader))
    assert tuple(batch["iq"].shape) == (2, 2, 1024)
    assert batch["label"].dtype == torch.long
    assert batch["snr"].dtype == torch.float32

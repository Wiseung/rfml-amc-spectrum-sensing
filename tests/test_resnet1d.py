from __future__ import annotations

import torch

from rfml.models.resnet1d import BasicBlock1D, build_resnet1d


def test_basicblock1d_preserves_shape_without_downsample() -> None:
    block = BasicBlock1D(64, 64, stride=1)
    x = torch.randn(2, 64, 128)
    y = block(x)
    assert tuple(y.shape) == (2, 64, 128)


def test_basicblock1d_downsamples_when_stride_is_two() -> None:
    block = BasicBlock1D(64, 128, stride=2)
    x = torch.randn(2, 64, 128)
    y = block(x)
    assert tuple(y.shape) == (2, 128, 64)


def test_resnet1d_small_forward_shape() -> None:
    model = build_resnet1d("resnet1d-small", num_classes=24, dropout=0.2, classifier_hidden_dim=256)
    x = torch.randn(4, 2, 1024)
    y = model(x)
    assert tuple(y.shape) == (4, 24)


def test_resnet1d_medium_forward_shape() -> None:
    model = build_resnet1d("resnet1d-medium", num_classes=24, dropout=0.2, classifier_hidden_dim=256)
    x = torch.randn(2, 2, 1024)
    y = model(x)
    assert tuple(y.shape) == (2, 24)

#!/usr/bin/env python3
"""Minimal import and forward-pass smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import delegate_to_conda_if_needed, delegated_env_name


delegate_to_conda_if_needed(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    print("RFML smoke test")
    print(f"repo_root: {REPO_ROOT}")
    print(f"delegated_conda_env: {delegated_env_name() or '<none>'}")

    try:
        import torch
    except Exception as exc:
        print(f"torch_import: failed ({exc!r})")
        return 1

    try:
        from rfml.models.cnn1d import TinyCNN1D
    except Exception as exc:
        print(f"rfml_model_import: failed ({exc!r})")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")
    else:
        print("warning: CUDA is unavailable, running CPU-only smoke test.")

    batch_size = 4
    iq_channels = 2
    seq_len = 1024
    num_classes = 24

    model = TinyCNN1D(in_channels=iq_channels, num_classes=num_classes).to(device)
    model.eval()

    x = torch.randn(batch_size, iq_channels, seq_len, device=device)
    with torch.no_grad():
        logits = model(x)

    expected_shape = (batch_size, num_classes)
    print(f"input_shape: {tuple(x.shape)}")
    print(f"output_shape: {tuple(logits.shape)}")
    print(f"expected_output_shape: {expected_shape}")

    if tuple(logits.shape) != expected_shape:
        print("smoke_test: failed due to unexpected output shape")
        return 1

    if torch.isnan(logits).any().item():
        print("smoke_test: failed due to NaN output")
        return 1

    print("smoke_test: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

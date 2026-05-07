"""Loss helpers for RFML training."""

from __future__ import annotations

from torch import nn


def build_classification_loss(name: str = "cross_entropy") -> nn.Module:
    normalized = name.lower()
    if normalized == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss: {name}")

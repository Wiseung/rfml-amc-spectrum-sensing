"""Loss helpers for RFML training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MultiTaskLossOutput:
    total_loss: torch.Tensor
    modulation_loss: torch.Tensor
    sensing_loss: torch.Tensor


def build_classification_loss(name: str = "cross_entropy") -> nn.Module:
    normalized = name.lower()
    if normalized == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss: {name}")


def compute_multitask_loss(
    modulation_logits: torch.Tensor,
    modulation_targets: torch.Tensor,
    modulation_mask: torch.Tensor,
    sensing_logits: torch.Tensor,
    sensing_targets: torch.Tensor,
    *,
    lambda_sensing: float = 1.0,
) -> MultiTaskLossOutput:
    if modulation_mask.ndim != 1:
        modulation_mask = modulation_mask.view(-1)
    valid_mask = modulation_mask > 0.5
    if torch.any(valid_mask):
        modulation_loss = F.cross_entropy(modulation_logits[valid_mask], modulation_targets[valid_mask])
    else:
        modulation_loss = modulation_logits.sum() * 0.0

    sensing_loss = F.cross_entropy(sensing_logits, sensing_targets)
    total_loss = modulation_loss + float(lambda_sensing) * sensing_loss
    return MultiTaskLossOutput(
        total_loss=total_loss,
        modulation_loss=modulation_loss,
        sensing_loss=sensing_loss,
    )

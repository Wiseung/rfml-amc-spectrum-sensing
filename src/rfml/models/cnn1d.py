"""CNN1D model placeholder."""

from __future__ import annotations

import torch
from torch import nn


class TinyCNN1D(nn.Module):
    """Minimal forward-pass model for Phase 0 smoke testing.

    The real AMC CNN1D will replace this in a later phase.
    """

    def __init__(self, in_channels: int = 2, hidden_channels: int = 16, num_classes: int = 24) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)

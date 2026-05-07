"""1D CNN models for modulation classification."""

from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    """Baseline 1D CNN for RadioML AMC.

    Expected input shape:
        (batch, 2, 1024)
    """

    def __init__(
        self,
        *,
        in_channels: int = 2,
        num_classes: int = 24,
        channels: tuple[int, int, int] = (64, 128, 256),
        kernel_sizes: tuple[int, int, int] = (7, 5, 3),
        dropout: float = 0.3,
        classifier_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        k1, k2, k3 = kernel_sizes

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=k1, padding=k1 // 2, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class TinyCNN1D(nn.Module):
    """Minimal forward-pass model kept for lightweight smoke tests."""

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

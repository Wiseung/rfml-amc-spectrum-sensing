"""CNN models for STFT spectrogram inputs."""

from __future__ import annotations

import torch
from torch import nn


class ResidualConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.relu(out)


class STFTCNN(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 24,
        channels: tuple[int, int, int] = (32, 64, 128),
        dropout: float = 0.3,
        classifier_hidden_dim: int = 256,
        backbone: str = "basic",
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        normalized = backbone.lower()
        if normalized == "basic":
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif normalized == "deep":
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, c1, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                ResidualConvBlock2D(c1, c1, stride=1),
                ResidualConvBlock2D(c1, c2, stride=2),
                ResidualConvBlock2D(c2, c2, stride=1),
                ResidualConvBlock2D(c2, c3, stride=2),
                ResidualConvBlock2D(c3, c3, stride=1),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        else:
            raise ValueError(f"Unsupported STFT backbone: {backbone}")
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

"""ResNet1D variants for RF modulation classification."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 2,
        num_classes: int = 24,
        stem_channels: int = 64,
        stage_channels: tuple[int, int, int] = (64, 128, 256),
        stage_blocks: tuple[int, int, int] = (2, 2, 2),
        classifier_hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )

        self.in_channels = stem_channels
        self.stage1 = self._make_stage(stage_channels[0], stage_blocks[0], stride=1)
        self.stage2 = self._make_stage(stage_channels[1], stage_blocks[1], stride=2)
        self.stage3 = self._make_stage(stage_channels[2], stage_blocks[2], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stage_channels[-1], classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def _make_stage(self, out_channels: int, num_blocks: int, *, stride: int) -> nn.Sequential:
        blocks = [BasicBlock1D(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock1D(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.head(x)


def build_resnet1d(model_size: str, *, num_classes: int, dropout: float, classifier_hidden_dim: int) -> ResNet1D:
    normalized = model_size.lower()
    if normalized == "resnet1d-small":
        return ResNet1D(
            num_classes=num_classes,
            stem_channels=64,
            stage_channels=(64, 128, 256),
            stage_blocks=(2, 2, 2),
            classifier_hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )
    if normalized == "resnet1d-medium":
        return ResNet1D(
            num_classes=num_classes,
            stem_channels=64,
            stage_channels=(64, 128, 256),
            stage_blocks=(3, 3, 3),
            classifier_hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported ResNet1D model size: {model_size}")

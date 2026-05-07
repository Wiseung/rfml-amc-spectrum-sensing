"""Multi-task RF model for AMC and spectrum sensing."""

from __future__ import annotations

import torch
from torch import nn

from rfml.models.resnet1d import BasicBlock1D


class CNN1DEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 2,
        channels: tuple[int, int, int] = (64, 128, 256),
        kernel_sizes: tuple[int, int, int] = (7, 5, 3),
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
        self.output_dim = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, start_dim=1)


class ResNet1DEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 2,
        stem_channels: int = 64,
        stage_channels: tuple[int, int, int] = (64, 128, 256),
        stage_blocks: tuple[int, int, int] = (2, 2, 2),
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
        self.output_dim = stage_channels[-1]

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
        return torch.flatten(x, start_dim=1)


class MultiTaskRFModel(nn.Module):
    def __init__(
        self,
        *,
        backbone: str = "cnn1d",
        modulation_num_classes: int = 24,
        sensing_num_classes: int = 2,
        in_channels: int = 2,
        channels: tuple[int, int, int] = (64, 128, 256),
        kernel_sizes: tuple[int, int, int] = (7, 5, 3),
        classifier_hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        normalized = backbone.lower()
        if normalized == "cnn1d":
            self.encoder = CNN1DEncoder(
                in_channels=in_channels,
                channels=channels,
                kernel_sizes=kernel_sizes,
            )
        elif normalized == "resnet1d-small":
            self.encoder = ResNet1DEncoder(
                in_channels=in_channels,
                stem_channels=64,
                stage_channels=(64, 128, 256),
                stage_blocks=(2, 2, 2),
            )
        elif normalized == "resnet1d-medium":
            self.encoder = ResNet1DEncoder(
                in_channels=in_channels,
                stem_channels=64,
                stage_channels=(64, 128, 256),
                stage_blocks=(3, 3, 3),
            )
        else:
            raise ValueError(f"Unsupported multitask backbone: {backbone}")

        feature_dim = self.encoder.output_dim
        self.modulation_head = self._build_head(feature_dim, classifier_hidden_dim, modulation_num_classes, dropout)
        self.sensing_head = self._build_head(feature_dim, classifier_hidden_dim, sensing_num_classes, dropout)

    @staticmethod
    def _build_head(
        feature_dim: int,
        classifier_hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(feature_dim, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        return {
            "modulation_logits": self.modulation_head(features),
            "sensing_logits": self.sensing_head(features),
        }

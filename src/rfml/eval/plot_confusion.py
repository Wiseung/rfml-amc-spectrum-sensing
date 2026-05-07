"""Confusion-matrix plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from rfml.training.metrics import plot_confusion_matrix as _plot_confusion_matrix


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str | Path,
    *,
    class_names: Sequence[str] | None = None,
) -> Path:
    """Thin wrapper that exposes the confusion-plot helper under rfml.eval."""

    return _plot_confusion_matrix(cm, output_path, class_names=class_names)


__all__ = ["plot_confusion_matrix"]

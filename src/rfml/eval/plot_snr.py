"""Accuracy-vs-SNR plotting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rfml.training.metrics import plot_accuracy_vs_snr as _plot_accuracy_vs_snr


def plot_accuracy_vs_snr(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Thin wrapper that exposes the SNR plot helper under rfml.eval."""

    return _plot_accuracy_vs_snr(df, output_path)


__all__ = ["plot_accuracy_vs_snr"]

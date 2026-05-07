"""Spectrum sensing metrics and plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


@dataclass(frozen=True)
class SensingEvaluationResult:
    metrics: pd.DataFrame
    roc_curve: pd.DataFrame
    pd_vs_snr: pd.DataFrame
    accuracy: float
    auc_value: float


def evaluate_sensing_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    snrs: np.ndarray,
    *,
    target_pfas: Sequence[float] = (0.10, 0.05),
) -> SensingEvaluationResult:
    if y_true.ndim != 1 or y_score.ndim != 1 or snrs.ndim != 1:
        raise ValueError("y_true, y_score, and snrs must be 1D arrays")
    if not (y_true.shape == y_score.shape == snrs.shape):
        raise ValueError("y_true, y_score, and snrs must have the same shape")
    if set(np.unique(y_true).tolist()) - {0, 1}:
        raise ValueError("y_true must only contain binary labels 0/1")

    y_pred = (y_score >= 0.5).astype(np.int64)
    accuracy = float(accuracy_score(y_true, y_pred))

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    auc_value = float(roc_auc_score(y_true, y_score))
    roc_df = pd.DataFrame(
        {
            "threshold": thresholds.astype(np.float32, copy=False),
            "pfa": fpr.astype(np.float32, copy=False),
            "pd": tpr.astype(np.float32, copy=False),
        }
    ).sort_values("pfa", kind="mergesort")

    metrics_row: dict[str, float | int] = {
        "accuracy": accuracy,
        "roc_auc": auc_value,
        "num_signal_samples": int(np.sum(y_true == 1)),
        "num_noise_samples": int(np.sum(y_true == 0)),
    }
    thresholds_by_pfa: dict[float, float] = {}
    for target_pfa in target_pfas:
        best = select_threshold_at_pfa(roc_df, target_pfa)
        suffix = _pfa_suffix(target_pfa)
        metrics_row[f"pd_at_pfa_{suffix}"] = float(best["pd"])
        metrics_row[f"threshold_at_pfa_{suffix}"] = float(best["threshold"])
        metrics_row[f"actual_pfa_at_{suffix}"] = float(best["pfa"])
        thresholds_by_pfa[target_pfa] = float(best["threshold"])

    pd_vs_snr_rows: list[dict[str, float | int]] = []
    for snr in sorted(np.unique(snrs).tolist()):
        pos_mask = (y_true == 1) & (snrs == snr)
        neg_mask = (y_true == 0) & (snrs == snr)
        row: dict[str, float | int] = {
            "snr": float(snr),
            "num_positive_samples": int(np.sum(pos_mask)),
            "num_negative_samples": int(np.sum(neg_mask)),
        }
        for target_pfa in target_pfas:
            threshold = thresholds_by_pfa[target_pfa]
            suffix = _pfa_suffix(target_pfa)
            row[f"pd_at_pfa_{suffix}"] = (
                float(np.mean(y_score[pos_mask] >= threshold)) if np.any(pos_mask) else float("nan")
            )
            row[f"pfa_at_pfa_{suffix}"] = (
                float(np.mean(y_score[neg_mask] >= threshold)) if np.any(neg_mask) else float("nan")
            )
        pd_vs_snr_rows.append(row)

    return SensingEvaluationResult(
        metrics=pd.DataFrame([metrics_row]),
        roc_curve=roc_df.reset_index(drop=True),
        pd_vs_snr=pd.DataFrame(pd_vs_snr_rows),
        accuracy=accuracy,
        auc_value=auc_value,
    )


def select_threshold_at_pfa(roc_df: pd.DataFrame, target_pfa: float) -> pd.Series:
    eligible = roc_df.loc[roc_df["pfa"] <= target_pfa]
    if not eligible.empty:
        return eligible.iloc[-1]
    return roc_df.iloc[0]


def plot_sensing_roc(output_path: str | Path, roc_df: pd.DataFrame, *, label: str = "CNN detector") -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc_df["pfa"], roc_df["pd"], linewidth=1.5, label=label)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="gray", label="Chance")
    ax.set_xlabel("Pfa")
    ax.set_ylabel("Pd")
    ax.set_title("Spectrum Sensing ROC")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_pd_vs_snr(
    output_path: str | Path,
    pd_vs_snr_df: pd.DataFrame,
    *,
    target_pfas: Sequence[float] = (0.10, 0.05),
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for target_pfa in target_pfas:
        suffix = _pfa_suffix(target_pfa)
        ax.plot(
            pd_vs_snr_df["snr"],
            pd_vs_snr_df[f"pd_at_pfa_{suffix}"],
            marker="o",
            linewidth=1.5,
            label=f"Pd @ Pfa={target_pfa:.2f}",
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Pd")
    ax.set_title("Pd vs SNR")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _pfa_suffix(target_pfa: float) -> str:
    return f"{target_pfa:.2f}".replace(".", "p")

"""Training and evaluation metrics for RFML."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from rfml.data.radioml2018 import build_label_name_map


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_accuracy_vs_snr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snrs: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for snr in sorted(np.unique(snrs).tolist()):
        mask = snrs == snr
        rows.append(
            {
                "snr": float(snr),
                "num_samples": int(np.sum(mask)),
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            }
        )
    return pd.DataFrame(rows)


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
) -> tuple[str, dict]:
    num_classes = len(class_names) if class_names is not None else int(max(y_true.max(), y_pred.max())) + 1
    label_name_map = build_label_name_map(num_classes, class_names)
    labels = list(range(num_classes))
    target_names = [label_name_map[idx] for idx in labels]
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return report_text, report_dict


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def plot_accuracy_vs_snr(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["snr"], df["accuracy"], marker="o", linewidth=1.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs SNR")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str | Path,
    *,
    class_names: Sequence[str] | None = None,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if class_names is not None:
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

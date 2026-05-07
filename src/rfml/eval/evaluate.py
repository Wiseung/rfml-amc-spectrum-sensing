"""Reusable evaluation helpers for AMC and spectrum sensing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from rfml.eval.plot_confusion import plot_confusion_matrix
from rfml.eval.plot_snr import plot_accuracy_vs_snr
from rfml.eval.sensing_metrics import (
    SensingEvaluationResult,
    evaluate_sensing_predictions,
    plot_pd_vs_snr,
    plot_sensing_roc,
)
from rfml.training.metrics import (
    compute_accuracy,
    compute_accuracy_vs_snr,
    compute_classification_report,
    compute_confusion_matrix,
)


@dataclass(frozen=True)
class AMCEvaluationResult:
    overall_accuracy: float
    accuracy_vs_snr: pd.DataFrame
    confusion_matrix: np.ndarray
    classification_report_text: str
    classification_report_dict: dict[str, Any]


def evaluate_amc_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snrs: np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
) -> AMCEvaluationResult:
    """Evaluate AMC predictions with summary tables and matrices."""

    if y_true.ndim != 1 or y_pred.ndim != 1 or snrs.ndim != 1:
        raise ValueError("y_true, y_pred, and snrs must be 1D arrays")
    if not (y_true.shape == y_pred.shape == snrs.shape):
        raise ValueError("y_true, y_pred, and snrs must have the same shape")

    overall_accuracy = compute_accuracy(y_true, y_pred)
    accuracy_vs_snr = compute_accuracy_vs_snr(y_true, y_pred, snrs)
    classification_report_text, classification_report_dict = compute_classification_report(
        y_true,
        y_pred,
        class_names=class_names,
    )
    num_classes = len(class_names) if class_names is not None else int(max(y_true.max(), y_pred.max())) + 1
    confusion = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    return AMCEvaluationResult(
        overall_accuracy=overall_accuracy,
        accuracy_vs_snr=accuracy_vs_snr,
        confusion_matrix=confusion,
        classification_report_text=classification_report_text,
        classification_report_dict=classification_report_dict,
    )


def save_amc_evaluation(
    result: AMCEvaluationResult,
    output_dir: str | Path,
    *,
    class_names: Sequence[str] | None = None,
    prefix: str = "",
) -> dict[str, Path]:
    """Persist AMC evaluation CSV, JSON, text, and figure artifacts."""

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{prefix}_" if prefix else ""
    acc_csv = out_dir / f"{file_prefix}accuracy_vs_snr.csv"
    cm_csv = out_dir / f"{file_prefix}confusion_matrix.csv"
    report_txt = out_dir / f"{file_prefix}classification_report.txt"
    report_json = out_dir / f"{file_prefix}classification_report.json"
    summary_json = out_dir / f"{file_prefix}summary.json"
    acc_png = plot_accuracy_vs_snr(result.accuracy_vs_snr, out_dir / f"{file_prefix}acc_vs_snr.png")
    cm_png = plot_confusion_matrix(
        result.confusion_matrix,
        out_dir / f"{file_prefix}confusion_matrix.png",
        class_names=class_names,
    )

    result.accuracy_vs_snr.to_csv(acc_csv, index=False)
    pd.DataFrame(result.confusion_matrix).to_csv(cm_csv, index=False)
    report_txt.write_text(result.classification_report_text, encoding="utf-8")
    report_json.write_text(json.dumps(result.classification_report_dict, indent=2), encoding="utf-8")
    summary_json.write_text(
        json.dumps({"overall_accuracy": result.overall_accuracy}, indent=2),
        encoding="utf-8",
    )

    return {
        "accuracy_vs_snr_csv": acc_csv,
        "confusion_matrix_csv": cm_csv,
        "classification_report_txt": report_txt,
        "classification_report_json": report_json,
        "summary_json": summary_json,
        "acc_vs_snr_png": acc_png,
        "confusion_matrix_png": cm_png,
    }


def save_sensing_evaluation(
    result: SensingEvaluationResult,
    output_dir: str | Path,
    *,
    prefix: str = "",
) -> dict[str, Path]:
    """Persist sensing evaluation CSV and figure artifacts."""

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{prefix}_" if prefix else ""
    metrics_csv = out_dir / f"{file_prefix}sensing_metrics.csv"
    roc_csv = out_dir / f"{file_prefix}sensing_roc_curve.csv"
    pd_snr_csv = out_dir / f"{file_prefix}pd_vs_snr.csv"
    summary_json = out_dir / f"{file_prefix}sensing_summary.json"

    result.metrics.to_csv(metrics_csv, index=False)
    result.roc_curve.to_csv(roc_csv, index=False)
    result.pd_vs_snr.to_csv(pd_snr_csv, index=False)
    roc_png = plot_sensing_roc(out_dir / f"{file_prefix}sensing_roc.png", result.roc_curve)
    pd_snr_png = plot_pd_vs_snr(out_dir / f"{file_prefix}pd_vs_snr.png", result.pd_vs_snr)

    summary_payload = {
        "accuracy": result.accuracy,
        "roc_auc": result.auc_value,
    }
    if not result.metrics.empty:
        row = result.metrics.iloc[0].to_dict()
        for key, value in row.items():
            if isinstance(value, (np.integer, int)):
                summary_payload[key] = int(value)
            else:
                summary_payload[key] = float(value)
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "sensing_metrics_csv": metrics_csv,
        "sensing_roc_curve_csv": roc_csv,
        "pd_vs_snr_csv": pd_snr_csv,
        "sensing_summary_json": summary_json,
        "sensing_roc_png": roc_png,
        "pd_vs_snr_png": pd_snr_png,
    }


__all__ = [
    "AMCEvaluationResult",
    "SensingEvaluationResult",
    "evaluate_amc_predictions",
    "evaluate_sensing_predictions",
    "save_amc_evaluation",
    "save_sensing_evaluation",
]

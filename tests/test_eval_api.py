from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rfml.eval import (
    evaluate_amc_predictions,
    evaluate_sensing_predictions,
    save_amc_evaluation,
    save_sensing_evaluation,
)


def test_amc_eval_api_computes_metrics_and_saves_artifacts(tmp_path: Path) -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1, 2, 0], dtype=np.int64)
    snrs = np.array([-20.0, -20.0, 0.0, 0.0, 10.0, 10.0], dtype=np.float32)

    result = evaluate_amc_predictions(
        y_true,
        y_pred,
        snrs,
        class_names=["BPSK", "QPSK", "QAM16"],
    )
    paths = save_amc_evaluation(
        result,
        tmp_path / "amc_eval",
        class_names=["BPSK", "QPSK", "QAM16"],
    )

    assert 0.0 <= result.overall_accuracy <= 1.0
    assert not result.accuracy_vs_snr.empty
    assert result.confusion_matrix.shape == (3, 3)
    for path in paths.values():
        assert path.exists()

    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert "overall_accuracy" in summary


def test_sensing_eval_api_saves_metrics_and_figures(tmp_path: Path) -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    y_score = np.array([0.9, 0.8, 0.7, 0.4, 0.2, 0.1], dtype=np.float32)
    snrs = np.array([-10.0, 0.0, 10.0, -10.0, 0.0, 10.0], dtype=np.float32)

    result = evaluate_sensing_predictions(y_true, y_score, snrs)
    paths = save_sensing_evaluation(result, tmp_path / "sensing_eval")

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.auc_value <= 1.0
    assert not result.roc_curve.empty
    assert not result.pd_vs_snr.empty
    for path in paths.values():
        assert path.exists()

    summary = json.loads(paths["sensing_summary_json"].read_text(encoding="utf-8"))
    assert "roc_auc" in summary

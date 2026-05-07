"""Scikit-learn baselines for automatic modulation classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rfml.baselines.common import build_feature_batch, load_split, resolve_split_indices
from rfml.data.radioml2018 import build_label_name_map


ClassifierName = Literal["logreg", "svm", "rf", "gb"]


@dataclass(frozen=True)
class SklearnBaselineResult:
    classifier_name: str
    train_accuracy: float
    eval_accuracy: float
    classification_report_text: str
    classification_report_dict: dict[str, Any]
    accuracy_vs_snr: pd.DataFrame
    feature_dim: int
    train_size: int
    eval_size: int


def build_classifier(classifier_name: ClassifierName, *, random_state: int = 42) -> Pipeline:
    if classifier_name == "logreg":
        estimator = LogisticRegression(
            max_iter=2000,
            n_jobs=None,
            random_state=random_state,
            multi_class="auto",
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    if classifier_name == "svm":
        estimator = SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    if classifier_name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline([("model", estimator)])

    if classifier_name == "gb":
        estimator = GradientBoostingClassifier(random_state=random_state)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    raise ValueError(f"Unsupported classifier_name: {classifier_name}")


def compute_accuracy_vs_snr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    snrs: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    unique_snrs = sorted(np.unique(snrs).tolist())
    for snr in unique_snrs:
        mask = snrs == snr
        rows.append(
            {
                "snr": float(snr),
                "num_samples": int(np.sum(mask)),
                "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            }
        )
    return pd.DataFrame(rows)


def run_sklearn_baseline(
    h5_path: str | Path,
    split_path: str | Path,
    *,
    classifier_name: ClassifierName = "svm",
    train_split: str = "train",
    eval_split: str = "test",
    snr_filter: Sequence[int | float] | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    random_state: int = 42,
    scan_chunk_size: int = 8192,
) -> SklearnBaselineResult:
    split_bundle = load_split(split_path)
    class_names = split_bundle.class_names
    train_indices = resolve_split_indices(split_bundle, train_split)
    eval_indices = resolve_split_indices(split_bundle, eval_split)

    train_batch = build_feature_batch(
        h5_path,
        train_indices,
        class_names=class_names,
        snr_filter=snr_filter,
        max_samples=max_train_samples,
        scan_chunk_size=scan_chunk_size,
    )
    eval_batch = build_feature_batch(
        h5_path,
        eval_indices,
        class_names=class_names,
        snr_filter=snr_filter,
        max_samples=max_eval_samples,
        scan_chunk_size=scan_chunk_size,
    )

    classifier = build_classifier(classifier_name, random_state=random_state)
    classifier.fit(train_batch.features, train_batch.labels)

    train_pred = classifier.predict(train_batch.features)
    eval_pred = classifier.predict(eval_batch.features)
    num_classes = len(class_names) if class_names is not None else int(max(train_batch.labels.max(), eval_batch.labels.max())) + 1
    label_name_map = build_label_name_map(num_classes, class_names)
    all_labels = list(range(num_classes))
    target_names = [label_name_map[idx] for idx in all_labels]

    report_dict = classification_report(
        eval_batch.labels,
        eval_pred,
        labels=all_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        eval_batch.labels,
        eval_pred,
        labels=all_labels,
        target_names=target_names,
        zero_division=0,
    )

    return SklearnBaselineResult(
        classifier_name=classifier_name,
        train_accuracy=float(accuracy_score(train_batch.labels, train_pred)),
        eval_accuracy=float(accuracy_score(eval_batch.labels, eval_pred)),
        classification_report_text=report_text,
        classification_report_dict=report_dict,
        accuracy_vs_snr=compute_accuracy_vs_snr(eval_batch.labels, eval_pred, eval_batch.snrs),
        feature_dim=int(train_batch.features.shape[1]),
        train_size=int(train_batch.features.shape[0]),
        eval_size=int(eval_batch.features.shape[0]),
    )

#!/usr/bin/env python3
"""Build compact baseline/model comparison tables and SNR comparison plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _bootstrap import delegate_to_conda_if_needed, delegated_env_name


delegate_to_conda_if_needed(__file__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-acc-vs-snr", default=None)
    parser.add_argument("--baseline-overall-acc", type=float, default=None)
    parser.add_argument("--cnn-run-dir", default=None)
    parser.add_argument("--resnet-run-dir", default=None)
    parser.add_argument("--stft-run-dir", default=None)
    parser.add_argument("--out-dir", default="outputs/comparisons")
    return parser.parse_args()


def _load_summary(run_dir: Path) -> tuple[float, pd.DataFrame]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    acc_vs_snr = pd.read_csv(run_dir / "accuracy_vs_snr.csv")
    return float(summary["overall_accuracy"]), acc_vs_snr


def _write_metric_csv(path: Path, model_name: str, overall_accuracy: float) -> None:
    pd.DataFrame([{"model": model_name, "overall_accuracy": overall_accuracy}]).to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_frames = []

    if args.baseline_acc_vs_snr and args.baseline_overall_acc is not None:
        baseline_df = pd.read_csv(args.baseline_acc_vs_snr)
        baseline_df["model"] = "baseline_svm"
        comparison_frames.append(baseline_df)
        _write_metric_csv(out_dir / "baseline_svm.csv", "baseline_svm", args.baseline_overall_acc)

    if args.cnn_run_dir:
        cnn_dir = Path(args.cnn_run_dir).expanduser().resolve()
        cnn_acc, cnn_df = _load_summary(cnn_dir)
        cnn_df["model"] = "cnn1d"
        comparison_frames.append(cnn_df)
        _write_metric_csv(out_dir / "cnn1d_metrics.csv", "cnn1d", cnn_acc)

    if args.resnet_run_dir:
        resnet_dir = Path(args.resnet_run_dir).expanduser().resolve()
        resnet_acc, resnet_df = _load_summary(resnet_dir)
        resnet_df["model"] = "resnet1d"
        comparison_frames.append(resnet_df)
        _write_metric_csv(out_dir / "resnet1d_metrics.csv", "resnet1d", resnet_acc)

    if args.stft_run_dir:
        stft_dir = Path(args.stft_run_dir).expanduser().resolve()
        stft_acc, stft_df = _load_summary(stft_dir)
        stft_df["model"] = "stft_cnn"
        comparison_frames.append(stft_df)
        _write_metric_csv(out_dir / "stft_cnn_metrics.csv", "stft_cnn", stft_acc)

    if comparison_frames:
        combined = pd.concat(comparison_frames, ignore_index=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name, frame in combined.groupby("model"):
            ax.plot(frame["snr"], frame["accuracy"], marker="o", linewidth=1.5, label=model_name)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs SNR Comparison")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / "acc_vs_snr_compare.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"delegated_conda_env: {delegated_env_name() or '<none>'}")
        print(f"compare_plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

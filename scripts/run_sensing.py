#!/usr/bin/env python3
"""Run spectrum sensing experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from _bootstrap import delegate_to_conda_if_needed, delegated_env_name


delegate_to_conda_if_needed(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfml.baselines.energy_detection import run_energy_detection_from_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GOLD_XYZ_OSC.0001_1024.hdf5")
    parser.add_argument("--method", default="energy", choices=["energy", "cnn"])
    parser.add_argument("--split", required=True, help="Path to saved split npz")
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", default=None, help="Required for --method cnn")
    parser.add_argument("--ckpt", default=None, help="Required for --method cnn")
    parser.add_argument("--eval-out-dir", default="outputs/runs/sensing_eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--snr-filter", type=float, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-chunk-size", type=int, default=8192)
    parser.add_argument("--num-thresholds", type=int, default=256)
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--figures-dir", default="outputs/figures")
    return parser.parse_args()


def _plot_roc_curve(output_path: Path, roc_df) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc_df["pfa"], roc_df["pd"], linewidth=1.5, label="Energy Detection")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", label="Chance")
    ax.set_xlabel("Pfa")
    ax.set_ylabel("Pd")
    ax.set_title("Energy Detection ROC")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_pd_pfa_vs_snr(output_path: Path, pd_snr_df) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pd_snr_df["snr"], pd_snr_df["pd"], marker="o", label="Pd")
    ax.plot(pd_snr_df["snr"], pd_snr_df["pfa"], marker="s", label="Pfa")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Probability")
    ax.set_title("Pd / Pfa vs SNR")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5).expanduser().resolve()
    split_path = Path(args.split).expanduser().resolve()
    if not h5_path.exists():
        print(f"error: HDF5 file not found: {h5_path}")
        return 1
    if not split_path.exists():
        print(f"error: split file not found: {split_path}")
        return 1

    metrics_dir = Path(args.metrics_dir).expanduser().resolve()
    figures_dir = Path(args.figures_dir).expanduser().resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "cnn":
        if args.config is None or args.ckpt is None:
            print("error: --config and --ckpt are required when --method cnn")
            return 1

        from evaluate import main as evaluate_main

        eval_out_dir = Path(args.eval_out_dir).expanduser().resolve()
        original_argv = sys.argv[:]
        try:
            sys.argv = [
                "evaluate.py",
                "--config",
                str(Path(args.config).expanduser().resolve()),
                "--h5",
                str(h5_path),
                "--split",
                str(split_path),
                "--ckpt",
                str(Path(args.ckpt).expanduser().resolve()),
                "--split-name",
                args.split_name,
                "--out-dir",
                str(eval_out_dir),
            ]
            rc = evaluate_main()
        finally:
            sys.argv = original_argv
        if rc != 0:
            return rc

        summary = json.loads((eval_out_dir / "summary.json").read_text(encoding="utf-8"))
        print(f"delegated_conda_env: {delegated_env_name() or '<none>'}")
        print(f"h5_path: {h5_path}")
        print(f"split_path: {split_path}")
        print(f"roc_auc: {float(summary['roc_auc']):.6f}")
        print(f"pd_at_pfa_0p10: {float(summary['pd_at_pfa_0p10']):.6f}")
        print(f"pd_at_pfa_0p05: {float(summary['pd_at_pfa_0p05']):.6f}")
        print(f"metrics_csv: {eval_out_dir / 'sensing_metrics.csv'}")
        print(f"figure_roc: {eval_out_dir / 'sensing_roc.png'}")
        print(f"figure_pd_vs_snr: {eval_out_dir / 'pd_vs_snr.png'}")
        return 0

    result = run_energy_detection_from_split(
        h5_path,
        split_path,
        split_name=args.split_name,
        snr_filter=args.snr_filter,
        max_samples=args.max_samples,
        scan_chunk_size=args.scan_chunk_size,
        seed=args.seed,
        num_thresholds=args.num_thresholds,
    )

    metrics_path = metrics_dir / "energy_detection.csv"
    roc_path = figures_dir / "energy_roc.png"
    pd_pfa_path = figures_dir / "pd_pfa_vs_snr.png"

    result.metrics.to_csv(metrics_path, index=False)
    _plot_roc_curve(roc_path, result.roc_curve)
    _plot_pd_pfa_vs_snr(pd_pfa_path, result.pd_vs_snr)

    print(f"delegated_conda_env: {delegated_env_name() or '<none>'}")
    print(f"h5_path: {h5_path}")
    print(f"split_path: {split_path}")
    print(f"auc: {result.auc_value:.6f}")
    print(f"num_signal_samples: {result.num_signal_samples}")
    print(f"num_noise_samples: {result.num_noise_samples}")
    print(f"metrics_csv: {metrics_path}")
    print(f"figure_roc: {roc_path}")
    print(f"figure_pd_pfa_vs_snr: {pd_pfa_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create stratified train/val/test splits and split diagnostics for RadioML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from _bootstrap import delegate_to_conda_if_needed, delegated_env_name


delegate_to_conda_if_needed(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfml.data.radioml2018 import RadioML2018Dataset, build_label_name_map, infer_class_names_from_h5
from rfml.data.splits import (
    build_split_report,
    create_stratified_splits_from_h5,
    save_split_bundle,
    scan_labels_and_snrs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GOLD_XYZ_OSC.0001_1024.hdf5")
    parser.add_argument("--out", required=True, help="Output .npz path for split indices")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--figures-dir",
        default="outputs/figures",
        help="Directory for generated figures",
    )
    parser.add_argument(
        "--scan-chunk-size",
        type=int,
        default=8192,
        help="Chunk size for metadata scanning",
    )
    parser.add_argument(
        "--example-index",
        type=int,
        default=None,
        help="Optional dataset index for the waveform/STFT figure; defaults to first train sample.",
    )
    return parser.parse_args()


def _prepare_output_dirs(split_path: Path, figures_dir: Path) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)


def _plot_iq_examples(
    dataset: RadioML2018Dataset,
    sample_index: int,
    label_name: str,
    snr: float,
    output_path: Path,
) -> None:
    sample = dataset[sample_index]
    iq = sample["iq"].numpy()
    complex_signal = iq[0] + 1j * iq[1]
    amplitude = np.abs(complex_signal)
    phase = np.angle(complex_signal)

    _, _, stft_matrix = signal.stft(
        complex_signal,
        nperseg=128,
        noverlap=96,
        return_onesided=False,
    )
    stft_db = 20.0 * np.log10(np.abs(np.fft.fftshift(stft_matrix, axes=0)) + 1e-8)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    fig.suptitle(f"index={sample['index']} label={label_name} snr={snr:g} dB", fontsize=14)

    axes[0].plot(iq[0], label="I", linewidth=0.8)
    axes[0].plot(iq[1], label="Q", linewidth=0.8)
    axes[0].set_title("I/Q Waveform")
    axes[0].set_ylabel("amplitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(amplitude, color="tab:orange", linewidth=0.8)
    axes[1].set_title("Amplitude")
    axes[1].set_ylabel("|x|")
    axes[1].grid(alpha=0.25)

    axes[2].plot(phase, color="tab:green", linewidth=0.8)
    axes[2].set_title("Phase")
    axes[2].set_ylabel("rad")
    axes[2].grid(alpha=0.25)

    image = axes[3].imshow(
        stft_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
    )
    axes[3].set_title("STFT Spectrogram")
    axes[3].set_ylabel("frequency bin")
    axes[3].set_xlabel("frame")
    fig.colorbar(image, ax=axes[3], label="dB")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_distribution(
    labels: np.ndarray,
    split_indices_by_name: dict[str, np.ndarray],
    output_path: Path,
    label_name_map: dict[int, str],
) -> None:
    unique_labels = sorted(np.unique(labels).tolist())
    label_positions = np.arange(len(unique_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    for offset, (split_name, indices) in zip(
        (-width, 0.0, width),
        split_indices_by_name.items(),
        strict=True,
    ):
        split_labels = labels[indices]
        counts = [int(np.sum(split_labels == label)) for label in unique_labels]
        ax.bar(label_positions + offset, counts, width=width, label=split_name)

    ax.set_xticks(label_positions)
    ax.set_xticklabels([label_name_map[label] for label in unique_labels], rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Class Distribution by Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_snr_distribution(
    snrs: np.ndarray,
    split_indices_by_name: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    unique_snrs = sorted(np.unique(snrs).tolist())
    snr_positions = np.arange(len(unique_snrs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for offset, (split_name, indices) in zip(
        (-width, 0.0, width),
        split_indices_by_name.items(),
        strict=True,
    ):
        split_snrs = snrs[indices]
        counts = [int(np.sum(split_snrs == snr)) for snr in unique_snrs]
        ax.bar(snr_positions + offset, counts, width=width, label=split_name)

    ax.set_xticks(snr_positions)
    ax.set_xticklabels([f"{snr:g}" for snr in unique_snrs])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("count")
    ax.set_title("SNR Distribution by Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5).expanduser().resolve()
    if not h5_path.exists():
        print(f"error: HDF5 file not found: {h5_path}")
        return 1

    out_path = Path(args.out).expanduser().resolve()
    figures_dir = Path(args.figures_dir).expanduser().resolve()
    _prepare_output_dirs(out_path, figures_dir)

    class_names = infer_class_names_from_h5(h5_path)
    bundle = create_stratified_splits_from_h5(
        h5_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        scan_chunk_size=args.scan_chunk_size,
        class_names=class_names,
    )
    save_split_bundle(bundle, out_path)

    labels, snrs = scan_labels_and_snrs(h5_path, scan_chunk_size=args.scan_chunk_size)
    report = build_split_report(labels, snrs, bundle)
    label_name_map = build_label_name_map(int(labels.max()) + 1, bundle.class_names)

    split_indices_by_name = {
        "train": bundle.train_indices,
        "val": bundle.val_indices,
        "test": bundle.test_indices,
    }

    example_dataset = RadioML2018Dataset(
        h5_path,
        split_indices=bundle.train_indices,
        class_names=bundle.class_names,
        scan_chunk_size=args.scan_chunk_size,
    )
    example_index = args.example_index if args.example_index is not None else 0
    if len(example_dataset) == 0:
        print("error: train split is empty")
        return 1
    if example_index < 0 or example_index >= len(example_dataset):
        print(
            f"error: example_index {example_index} is out of range for train split size {len(example_dataset)}"
        )
        return 1

    example_sample = example_dataset[example_index]
    example_label = int(example_sample["label"].item())
    example_snr = float(example_sample["snr"].item())

    iq_examples_path = figures_dir / "iq_examples.png"
    snr_distribution_path = figures_dir / "snr_distribution.png"
    class_distribution_path = figures_dir / "class_distribution.png"

    _plot_iq_examples(
        example_dataset,
        example_index,
        label_name_map[example_label],
        example_snr,
        iq_examples_path,
    )
    _plot_snr_distribution(snrs, split_indices_by_name, snr_distribution_path)
    _plot_distribution(labels, split_indices_by_name, class_distribution_path, label_name_map)

    print(f"delegated_conda_env: {delegated_env_name() or '<none>'}")
    print(f"h5_path: {h5_path}")
    print(f"split_out: {out_path}")
    print(
        "note: exact 70/15/15 may not be possible for very small per-(label,SNR) groups; "
        "the split remains strictly stratified by modulation×SNR."
    )
    for split_name, summary in report.items():
        print(f"{split_name}: size={summary['size']}")
    print(f"figure_iq_examples: {iq_examples_path}")
    print(f"figure_snr_distribution: {snr_distribution_path}")
    print(f"figure_class_distribution: {class_distribution_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

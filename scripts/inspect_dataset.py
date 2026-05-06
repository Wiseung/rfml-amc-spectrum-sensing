#!/usr/bin/env python3
"""Inspect RadioML 2018.01A HDF5 metadata and sample waveforms."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _bootstrap import delegate_to_conda_if_needed, delegated_env_name


delegate_to_conda_if_needed(__file__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rfml.data.radioml2018 import (
    RadioML2018Dataset,
    build_label_name_map,
    infer_class_names_from_h5,
    load_class_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Path to GOLD_XYZ_OSC.0001_1024.hdf5")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the selected sample count after filters are applied.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        nargs="*",
        default=None,
        help="Optional SNR filter, for example: --snr -20 -10 0 10",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class filter as label ids or class names.",
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Optional plain-text file with one class name per line.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/inspect_dataset",
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for waveform sampling.",
    )
    return parser.parse_args()


def _parse_classes(raw_values: list[str] | None) -> list[int | str] | None:
    if raw_values is None:
        return None
    parsed: list[int | str] = []
    for value in raw_values:
        try:
            parsed.append(int(value))
        except ValueError:
            parsed.append(value)
    return parsed


def _plot_random_iq_samples(
    dataset: RadioML2018Dataset,
    label_names: dict[int, str],
    output_path: Path,
    *,
    seed: int,
    count: int = 8,
) -> None:
    if len(dataset) == 0:
        return

    rng = random.Random(seed)
    actual_count = min(count, len(dataset))
    selected = rng.sample(range(len(dataset)), k=actual_count)

    fig, axes = plt.subplots(actual_count, 1, figsize=(14, 2.5 * actual_count), sharex=True)
    if actual_count == 1:
        axes = [axes]

    for axis, sample_idx in zip(axes, selected, strict=True):
        sample = dataset[sample_idx]
        iq = sample["iq"].numpy()
        label = int(sample["label"].item())
        snr = float(sample["snr"].item())
        index = sample["index"]

        axis.plot(iq[0], label="I", linewidth=0.8)
        axis.plot(iq[1], label="Q", linewidth=0.8)
        axis.set_title(
            f"sample_index={index} label={label_names[label]} ({label}) snr={snr:g} dB"
        )
        axis.set_ylabel("amplitude")
        axis.grid(alpha=0.25)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = Path(args.h5).expanduser().resolve()
    if not h5_path.exists():
        print(f"error: HDF5 file not found: {h5_path}")
        return 1

    class_names = None
    if args.class_names:
        class_names = load_class_names(args.class_names)
    else:
        class_names = infer_class_names_from_h5(h5_path)

    dataset = RadioML2018Dataset(
        h5_path,
        snr_filter=args.snr,
        class_filter=_parse_classes(args.classes),
        max_samples=args.max_samples,
        class_names=class_names,
    )
    summary = dataset.describe()
    label_names = build_label_name_map(dataset.num_classes, class_names)

    print(f"delegated_conda_env: {delegated_env_name() or '<none>'}")
    print(f"h5_path: {summary['h5_path']}")
    print(f"样本总数: {summary['num_selected_samples']} (原始总数: {summary['num_total_samples']})")
    print(f"类别数: {summary['num_classes']}")
    print(f"SNR 列表: {summary['snr_values']}")
    print("每类样本统计:")
    for key, count in summary["class_counts"].items():
        print(f"  {key}: {count}")
    print("每个 SNR 样本统计:")
    for snr, count in summary["snr_counts"].items():
        print(f"  {snr:g}: {count}")

    plot_path = output_dir / "random_iq_waveforms.png"
    _plot_random_iq_samples(dataset, label_names, plot_path, seed=args.seed)
    if plot_path.exists():
        print(f"随机 8 个 IQ 波形图: {plot_path}")
    else:
        print("随机 8 个 IQ 波形图: dataset is empty, no plot generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

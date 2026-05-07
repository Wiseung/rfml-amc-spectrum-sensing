#!/usr/bin/env python3
"""Plot representative STFT spectrograms for different modulation classes."""

from __future__ import annotations

import argparse
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

from rfml.data.radioml2018 import RadioML2018Dataset, build_label_name_map, infer_class_names_from_h5
from rfml.data.transforms import STFTTransform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--out", default="outputs/figures/stft_spectrogram_examples.png")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--backend", default="torch", choices=["torch", "scipy"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5).expanduser().resolve()
    if not h5_path.exists():
        print(f"error: HDF5 file not found: {h5_path}")
        return 1

    transform = STFTTransform(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        output="log_power",
        backend=args.backend,
    )
    class_names = infer_class_names_from_h5(h5_path)
    dataset = RadioML2018Dataset(
        h5_path,
        class_names=class_names,
        snr_filter=[args.snr],
    )
    label_name_map = build_label_name_map(dataset.num_classes, class_names)

    examples = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        label = int(sample["label"].item())
        if label not in examples:
            examples[label] = sample
        if len(examples) >= args.num_classes:
            break

    if not examples:
        print("error: no samples available for the requested SNR")
        return 1

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 3 * len(examples)))
    if len(examples) == 1:
        axes = [axes]

    for axis, label in zip(axes, sorted(examples), strict=True):
        sample = examples[label]
        spec = transform(sample["iq"]).numpy()[0]
        image = axis.imshow(spec, aspect="auto", origin="lower", cmap="magma")
        axis.set_title(f"{label_name_map[label]} | SNR={float(sample['snr'].item()):g} dB")
        axis.set_ylabel("freq bin")
        fig.colorbar(image, ax=axis, label="log power")

    axes[-1].set_xlabel("time frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"delegated_conda_env: {delegated_env_name() or '<none>'}")
    print(f"spectrogram_figure: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

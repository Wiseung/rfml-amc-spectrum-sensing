# RFML AMC Spectrum Sensing

PyTorch project scaffold for automatic modulation classification (AMC) and spectrum sensing on the RadioML 2018.01A dataset.

This repository is being built in phases. Phases 0 to 6 currently cover:

- project structure
- environment validation
- lazy-loading RadioML 2018.01A dataset access
- stratified modulation x SNR split generation
- traditional baselines for AMC and spectrum sensing
- 1D CNN training and evaluation
- ResNet1D training and evaluation
- STFT spectrogram generation and STFT-CNN training

## Project Layout

```text
rfml-amc-spectrum-sensing/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── cnn1d.yaml
│   ├── resnet1d.yaml
│   ├── stft_cnn.yaml
│   └── multitask.yaml
├── data/
│   └── .gitkeep
├── src/
│   └── rfml/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── radioml2018.py
│       │   ├── splits.py
│       │   └── transforms.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── cnn1d.py
│       │   ├── resnet1d.py
│       │   ├── stft_cnn.py
│       │   └── multitask.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── metrics.py
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── evaluate.py
│       │   ├── plot_snr.py
│       │   ├── plot_confusion.py
│       │   └── sensing_metrics.py
│       └── baselines/
│           ├── __init__.py
│           ├── energy_detection.py
│           ├── cumulants.py
│           └── sklearn_baselines.py
├── scripts/
│   ├── check_env.py
│   ├── inspect_dataset.py
│   ├── make_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_sensing.py
│   └── smoke_test.py
├── notebooks/
│   └── 00_dataset_preview.ipynb
├── outputs/
│   └── .gitkeep
└── reports/
    └── experiment_report.md
```

## Environment

Recommended:

- Python 3.10 to 3.12
- NVIDIA GPU with CUDA support
- single-GPU training on RTX 5090 24 GB

Important on this machine:

- shell `python3` currently points to Anaconda Python 3.13
- `/usr/bin/python3` is system Python 3.12
- Phase 0 scripts support both, but PyTorch wheels may be easier to manage in a dedicated virtual environment or Conda env using Python 3.10 to 3.12

## Install

Using `venv` with system Python:

```bash
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you want a CUDA-enabled PyTorch build, install the correct wheel for your environment first, then install the rest:

```bash
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Phase 0 Validation

Check environment:

```bash
python scripts/check_env.py
```

Run smoke test:

```bash
python scripts/smoke_test.py
```

Both scripts are designed to fail clearly if `torch` is missing or CUDA is unavailable, while still printing actionable diagnostics.

## Dataset Path

Expected target dataset:

```text
data/GOLD_XYZ_OSC.0001_1024.hdf5
```

Current code expects:

- lazy HDF5 access to `X`, `Y`, and `Z` with `h5py`
- SNR/modulation filters and split-index driven sampling
- train/val/test split generation under `outputs/splits`
- AMC and spectrum sensing experiment pipelines

## Planned Commands

Current commands:

```bash
python scripts/inspect_dataset.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --max-samples 4096

python scripts/make_splits.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --out outputs/splits/radioml2018_seed42.npz \
  --seed 42

python scripts/run_sensing.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --method energy \
  --split outputs/splits/radioml2018_seed42.npz

python scripts/train.py \
  --config configs/cnn1d.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/cnn1d_seed42

python scripts/train.py \
  --config configs/resnet1d.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/resnet1d_seed42

python scripts/train.py \
  --config configs/stft_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/stft_cnn_seed42

python scripts/train.py \
  --config configs/sensing_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/sensing_cnn_seed42

python scripts/train.py \
  --config configs/multitask.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/multitask_seed42

python scripts/evaluate.py \
  --config configs/cnn1d.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/cnn1d_seed42/best.pt

python scripts/evaluate.py \
  --config configs/resnet1d.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/resnet1d_seed42/best.pt

python scripts/evaluate.py \
  --config configs/stft_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/stft_cnn_seed42/best.pt

python scripts/evaluate.py \
  --config configs/sensing_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/sensing_cnn_seed42/best.pt

python scripts/evaluate.py \
  --config configs/multitask.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/multitask_seed42/best.pt

python scripts/plot_spectrograms.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --out outputs/figures/stft_spectrogram_examples.png \
  --num-classes 8 \
  --snr 10 \
  --n-fft 128 \
  --hop-length 32

python scripts/compare_results.py \
  --baseline-acc-vs-snr outputs/baselines/svm_accuracy_vs_snr.csv \
  --baseline-overall-acc 0.70 \
  --cnn-run-dir outputs/runs/cnn1d_seed42 \
  --resnet-run-dir outputs/runs/resnet1d_seed42 \
  --stft-run-dir outputs/runs/stft_cnn_seed42 \
  --out-dir outputs/comparisons

python scripts/run_sensing.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --method cnn \
  --split outputs/splits/radioml2018_seed42.npz \
  --config configs/sensing_cnn.yaml \
  --ckpt outputs/runs/sensing_cnn_seed42/best.pt \
  --eval-out-dir outputs/runs/sensing_cnn_seed42
```

## Reproducibility Roadmap

1. Phase 0: scaffold, environment checks, smoke tests
2. Phase 1: RadioML lazy-loading dataset and split tooling
3. Phase 2: stratified splits and dataset visualization
4. Phase 3: baselines for AMC and spectrum sensing
5. Phase 4: CNN1D training pipeline and evaluation
6. Phase 5: ResNet1D / MRResNet experiments
7. Phase 6: STFT spectrogram transform and STFT-CNN experiments
8. Phase 7: deep spectrum sensing with binary CNN detector
9. Phase 8: multi-task AMC plus spectrum sensing model

## Initial CNN1D Notes

- Default training config is [configs/cnn1d.yaml](/home/developer716/workspace/rfml-amc-spectrum-sensing/configs/cnn1d.yaml).
- The intended starting point for this RTX 5090 24 GB laptop is `batch_size: 512`.
- If thermals and power headroom allow, try `1024` or `2048`.
- During long runs, monitor GPU memory, power, and temperature:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw --format=csv,noheader
```

## STFT-CNN Notes

- `STFTTransform` lives in [src/rfml/data/transforms.py](/home/developer716/workspace/rfml-amc-spectrum-sensing/src/rfml/data/transforms.py) and currently supports `torch.stft` or `scipy.signal.stft`.
- Configurable STFT parameters include `stft_n_fft`, `stft_hop_length`, `stft_window`, `stft_output`, and `stft_backend`.
- The default [configs/stft_cnn.yaml](/home/developer716/workspace/rfml-amc-spectrum-sensing/configs/stft_cnn.yaml) uses `n_fft=128`, `hop_length=32`, `window=hann`, and `output=log_power`.
- On real RadioML data, use the same split file as the 1D models so `cnn1d`, `resnet1d`, and `stft_cnn` can be compared fairly by SNR.

## Spectrum Sensing Notes

- `[src/rfml/data/spectrum_sensing.py](/home/developer716/workspace/rfml-amc-spectrum-sensing/src/rfml/data/spectrum_sensing.py)` builds a binary detection dataset from RadioML signal samples and lazily generated AWGN noise-only samples.
- Positive samples are labeled `1` and negative noise-only samples are labeled `0`.
- `[configs/sensing_cnn.yaml](/home/developer716/workspace/rfml-amc-spectrum-sensing/configs/sensing_cnn.yaml)` reuses the 1D CNN backbone with `num_classes=2`.
- The evaluation pipeline writes `sensing_metrics.csv`, `sensing_roc_curve.csv`, `pd_vs_snr.csv`, `sensing_roc.png`, and `pd_vs_snr.png`.
- Reported sensing metrics include `Accuracy`, `ROC-AUC`, `Pd@Pfa=0.10`, `Pd@Pfa=0.05`, and `Pd vs SNR`.

## Multi-Task Notes

- `[src/rfml/data/multitask.py](/home/developer716/workspace/rfml-amc-spectrum-sensing/src/rfml/data/multitask.py)` mixes RadioML signal samples with synthetic AWGN noise-only samples in one dataset.
- `[src/rfml/models/multitask.py](/home/developer716/workspace/rfml-amc-spectrum-sensing/src/rfml/models/multitask.py)` implements a shared encoder with two heads: `modulation_logits` and `sensing_logits`.
- Noise-only samples participate in sensing supervision only; their modulation loss is masked out during training.
- `[configs/multitask.yaml](/home/developer716/workspace/rfml-amc-spectrum-sensing/configs/multitask.yaml)` uses `loss = loss_modulation + lambda_sensing * loss_sensing`.
- Multi-task evaluation writes modulation `accuracy_vs_snr` plus sensing `ROC / AUC / Pd@Pfa / Pd vs SNR` artifacts in the same run directory.

## Current Results Snapshot

- Phase 4 code path supports AMP, checkpoint, resume, CSV log, TensorBoard, overall accuracy, accuracy vs SNR, and confusion matrix outputs.
- Phase 5 adds ResNet1D-small and ResNet1D-medium with the same trainer/evaluate pipeline plus comparison-table tooling.
- Phase 6 adds spectrogram plotting, STFT preprocessing, and STFT-CNN training/evaluation support with the same checkpoint and reporting flow.
- Phase 7 adds deep spectrum sensing with a binary CNN detector, AWGN negative-sample synthesis, and ROC/Pd/Pfa reporting.
- Phase 8 adds a shared-encoder multi-task model for AMC plus spectrum sensing.
- Real RadioML training/evaluation artifacts still depend on placing `GOLD_XYZ_OSC.0001_1024.hdf5` under `data/`.

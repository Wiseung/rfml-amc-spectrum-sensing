# RFML AMC Spectrum Sensing

PyTorch project scaffold for automatic modulation classification (AMC) and spectrum sensing on the RadioML 2018.01A dataset.

This repository is being built in phases. Phase 0 focuses on:

- project structure
- environment validation
- import and CUDA smoke tests
- a minimal package layout that future phases can extend safely

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

Planned target dataset:

```text
data/GOLD_XYZ_OSC.0001_1024.hdf5
```

Future phases will add:

- lazy HDF5 dataset loading with `h5py`
- SNR/modulation filtering
- train/val/test split generation
- AMC and spectrum sensing pipelines

## Planned Commands

These commands are placeholders for later phases:

```bash
python scripts/inspect_dataset.py --help
python scripts/make_splits.py --help
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/run_sensing.py --help
```

## Reproducibility Roadmap

1. Phase 0: scaffold, environment checks, smoke tests
2. Phase 1: RadioML lazy-loading dataset and split tooling
3. Phase 2: baselines for AMC and spectrum sensing
4. Phase 3: CNN1D, ResNet1D, STFT-CNN training pipeline
5. Phase 4: evaluation, plotting, and experiment report
6. Phase 5: multi-task AMC plus spectrum sensing model

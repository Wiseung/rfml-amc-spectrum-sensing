# RFML AMC Spectrum Sensing

PyTorch project scaffold for automatic modulation classification (AMC) and spectrum sensing on the RadioML 2018.01A dataset.

This repository is being built in phases. Phases 0 to 4 currently cover:

- project structure
- environment validation
- lazy-loading RadioML 2018.01A dataset access
- stratified modulation x SNR split generation
- traditional baselines for AMC and spectrum sensing
- 1D CNN training and evaluation

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

python scripts/compare_results.py \
  --baseline-acc-vs-snr outputs/baselines/svm_accuracy_vs_snr.csv \
  --baseline-overall-acc 0.70 \
  --cnn-run-dir outputs/runs/cnn1d_seed42 \
  --resnet-run-dir outputs/runs/resnet1d_seed42 \
  --out-dir outputs/comparisons
```

## Reproducibility Roadmap

1. Phase 0: scaffold, environment checks, smoke tests
2. Phase 1: RadioML lazy-loading dataset and split tooling
3. Phase 2: baselines for AMC and spectrum sensing
4. Phase 3: CNN1D training pipeline and evaluation
5. Phase 4: ResNet1D / MRResNet, STFT-CNN, and broader experiments
6. Phase 5: multi-task AMC plus spectrum sensing model

## Initial CNN1D Notes

- Default training config is [configs/cnn1d.yaml](/home/developer716/workspace/rfml-amc-spectrum-sensing/configs/cnn1d.yaml).
- The intended starting point for this RTX 5090 24 GB laptop is `batch_size: 512`.
- If thermals and power headroom allow, try `1024` or `2048`.
- During long runs, monitor GPU memory, power, and temperature:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw --format=csv,noheader
```

## Current Results Snapshot

- Phase 4 code path supports AMP, checkpoint, resume, CSV log, TensorBoard, overall accuracy, accuracy vs SNR, and confusion matrix outputs.
- Phase 5 adds ResNet1D-small and ResNet1D-medium with the same trainer/evaluate pipeline plus comparison-table tooling.
- Real RadioML training/evaluation artifacts still depend on placing `GOLD_XYZ_OSC.0001_1024.hdf5` under `data/`.

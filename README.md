# RFML AMC Spectrum Sensing

PyTorch project for automatic modulation classification (AMC) and spectrum sensing on the RadioML 2018.01A dataset.

The repository is designed for single-GPU training on an RTX 5090 24 GB class device and keeps the large RadioML HDF5 file on disk. Dataset access is lazy by default: samples are read with `h5py` inside `Dataset.__getitem__`, not loaded into RAM up front.

## Status

Implemented through Phase 8:

- Phase 0: project scaffold, environment checks, smoke tests
- Phase 1: lazy RadioML 2018.01A dataset loader with HDF5 worker safety
- Phase 2: stratified modulation x SNR split generation and dataset figures
- Phase 3: AMC sklearn baselines and energy detection sensing baseline
- Phase 4: CNN1D training and evaluation pipeline
- Phase 5: ResNet1D-small / ResNet1D-medium
- Phase 6: STFT transform, spectrogram plotting, STFT-CNN
- Phase 7: deep spectrum sensing with AWGN negative samples
- Phase 8: multi-task AMC plus spectrum sensing model

Current code paths are fully smoke-tested on tiny synthetic datasets. Real experiment metrics still require placing `GOLD_XYZ_OSC.0001_1024.hdf5` under `data/`.

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
│   ├── sensing_cnn.yaml
│   ├── stft_cnn.yaml
│   └── multitask.yaml
├── data/
├── src/rfml/
│   ├── baselines/
│   ├── data/
│   ├── eval/
│   ├── models/
│   └── training/
├── scripts/
├── notebooks/
├── outputs/
└── reports/
```

## Environment

Recommended:

- Python `3.10` to `3.12`
- CUDA-capable PyTorch build
- single GPU training

Machine-specific note for this workstation:

- shell `python3` may point to Anaconda Python
- `/usr/bin/python3` is the safer system interpreter
- project scripts use [`scripts/_bootstrap.py`](scripts/_bootstrap.py) to delegate to a known Conda env if `torch` is missing in the current interpreter

## Installation

Using `venv`:

```bash
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you want a specific CUDA wheel, install the matching `torch` build first, then install the rest:

```bash
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Dataset

Expected dataset path:

```text
data/GOLD_XYZ_OSC.0001_1024.hdf5
```

The dataset loader in [src/rfml/data/radioml2018.py](/home/developer716/workspace/rfml-amc-spectrum-sensing/src/rfml/data/radioml2018.py) supports:

- lazy HDF5 access to `X`, `Y`, `Z`
- `X: (1024, 2) -> Tensor(2, 1024)`
- one-hot `Y -> int` label
- `Z -> snr`
- `snr_filter`, `class_filter`, `max_samples`, `split_indices`
- multi-worker safe HDF5 reopening

Returned sample format:

```python
{
    "iq": Tensor[2, 1024],
    "label": LongTensor[],
    "snr": FloatTensor[],
    "index": int,
}
```

## Phase 0 Validation

Check the runtime:

```bash
python scripts/check_env.py
```

Run the import and random-forward smoke test:

```bash
python scripts/smoke_test.py
```

## Data Inspection And Splits

Inspect dataset metadata and draw random IQ waveforms:

```bash
python scripts/inspect_dataset.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --max-samples 4096
```

Create the stratified `train/val/test = 70/15/15` split:

```bash
python scripts/make_splits.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --out outputs/splits/radioml2018_seed42.npz \
  --seed 42
```

Expected outputs:

- `outputs/splits/radioml2018_seed42.npz`
- `outputs/figures/iq_examples.png`
- `outputs/figures/snr_distribution.png`
- `outputs/figures/class_distribution.png`

## Traditional Baselines

### AMC sklearn baseline

The statistical-feature baseline is implemented in [src/rfml/baselines/sklearn_baselines.py](/home/developer716/workspace/rfml-amc-spectrum-sensing/src/rfml/baselines/sklearn_baselines.py). It extracts light-weight features from IQ sequences and supports `logreg`, `svm`, `rf`, and `gb`.

Example command:

```bash
python - <<'PY'
from pathlib import Path
import json

from rfml.baselines.sklearn_baselines import run_sklearn_baseline

result = run_sklearn_baseline(
    h5_path="data/GOLD_XYZ_OSC.0001_1024.hdf5",
    split_path="outputs/splits/radioml2018_seed42.npz",
    classifier_name="svm",
    max_train_samples=20000,
    max_eval_samples=5000,
    random_state=42,
)

out_dir = Path("outputs/baselines")
out_dir.mkdir(parents=True, exist_ok=True)
result.accuracy_vs_snr.to_csv(out_dir / "svm_accuracy_vs_snr.csv", index=False)
(out_dir / "svm_classification_report.txt").write_text(result.classification_report_text, encoding="utf-8")
(out_dir / "svm_summary.json").write_text(
    json.dumps(
        {
            "classifier": result.classifier_name,
            "train_accuracy": result.train_accuracy,
            "eval_accuracy": result.eval_accuracy,
            "feature_dim": result.feature_dim,
            "train_size": result.train_size,
            "eval_size": result.eval_size,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("baseline_eval_accuracy:", result.eval_accuracy)
print("baseline_accuracy_vs_snr:", out_dir / "svm_accuracy_vs_snr.csv")
PY
```

### Energy detection sensing baseline

```bash
python scripts/run_sensing.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --method energy \
  --split outputs/splits/radioml2018_seed42.npz
```

Expected outputs:

- `outputs/metrics/energy_detection.csv`
- `outputs/figures/energy_roc.png`
- `outputs/figures/pd_pfa_vs_snr.png`

## Deep AMC Training

### CNN1D

```bash
python scripts/train.py \
  --config configs/cnn1d.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/cnn1d_seed42
```

Evaluate:

```bash
python scripts/evaluate.py \
  --config configs/cnn1d.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/cnn1d_seed42/best.pt
```

### ResNet1D

```bash
python scripts/train.py \
  --config configs/resnet1d.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/resnet1d_seed42
```

```bash
python scripts/evaluate.py \
  --config configs/resnet1d.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/resnet1d_seed42/best.pt
```

### STFT-CNN

Plot representative spectrograms:

```bash
python scripts/plot_spectrograms.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --out outputs/figures/stft_spectrogram_examples.png \
  --num-classes 8 \
  --snr 10 \
  --n-fft 128 \
  --hop-length 32
```

Train:

```bash
python scripts/train.py \
  --config configs/stft_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/stft_cnn_seed42
```

Evaluate:

```bash
python scripts/evaluate.py \
  --config configs/stft_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/stft_cnn_seed42/best.pt
```

## Deep Spectrum Sensing

Train the binary CNN detector:

```bash
python scripts/train.py \
  --config configs/sensing_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/sensing_cnn_seed42
```

Evaluate:

```bash
python scripts/evaluate.py \
  --config configs/sensing_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/sensing_cnn_seed42/best.pt
```

Or use the sensing wrapper:

```bash
python scripts/run_sensing.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --method cnn \
  --split outputs/splits/radioml2018_seed42.npz \
  --config configs/sensing_cnn.yaml \
  --ckpt outputs/runs/sensing_cnn_seed42/best.pt \
  --eval-out-dir outputs/runs/sensing_cnn_seed42
```

## Multi-Task AMC Plus Sensing

Train:

```bash
python scripts/train.py \
  --config configs/multitask.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/multitask_seed42
```

Evaluate:

```bash
python scripts/evaluate.py \
  --config configs/multitask.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/multitask_seed42/best.pt
```

## Comparison

Build compact comparison tables and an `accuracy vs SNR` overlay:

```bash
python scripts/compare_results.py \
  --baseline-acc-vs-snr outputs/baselines/svm_accuracy_vs_snr.csv \
  --baseline-overall-acc 0.70 \
  --cnn-run-dir outputs/runs/cnn1d_seed42 \
  --resnet-run-dir outputs/runs/resnet1d_seed42 \
  --stft-run-dir outputs/runs/stft_cnn_seed42 \
  --out-dir outputs/comparisons
```

The `--baseline-overall-acc` value should be replaced with the actual SVM eval accuracy from your baseline summary.

## Training Features

The trainer in [src/rfml/training/trainer.py](/home/developer716/workspace/rfml-amc-spectrum-sensing/src/rfml/training/trainer.py) supports:

- YAML config
- AMP mixed precision
- checkpoint save and resume
- CSV log and TensorBoard log
- gradient clipping
- early stopping
- AMC, sensing, and multi-task training modes

Typical run artifacts:

- `best.pt`
- `last.pt`
- `train_log.csv`
- `history.json`
- TensorBoard event files

Evaluation artifacts:

- `summary.json`
- `accuracy_vs_snr.csv` or `modulation_accuracy_vs_snr.csv`
- `confusion_matrix.csv`
- `classification_report.txt`
- `confusion_matrix.png`
- `acc_vs_snr.png`
- sensing-specific ROC and Pd/Pfa files when applicable

## Notebook And Report

- Dataset preview notebook: [notebooks/00_dataset_preview.ipynb](/home/developer716/workspace/rfml-amc-spectrum-sensing/notebooks/00_dataset_preview.ipynb)
- Experiment report template: [reports/experiment_report.md](/home/developer716/workspace/rfml-amc-spectrum-sensing/reports/experiment_report.md)

## Reproducibility Flow

1. Install dependencies and verify the environment.
2. Place `GOLD_XYZ_OSC.0001_1024.hdf5` under `data/`.
3. Inspect the dataset with `scripts/inspect_dataset.py`.
4. Create `outputs/splits/radioml2018_seed42.npz`.
5. Run sklearn AMC baseline and energy detection baseline.
6. Train and evaluate `cnn1d`.
7. Train and evaluate `resnet1d`.
8. Plot STFT spectrograms, then train and evaluate `stft_cnn`.
9. Train and evaluate `sensing_cnn`.
10. Train and evaluate `multitask`.
11. Run `scripts/compare_results.py`.
12. Fill the report in `reports/experiment_report.md` with real metrics and figures.

## Smoke-Test Scope

Smoke tests cover:

- import and CUDA detection
- random tensor forward passes
- synthetic HDF5 dataset loading
- split generation
- baseline execution
- trainer loop for AMC, sensing, STFT, ResNet1D, and multi-task

Smoke tests do not claim real RadioML leaderboard-quality results. They only confirm that the engineering pipeline runs end to end.

## GPU Notes

The default starting point for this RTX 5090 24 GB laptop is `batch_size: 512` for 1D models. If thermals allow, you can try `1024` or `2048`.

Monitor GPU state during long runs:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw --format=csv,noheader
```

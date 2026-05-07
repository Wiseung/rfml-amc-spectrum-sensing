# Experiment Report

## 1. Project Goal

This project targets two related RF machine learning tasks on RadioML 2018.01A:

- automatic modulation classification, 24-way classification
- spectrum sensing, binary `noise` vs `signal`

An additional multi-task model combines both tasks with a shared encoder.

## 2. Dataset And Access Pattern

Dataset file:

```text
data/GOLD_XYZ_OSC.0001_1024.hdf5
```

The implementation uses lazy HDF5 access and does not load the full `X` array into memory. Each sample is read on demand inside `Dataset.__getitem__`, which keeps the project usable on commodity RAM sizes while still supporting multi-worker `DataLoader` training.

Core fields:

- `X`: IQ waveform, expected raw shape `(1024, 2)`
- `Y`: one-hot modulation label
- `Z`: SNR

Returned training sample shape:

```python
{
    "iq": Tensor[2, 1024],
    "label": LongTensor[],
    "snr": FloatTensor[],
    "index": int,
}
```

## 3. Split Strategy

Split ratios:

- train: `70%`
- val: `15%`
- test: `15%`

The split is stratified by `(modulation label, SNR)` rather than by label only. This matters because AMC accuracy is strongly SNR-dependent and unbalanced SNR buckets can distort the evaluation.

Split artifact:

```text
outputs/splits/radioml2018_seed42.npz
```

Generation command:

```bash
python scripts/make_splits.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --out outputs/splits/radioml2018_seed42.npz \
  --seed 42
```

Generated figures:

- `outputs/figures/iq_examples.png`
- `outputs/figures/snr_distribution.png`
- `outputs/figures/class_distribution.png`

## 4. Baselines

### 4.1 AMC statistical-feature baselines

Implemented in `src/rfml/baselines/sklearn_baselines.py`.

Features include:

- I/Q mean and std
- amplitude mean and std
- amplitude skew and kurtosis
- phase-difference statistics
- instantaneous-frequency statistics

Supported classifiers:

- Logistic Regression
- SVM
- RandomForest
- GradientBoosting

Recommended first baseline:

- `svm`

### 4.2 Spectrum sensing baseline

Implemented in `src/rfml/baselines/energy_detection.py`.

Task definition:

- `H0`: noise only
- `H1`: signal plus noise

Negative samples are generated as matched AWGN sequences with the same IQ shape as the corresponding signal sample. The code reports:

- ROC
- AUC
- Pd
- Pfa
- Pd vs SNR

## 5. Deep Models

### 5.1 CNN1D

Input:

```text
(B, 2, 1024)
```

Architecture:

- Conv1d -> BN -> ReLU -> Pool
- Conv1d -> BN -> ReLU -> Pool
- Conv1d -> BN -> ReLU
- global average pooling
- MLP classifier

### 5.2 ResNet1D

Implemented variants:

- `resnet1d-small`
- `resnet1d-medium`

Default comparison model:

- `resnet1d-small`

### 5.3 STFT-CNN

The IQ waveform is mapped to a complex signal and transformed into a spectrogram with configurable STFT parameters:

- `n_fft`
- `hop_length`
- `window`
- output type: `magnitude`, `power`, `log_power`

### 5.4 Spectrum sensing CNN

This reuses the 1D CNN backbone with `num_classes=2`.

### 5.5 Multi-task model

The multi-task model uses a shared 1D encoder with:

- head 1: modulation classification
- head 2: signal detection

Loss:

```text
loss = loss_modulation + lambda_sensing * loss_sensing
```

Noise-only samples are masked out of the modulation loss.

## 6. Training Configuration

The trainer supports:

- YAML config
- AMP mixed precision
- checkpoint save and resume
- CSV log
- TensorBoard
- early stopping
- gradient clipping

For this machine, the default 1D model starting point is:

- `batch_size: 512`

Key config files:

- `configs/cnn1d.yaml`
- `configs/resnet1d.yaml`
- `configs/stft_cnn.yaml`
- `configs/sensing_cnn.yaml`
- `configs/multitask.yaml`

## 7. Reproduction Commands

### 7.1 Environment smoke test

```bash
python scripts/check_env.py
python scripts/smoke_test.py
```

### 7.2 Dataset inspection

```bash
python scripts/inspect_dataset.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --max-samples 4096
```

### 7.3 Split generation

```bash
python scripts/make_splits.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --out outputs/splits/radioml2018_seed42.npz \
  --seed 42
```

### 7.4 Energy detection baseline

```bash
python scripts/run_sensing.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --method energy \
  --split outputs/splits/radioml2018_seed42.npz
```

### 7.5 CNN1D AMC

```bash
python scripts/train.py \
  --config configs/cnn1d.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/cnn1d_seed42
```

```bash
python scripts/evaluate.py \
  --config configs/cnn1d.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/cnn1d_seed42/best.pt
```

### 7.6 ResNet1D AMC

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

### 7.7 STFT-CNN AMC

```bash
python scripts/plot_spectrograms.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --out outputs/figures/stft_spectrogram_examples.png \
  --num-classes 8 \
  --snr 10 \
  --n-fft 128 \
  --hop-length 32
```

```bash
python scripts/train.py \
  --config configs/stft_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/stft_cnn_seed42
```

```bash
python scripts/evaluate.py \
  --config configs/stft_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/stft_cnn_seed42/best.pt
```

### 7.8 Deep spectrum sensing

```bash
python scripts/train.py \
  --config configs/sensing_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/sensing_cnn_seed42
```

```bash
python scripts/evaluate.py \
  --config configs/sensing_cnn.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/sensing_cnn_seed42/best.pt
```

### 7.9 Multi-task AMC plus sensing

```bash
python scripts/train.py \
  --config configs/multitask.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/multitask_seed42
```

```bash
python scripts/evaluate.py \
  --config configs/multitask.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/multitask_seed42/best.pt
```

## 8. Metrics To Report

### 8.1 AMC

- overall accuracy
- accuracy vs SNR
- confusion matrix
- classification report

### 8.2 Spectrum sensing

- accuracy
- ROC-AUC
- Pd at `Pfa = 0.10`
- Pd at `Pfa = 0.05`
- Pd vs SNR

### 8.3 Multi-task

- modulation accuracy
- modulation accuracy vs SNR
- sensing ROC-AUC
- sensing Pd at fixed Pfa

## 9. Artifact Checklist

For each completed real-data run, keep:

- config file used
- split file
- training log CSV
- TensorBoard logs
- best checkpoint
- evaluation `summary.json`
- evaluation CSVs
- main figures

Recommended run directories:

- `outputs/runs/cnn1d_seed42`
- `outputs/runs/resnet1d_seed42`
- `outputs/runs/stft_cnn_seed42`
- `outputs/runs/sensing_cnn_seed42`
- `outputs/runs/multitask_seed42`

## 10. Smoke-Test Findings

The engineering pipeline has already passed local synthetic smoke tests:

- dataset loading
- split generation
- sklearn baseline execution
- energy detection sensing
- CNN1D training
- ResNet1D training
- STFT transform and STFT-CNN
- deep spectrum sensing
- multi-task training and evaluation

These smoke results verify code health and reproducibility plumbing, but they are not real RadioML experiment results.

## 11. Real-Experiment Table Template

Fill this section after running the full dataset:

| Model | Task | Overall Metric | Notes |
| --- | --- | --- | --- |
| SVM | AMC | TBD | statistical-feature baseline |
| CNN1D | AMC | TBD | first deep baseline |
| ResNet1D-small | AMC | TBD | residual 1D model |
| STFT-CNN | AMC | TBD | spectrogram model |
| Energy Detection | Sensing | TBD | non-deep baseline |
| CNN1D detector | Sensing | TBD | binary deep sensing |
| Multi-task | AMC + Sensing | TBD | shared encoder |

## 12. Discussion Points To Fill After Real Runs

- Does `accuracy vs SNR` rise clearly at high SNR for all AMC models?
- Does `ResNet1D` outperform plain `CNN1D` consistently?
- Is `STFT-CNN` more stable at low SNR?
- How much does deep sensing improve over energy detection?
- Does the multi-task model preserve AMC performance while improving sensing or efficiency?

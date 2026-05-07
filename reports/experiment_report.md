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

### 4.3 Round-1 real baseline results

All results below are from the real RadioML 2018.01A split
`outputs/splits/radioml2018_seed42.npz`.

AMC baselines:

- `SVM` with random subset sampling (`40k` train / `12k` eval): `0.3714`
- `RandomForest` with random subset sampling (`120k` train / `24k` eval): `0.4152`

Spectrum sensing baseline:

- `Energy Detection` test ROC-AUC: `0.5210`
- `best_pd = 0.0441` at `best_pfa = 0.00228`

Important note:

- an earlier `svm_summary.json = 0.7815` run was produced before fixing the subset sampling path and only covered two classes from the front of the file ordering
- that run is excluded from the final comparison

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

For real round-1 experiments, shorter configs were used:

- `configs/cnn1d_round1.yaml`
- `configs/resnet1d_round1.yaml`
- `configs/stft_cnn_round1.yaml`
- `configs/sensing_cnn_round1.yaml`
- `configs/multitask_round1.yaml`

These round-1 configs are intended to establish verified end-to-end results on
the real dataset before longer convergence runs.

## 7. Round-1 Real Results

### 7.1 Dataset and split

Verified on the real dataset:

- total samples: `2,555,904`
- modulation classes: `24`
- SNR values: `-20` to `30 dB` with step `2 dB`

Saved split:

- train: `1,789,008`
- val: `383,136`
- test: `383,760`

### 7.2 AMC results

Overall AMC accuracy on the test split:

| Model | Test accuracy |
| --- | ---: |
| SVM statistical baseline | `0.3714` |
| RandomForest statistical baseline | `0.4152` |
| STFT-CNN | `0.3433` |
| CNN1D | `0.5232` |
| ResNet1D-small | `0.5984` |

Observed ranking in round-1:

- `ResNet1D > CNN1D > RandomForest > SVM > STFT-CNN`

Key SNR trends:

- `CNN1D` low-SNR mean accuracy (`<= 0 dB`): `0.1708`
- `CNN1D` high-SNR mean accuracy (`>= 16 dB`): `0.8161`
- `ResNet1D` low-SNR mean accuracy (`<= 0 dB`): `0.1833`
- `ResNet1D` high-SNR mean accuracy (`>= 16 dB`): `0.9502`

Interpretation:

- `ResNet1D` clearly outperformed `CNN1D` on the round-1 budget
- the gain is especially large at higher SNR, where ResNet1D approached near-saturation
- the current lightweight STFT configuration did not outperform time-domain models in this first pass

### 7.3 Spectrum sensing results

Energy detection baseline:

- ROC-AUC: `0.5210`

Deep sensing CNN:

- test accuracy: `0.8491`
- ROC-AUC: `0.9834`
- `Pd @ Pfa = 0.10`: `0.9627`
- `Pd @ Pfa = 0.05`: `0.8557`

Interpretation:

- the CNN detector is dramatically stronger than energy detection on this dataset/task construction
- the detector remains weak near the most difficult SNR buckets, but quickly becomes near-perfect from moderate SNR upward

### 7.4 Multi-task status

The multi-task round-1 run has completed and can now be compared directly
against the single-task AMC and sensing baselines.

Final test metrics:

- modulation accuracy: `0.4563`
- sensing accuracy: `0.8677`
- sensing ROC-AUC: `0.9858`
- `Pd @ Pfa = 0.10`: `0.9732`
- `Pd @ Pfa = 0.05`: `0.8775`

Training signal:

- best observed validation metric occurred at epoch 3: `0.5019`

Interpretation:

- compared with the single-task sensing CNN, the multi-task model slightly
  improved sensing accuracy and ROC-AUC
- compared with single-task `CNN1D` AMC (`0.5232`) and `ResNet1D-small`
  (`0.5984`), the current shared-encoder setup reduced modulation-classification
  accuracy
- the current `lambda_sensing` and sampling balance are therefore better for the
  sensing objective than for preserving AMC performance
## 8. Reproduction Commands

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

Completed round-1 table:

| Model | Task | Overall Metric | Notes |
| --- | --- | --- | --- |
| SVM | AMC | `0.3714` | statistical-feature baseline |
| RandomForest | AMC | `0.4152` | stronger non-deep baseline than SVM |
| CNN1D | AMC | `0.5232` | first deep time-domain baseline |
| ResNet1D-small | AMC | `0.5984` | best AMC result in round-1 |
| STFT-CNN | AMC | `0.3433` | current spectrogram setup underperformed |
| Energy Detection | Sensing | ROC-AUC `0.5210` | non-deep baseline |
| CNN1D detector | Sensing | acc `0.8491`, AUC `0.9834` | binary deep sensing |
| Multi-task | AMC + Sensing | AMC `0.4563`, sensing AUC `0.9858` | shared encoder improved sensing, hurt AMC |

## 12. Discussion Points To Fill After Real Runs

- `accuracy vs SNR` rises clearly at high SNR for all AMC models, but the
  saturation level differs strongly by architecture
- `ResNet1D` consistently outperformed plain `CNN1D`, especially from
  mid-to-high SNR
- the current `STFT-CNN` was not more stable at low SNR and also lagged at high
  SNR, indicating the first STFT parameterization is not yet competitive
- deep sensing improved massively over energy detection, moving from ROC-AUC
  `0.5210` to `0.9834`
- the current multi-task model improved sensing slightly further to ROC-AUC
  `0.9858`, but it did not preserve AMC performance, so the shared-task balance
  still needs tuning

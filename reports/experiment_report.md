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
| STFT-CNN round-1 | `0.3433` |
| CNN1D | `0.5232` |
| ResNet1D-small | `0.5984` |

Observed ranking in round-1:

- `ResNet1D > CNN1D > RandomForest > SVM > STFT-CNN`
- the strongest current STFT result is reported separately in section `7.6`

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

### 7.5 Multi-task round-2 tuning

To recover AMC performance, the second multi-task sweep changed three things:

- shared backbone switched to `resnet1d-small`
- `lambda_sensing` reduced from the earlier stronger joint setting to `0.2`
- checkpoint selection changed from `val_loss` to `val_acc`

Best tuned config:

```text
configs/multitask_round2_resnet_lambda0p2_pos0p75.yaml
```

Final test metrics:

- modulation accuracy: `0.5699`
- sensing accuracy: `0.8585`
- sensing ROC-AUC: `0.9861`
- `Pd @ Pfa = 0.10`: `0.9728`
- `Pd @ Pfa = 0.05`: `0.8808`

SNR trend:

- low-SNR mean AMC accuracy (`<= 0 dB`): `0.1758`
- high-SNR mean AMC accuracy (`>= 16 dB`): `0.9066`

Interpretation:

- this run successfully recovered AMC above single-task `CNN1D` (`0.5232`)
- sensing quality remained essentially unchanged versus the previous multi-task
  run, with ROC-AUC still above `0.986`
- the tuned shared model still does not beat single-task `ResNet1D-small`
  (`0.5984`), so the best current AMC operating point remains the dedicated
  ResNet classifier
- in practice, the round-2 result is a meaningful step toward a usable
  multi-task trade-off rather than a final Pareto-optimal setting

### 7.6 STFT round-3 status

The strongest completed spectrogram experiment so far used a deeper residual 2D
backbone with denser time resolution:

```text
configs/stft_cnn_round3_nfft128_hop16_deeper.yaml
```

Final test metrics:

- overall accuracy: `0.4982`
- low-SNR mean accuracy (`<= 0 dB`): `0.1933`
- high-SNR mean accuracy (`>= 16 dB`): `0.7446`

Comparison against earlier models:

- versus STFT round-1 (`0.3433`), this remains a large absolute improvement
- versus STFT round-2 (`0.4722`), the deeper round-3 model gains another `0.0261`
  overall accuracy
- versus STFT round-2 low-SNR mean (`0.1909`), the round-3 model only improves
  slightly to `0.1933`
- versus STFT round-2 high-SNR mean (`0.6939`), the round-3 model improves more
  materially to `0.7446`
- versus `CNN1D` (`0.5232`), the strengthened STFT model is still lower overall
- versus `CNN1D` low-SNR mean (`0.1708`), the strengthened STFT model remains
  slightly better in the difficult low-SNR regime
- versus `CNN1D` high-SNR mean (`0.8161`), the strengthened STFT model is still
  behind at high SNR, though the gap is smaller than in round-2

Interpretation:

- the spectrogram route is now clearly viable and no longer just a weak side path
- the main round-3 gain comes from recovering high-SNR ceiling rather than from a
  dramatic low-SNR change
- the present bottleneck is therefore no longer only backbone depth; the current
  single-channel `log_power` representation is likely discarding useful phase or
  complex-valued structure
- the next STFT sweep should therefore prioritize richer spectrogram channels such
  as `log_power_phase` or `real_imag`, rather than only larger `n_fft`

### 7.7 STFT round-4 richer-channel status

The next completed spectrogram experiment kept the deeper residual 2D backbone and
the denser hop, but added richer STFT channels through `log_power_phase`:

```text
configs/stft_cnn_round4_nfft128_hop16_deeper_logpower_phase.yaml
```

Best validation point:

- epoch `6`
- validation accuracy: `0.5113`

Final test metrics:

- overall accuracy: `0.5103`
- low-SNR mean accuracy (`<= 0 dB`): `0.1928`
- high-SNR mean accuracy (`>= 16 dB`): `0.7659`
- best SNR bucket: `24 dB`
- best bucket accuracy: `0.7709`

Comparison against earlier STFT models:

- versus STFT round-3 (`0.4982`), round-4 gains another `0.0121` overall accuracy
- versus STFT round-3 low-SNR mean (`0.1933`), round-4 is essentially flat at `0.1928`
- versus STFT round-3 high-SNR mean (`0.7446`), round-4 improves to `0.7659`
- versus STFT round-2 (`0.4722`), round-4 gains `0.0382` overall accuracy
- versus `CNN1D` (`0.5232`), the remaining overall gap is only about `0.0129`

Interpretation:

- the richer `log_power_phase` representation did improve the spectrogram route in
  real test accuracy, so the extra channel information is useful
- the gain again comes mostly from higher-SNR performance rather than from improved
  low-SNR robustness
- spectrogram models are now clearly competitive with `CNN1D`, but still trail both
  `CNN1D` and `ResNet1D-small` overall
- the next most evidence-backed direction is to target low-SNR robustness explicitly,
  rather than only continuing to lift high-SNR ceiling

### 7.8 STFT round-6 low-SNR-weighted fine-tuning

The next completed spectrogram experiment kept the round-4 representation and 2D
backbone, but changed the optimization target rather than the spectrogram channels:

```text
configs/stft_cnn_round6_nfft128_hop16_deeper_logpower_phase_lowsnr_weighted.yaml
```

Two implementation details matter for interpreting this run:

- the run resumed from the stronger round-4 checkpoint
- the trainer resume path was corrected so the fine-tune actually used the new
  config learning rate `0.0005` rather than silently inheriting the old optimizer LR

Best validation point:

- epoch `8`
- validation accuracy: `0.5445`

Final test metrics:

- overall accuracy: `0.5445`
- low-SNR mean accuracy (`<= 0 dB`): `0.1991`
- high-SNR mean accuracy (`>= 16 dB`): `0.8294`
- best SNR bucket: `30 dB`
- best bucket accuracy: `0.8332`

Comparison against earlier STFT models:

- versus STFT round-4 (`0.5103`), round-6 gains `0.0342` overall accuracy
- versus STFT round-4 low-SNR mean (`0.1928`), round-6 improves to `0.1991`
- versus STFT round-4 high-SNR mean (`0.7659`), round-6 improves strongly to `0.8294`
- versus STFT round-3 (`0.4982`), round-6 gains `0.0463` overall accuracy
- versus `CNN1D` (`0.5232`), round-6 is now higher by about `0.0213`
- versus `ResNet1D-small` (`0.5984`), the remaining gap is about `0.0539`

Interpretation:

- the earlier diagnosis was correct: simply enriching spectrogram channels was not
  enough, and low-SNR-targeted optimization was the next high-value step
- the weighted fine-tune improved not only the difficult low-SNR regime, but also
  the high-SNR ceiling; the dominant absolute gain still comes from high SNR, but
  low-SNR accuracy also moved in the right direction this time
- this is the first STFT route in the project that surpasses the single-task
  `CNN1D` baseline on the real test split
- the best current overall AMC result remains single-task `ResNet1D-small`, but
  the gap is now materially smaller than before

### 7.9 STFT round-8 and round-9 stronger-backbone follow-up

After establishing round-6 as the spectrogram baseline, the next sweep tested
whether wider 2D models and more explicit low-SNR rebalancing could close the
remaining gap to `ResNet1D-small`.

Round-8 configuration:

```text
configs/stft_cnn_round8_deeper_wide_logpower_phase_lowsnr_mix.yaml
```

Key changes versus round-6:

- wider channels: `[64, 128, 256]`
- larger classifier head: `512`
- `batch_size = 512`
- low-SNR sample weighting plus moderate oversampling

Round-8 best validation point:

- epoch `2`
- validation accuracy: `0.3987`

Round-8 final test metrics:

- overall accuracy: `0.3989`
- low-SNR mean accuracy (`<= 0 dB`): `0.1796`
- high-SNR mean accuracy (`>= 16 dB`): `0.5670`
- best SNR bucket: `20 dB`
- best bucket accuracy: `0.5724`

Round-9 configuration:

```text
configs/stft_cnn_round9_deeper_wide_logpower_realimag_lowsnr_mix.yaml
```

Key change versus round-8:

- swap the STFT representation from `log_power_phase` to `log_power_real_imag`

Round-9 best validation point:

- epoch `2`
- validation accuracy: `0.4135`

Round-9 final test metrics:

- overall accuracy: `0.4130`
- low-SNR mean accuracy (`<= 0 dB`): `0.1808`
- high-SNR mean accuracy (`>= 16 dB`): `0.5894`
- best SNR bucket: `24 dB`
- best bucket accuracy: `0.5923`

Comparison against the established STFT baseline:

- versus STFT round-6 (`0.5445`), round-8 drops by about `0.1456`
- versus STFT round-6 (`0.5445`), round-9 drops by about `0.1315`
- versus STFT round-6 low-SNR mean (`0.1991`), both round-8 and round-9 are worse
- versus STFT round-6 high-SNR mean (`0.8294`), both round-8 and round-9 are much worse

Interpretation:

- simply widening the 2D spectrogram network was not enough; the wider round-8
  model overfit the validation path early and generalized much worse than round-6
- switching to `log_power_real_imag` recovered a small amount over round-8, but
  still did not approach the stronger round-6 recipe
- the strongest STFT route in the project therefore remains round-6, not the
  later wide-backbone sweeps
- this sweep narrows the search space: the next credible STFT experiments should
  focus on more targeted regularization, normalization, or hybrid time-frequency
  fusion rather than simply scaling up the 2D spectrogram backbone

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

Round-2 strengthened STFT reproduction command:

```bash
python scripts/train.py \
  --config configs/stft_cnn_round2_nfft128_hop16_deep.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/stft_cnn_round2_nfft128_hop16_deep
```

```bash
python scripts/evaluate.py \
  --config configs/stft_cnn_round2_nfft128_hop16_deep.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/stft_cnn_round2_nfft128_hop16_deep/best.pt \
  --out-dir outputs/runs/stft_cnn_round2_nfft128_hop16_deep_eval
```

Round-3 stronger STFT reproduction command:

```bash
python scripts/train.py \
  --config configs/stft_cnn_round3_nfft128_hop16_deeper.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/stft_cnn_round3_nfft128_hop16_deeper
```

```bash
python scripts/evaluate.py \
  --config configs/stft_cnn_round3_nfft128_hop16_deeper.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/stft_cnn_round3_nfft128_hop16_deeper/best.pt \
  --out-dir outputs/runs/stft_cnn_round3_nfft128_hop16_deeper_eval
```

Round-4 richer-channel STFT reproduction command:

```bash
python scripts/train.py \
  --config configs/stft_cnn_round4_nfft128_hop16_deeper_logpower_phase.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/stft_cnn_round4_nfft128_hop16_deeper_logpower_phase
```

```bash
python scripts/evaluate.py \
  --config configs/stft_cnn_round4_nfft128_hop16_deeper_logpower_phase.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/stft_cnn_round4_nfft128_hop16_deeper_logpower_phase/best.pt \
  --out-dir outputs/runs/stft_cnn_round4_nfft128_hop16_deeper_logpower_phase_eval
```

Round-6 low-SNR-weighted STFT reproduction command:

```bash
python scripts/train.py \
  --config configs/stft_cnn_round6_nfft128_hop16_deeper_logpower_phase_lowsnr_weighted.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/stft_cnn_round6_nfft128_hop16_deeper_logpower_phase_lowsnr_weighted \
  --resume outputs/runs/stft_cnn_round4_nfft128_hop16_deeper_logpower_phase/best.pt
```

```bash
python scripts/evaluate.py \
  --config configs/stft_cnn_round6_nfft128_hop16_deeper_logpower_phase_lowsnr_weighted.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/stft_cnn_round6_nfft128_hop16_deeper_logpower_phase_lowsnr_weighted/best.pt \
  --out-dir outputs/runs/stft_cnn_round6_nfft128_hop16_deeper_logpower_phase_lowsnr_weighted_eval
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

Round-2 tuned reproduction command:

```bash
python scripts/train.py \
  --config configs/multitask_round2_resnet_lambda0p2_pos0p75.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/multitask_round2_resnet_lambda0p2_pos0p75
```

```bash
python scripts/evaluate.py \
  --config configs/multitask_round2_resnet_lambda0p2_pos0p75.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/multitask_round2_resnet_lambda0p2_pos0p75/best.pt \
  --out-dir outputs/runs/multitask_round2_resnet_lambda0p2_pos0p75_eval
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
| STFT-CNN round-1 | AMC | `0.3433` | initial spectrogram setup underperformed |
| STFT-CNN round-2 | AMC | `0.4722` | deep 2D backbone greatly improved spectrogram route |
| STFT-CNN round-3 | AMC | `0.4982` | deeper backbone further improved high-SNR ceiling |
| STFT-CNN round-4 | AMC | `0.5103` | richer `log_power_phase` channels improved high-SNR ceiling again |
| STFT-CNN round-6 | AMC | `0.5445` | low-SNR-weighted fine-tune is the first STFT run to beat CNN1D |
| STFT-CNN round-8 | AMC | `0.3989` | wider `log_power_phase` model with low-SNR oversampling regressed badly |
| STFT-CNN round-9 | AMC | `0.4130` | `log_power_real_imag` recovered slightly over round-8, still far below round-6 |
| Energy Detection | Sensing | ROC-AUC `0.5210` | non-deep baseline |
| CNN1D detector | Sensing | acc `0.8491`, AUC `0.9834` | binary deep sensing |
| Multi-task round-1 | AMC + Sensing | AMC `0.4563`, sensing AUC `0.9858` | shared encoder improved sensing, hurt AMC |
| Multi-task round-2 | AMC + Sensing | AMC `0.5699`, sensing AUC `0.9861` | tuned loss balance restores AMC above CNN1D |

## 12. Discussion Points To Fill After Real Runs

- `accuracy vs SNR` rises clearly at high SNR for all AMC models, but the
  saturation level differs strongly by architecture
- `ResNet1D` consistently outperformed plain `CNN1D`, especially from
  mid-to-high SNR
- the best `STFT-CNN` route remains round-6 at `0.5445`; later wide-backbone
  sweeps (`0.3989`, `0.4130`) did not improve on that baseline
- deep sensing improved massively over energy detection, moving from ROC-AUC
  `0.5210` to `0.9834`
- the round-2 multi-task tuning result confirms that checkpoint criterion and
  task-loss balance matter materially; the tuned shared model restored AMC above
  `CNN1D` while preserving sensing ROC-AUC around `0.986`
- the current best multi-task setup is therefore viable for joint deployment,
  though it still falls short of single-task `ResNet1D-small` AMC accuracy
- the strengthened STFT experiments confirm that spectrogram models were not a
  dead end; round-6 makes them a competitive mainline alternative by beating CNN1D
- however, the round-8 and round-9 follow-up sweep also shows that naive scaling
  of the 2D backbone is not a free win and can degrade both low-SNR and high-SNR
  generalization substantially
- the remaining limitation is still the residual gap to the dedicated ResNet1D
  baseline, and the next useful STFT work should target representation fusion or
  regularization rather than only larger spectrogram backbones

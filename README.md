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

Current code paths are fully smoke-tested on tiny synthetic datasets.
Real RadioML 2018.01A round-1 experiments have been run on
`data/GOLD_XYZ_OSC.0001_1024.hdf5`.

## Real Results

Real split:

- dataset size: `2,555,904`
- classes: `24`
- split artifact: `outputs/splits/radioml2018_seed42.npz`

Verified headline results from the strongest completed run of each route:

| Task | Model | Result |
| --- | --- | ---: |
| AMC | SVM statistical baseline | `0.3714` |
| AMC | RandomForest statistical baseline | `0.4152` |
| AMC | STFT-CNN round-4 | `0.5103` |
| AMC | CNN1D | `0.5232` |
| AMC | ResNet1D-small | `0.5984` |
| AMC + Sensing | Multi-task shared encoder round-2 | modulation `0.5699`, sensing AUC `0.9861` |
| Sensing | Energy Detection ROC-AUC | `0.5210` |
| Sensing | CNN detector accuracy | `0.8491` |
| Sensing | CNN detector ROC-AUC | `0.9834` |

Current experiment takeaway:

- `ResNet1D` is the strongest AMC model in the current single-GPU budget
- the current best completed `STFT-CNN` route (`round-4`) is much stronger than the initial spectrogram baseline, but still does not beat time-domain `CNN1D` or `ResNet1D`
- deep spectrum sensing is dramatically stronger than the energy-detection baseline
- the tuned multi-task setting restores AMC above single-task `CNN1D` while keeping sensing ROC-AUC above `0.986`, but it still trails the dedicated single-task `ResNet1D-small`

Recent tuning highlights:

- tuned multi-task config `configs/multitask_round2_resnet_lambda0p2_pos0p75.yaml`
  reached modulation accuracy `0.5699`
- this restores multi-task AMC above `CNN1D` (`0.5232`) while keeping strong sensing
  ROC-AUC `0.9861`
- the tuned run is still below single-task `ResNet1D-small` (`0.5984`), so the current
  shared-encoder setting is a partial recovery rather than a full Pareto improvement
- richer-channel STFT round-4 with `log_power_phase`, `n_fft=128`, `hop=16`, and a
  deeper residual 2D backbone reached test accuracy `0.5103`
- this is a large improvement over round-1 STFT (`0.3433`), a gain over
  STFT round-2 (`0.4722`), and another improvement over STFT round-3 (`0.4982`),
  but it still does not beat `CNN1D` (`0.5232`)

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

Accepted round-1 statistical baseline artifacts:

- [outputs/baselines/svm_random_summary.json](/home/developer716/workspace/rfml-amc-spectrum-sensing/outputs/baselines/svm_random_summary.json)
- [outputs/baselines/rf_summary.json](/home/developer716/workspace/rfml-amc-spectrum-sensing/outputs/baselines/rf_summary.json)

Important note:

- the older `outputs/baselines/svm_summary.json` file came from an earlier incorrect subset-selection path and only covered two classes
- do not use that file in final comparisons

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

Real round-1 sensing baseline metric:

- `Energy Detection ROC-AUC = 0.5209778859`

## Deep AMC Training

For real first-pass experiments, shorter configs were used:

- `configs/cnn1d_round1.yaml`
- `configs/resnet1d_round1.yaml`
- `configs/stft_cnn_round1.yaml`
- `configs/sensing_cnn_round1.yaml`
- `configs/multitask_round1.yaml`

### CNN1D

```bash
python scripts/train.py \
  --config configs/cnn1d_round1.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/cnn1d_round1_seed42
```

Evaluate:

```bash
python scripts/evaluate.py \
  --config configs/cnn1d_round1.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/cnn1d_round1_seed42/best.pt \
  --out-dir outputs/runs/cnn1d_round1_seed42_eval
```

### ResNet1D

```bash
python scripts/train.py \
  --config configs/resnet1d_round1.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/resnet1d_round1_seed42
```

```bash
python scripts/evaluate.py \
  --config configs/resnet1d_round1.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/resnet1d_round1_seed42/best.pt \
  --out-dir outputs/runs/resnet1d_round1_seed42_eval
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
  --config configs/stft_cnn_round1.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/stft_cnn_round1_seed42
```

Evaluate:

```bash
python scripts/evaluate.py \
  --config configs/stft_cnn_round1.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/stft_cnn_round1_seed42/best.pt \
  --out-dir outputs/runs/stft_cnn_round1_seed42_eval
```

Round-1 AMC artifacts:

- [outputs/runs/cnn1d_round1_seed42_eval/summary.json](/home/developer716/workspace/rfml-amc-spectrum-sensing/outputs/runs/cnn1d_round1_seed42_eval/summary.json)
- [outputs/runs/resnet1d_round1_seed42_eval/summary.json](/home/developer716/workspace/rfml-amc-spectrum-sensing/outputs/runs/resnet1d_round1_seed42_eval/summary.json)
- [outputs/runs/stft_cnn_round1_seed42_eval/summary.json](/home/developer716/workspace/rfml-amc-spectrum-sensing/outputs/runs/stft_cnn_round1_seed42_eval/summary.json)
- [outputs/comparisons/acc_vs_snr_compare.png](/home/developer716/workspace/rfml-amc-spectrum-sensing/outputs/comparisons/acc_vs_snr_compare.png)

Previous round-3 STFT result:

- config: `configs/stft_cnn_round3_nfft128_hop16_deeper.yaml`
- test accuracy: `0.4982`
- low-SNR mean accuracy (`<= 0 dB`): `0.1933`
- high-SNR mean accuracy (`>= 16 dB`): `0.7446`

Current best round-4 STFT result:

- config: `configs/stft_cnn_round4_nfft128_hop16_deeper_logpower_phase.yaml`
- test accuracy: `0.5103`
- low-SNR mean accuracy (`<= 0 dB`): `0.1928`
- high-SNR mean accuracy (`>= 16 dB`): `0.7659`

Interpretation:

- adding richer spectrogram channels on top of the deeper residual 2D backbone
  improved the spectrogram route again
- compared with STFT round-3, the round-4 gain came mostly from a higher high-SNR
  ceiling: low-SNR mean changed slightly (`0.1933 -> 0.1928`), while high-SNR mean
  improved more clearly (`0.7446 -> 0.7659`)
- relative to `CNN1D`, the current STFT model is still lower overall, but the gap is
  now down to about `0.013`
- the next STFT sweep should focus on low-SNR robustness, because richer channels
  helped the high-SNR ceiling more than the difficult low-SNR regime

## Deep Spectrum Sensing

Train the binary CNN detector:

```bash
python scripts/train.py \
  --config configs/sensing_cnn_round1.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/sensing_cnn_round1_seed42
```

Evaluate:

```bash
python scripts/evaluate.py \
  --config configs/sensing_cnn_round1.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/sensing_cnn_round1_seed42/best.pt
```

Or use the sensing wrapper:

```bash
python scripts/run_sensing.py \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --method cnn \
  --split outputs/splits/radioml2018_seed42.npz \
  --config configs/sensing_cnn_round1.yaml \
  --ckpt outputs/runs/sensing_cnn_round1_seed42/best.pt \
  --eval-out-dir outputs/runs/sensing_cnn_round1_seed42_eval
```

Round-1 sensing artifacts:

- [outputs/metrics/energy_detection.csv](/home/developer716/workspace/rfml-amc-spectrum-sensing/outputs/metrics/energy_detection.csv)
- [outputs/runs/sensing_cnn_round1_seed42_eval/summary.json](/home/developer716/workspace/rfml-amc-spectrum-sensing/outputs/runs/sensing_cnn_round1_seed42_eval/summary.json)
- [outputs/runs/sensing_cnn_round1_seed42_eval/sensing_metrics.csv](/home/developer716/workspace/rfml-amc-spectrum-sensing/outputs/runs/sensing_cnn_round1_seed42_eval/sensing_metrics.csv)
- [outputs/runs/sensing_cnn_round1_seed42_eval/pd_vs_snr.png](/home/developer716/workspace/rfml-amc-spectrum-sensing/outputs/runs/sensing_cnn_round1_seed42_eval/pd_vs_snr.png)

## Multi-Task AMC Plus Sensing

Train:

```bash
python scripts/train.py \
  --config configs/multitask_round1.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --out outputs/runs/multitask_round1_seed42
```

Evaluate:

```bash
python scripts/evaluate.py \
  --config configs/multitask_round1.yaml \
  --h5 data/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --split outputs/splits/radioml2018_seed42.npz \
  --ckpt outputs/runs/multitask_round1_seed42/best.pt \
  --out-dir outputs/runs/multitask_round1_seed42_eval
```

Round-1 multi-task result:

- modulation accuracy: `0.4563`
- sensing accuracy: `0.8677`
- sensing ROC-AUC: `0.9858`
- `Pd @ Pfa = 0.10`: `0.9732`
- `Pd @ Pfa = 0.05`: `0.8775`

Interpretation:

- the shared encoder produced slightly better sensing metrics than the single-task sensing CNN
- the same run underperformed single-task `CNN1D` and `ResNet1D` on AMC, so the current joint-loss setting is not yet a Pareto improvement

Round-2 tuned multi-task result:

- config: `configs/multitask_round2_resnet_lambda0p2_pos0p75.yaml`
- modulation accuracy: `0.5699`
- sensing accuracy: `0.8585`
- sensing ROC-AUC: `0.9861`
- `Pd @ Pfa = 0.10`: `0.9728`
- `Pd @ Pfa = 0.05`: `0.8808`
- low-SNR mean AMC accuracy (`<= 0 dB`): `0.1758`
- high-SNR mean AMC accuracy (`>= 16 dB`): `0.9066`

Interpretation:

- selecting the best checkpoint by `val_acc` and reducing `lambda_sensing` to `0.2`
  recovered AMC performance substantially
- the tuned multi-task run now beats single-task `CNN1D` on AMC while preserving
  near-identical sensing AUC
- the current best single-task AMC model is still `ResNet1D-small`

## Comparison

Build compact comparison tables and an `accuracy vs SNR` overlay:

```bash
python scripts/compare_results.py \
  --baseline-acc-vs-snr outputs/baselines/svm_random_accuracy_vs_snr.csv \
  --baseline-overall-acc 0.3714166666666667 \
  --cnn-run-dir outputs/runs/cnn1d_round1_seed42_eval \
  --resnet-run-dir outputs/runs/resnet1d_round1_seed42_eval \
  --stft-run-dir outputs/runs/stft_cnn_round3_nfft128_hop16_deeper_eval \
  --multitask-run-dir outputs/runs/multitask_round2_resnet_lambda0p2_pos0p75_eval \
  --out-dir outputs/comparisons
```

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

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from rfml.monitor import DashboardFilters, collect_gpu_stats, load_run_snapshot, render_dashboard_html
from rfml.training.losses import compute_weighted_cross_entropy
from rfml.data.splits import create_stratified_splits_from_h5, save_split_bundle
from rfml.training.metrics import compute_accuracy_vs_snr
from rfml.training.trainer import RFMLTrainer, TrainerConfig


def test_trainer_runs_and_saves_checkpoints(tmp_path: Path) -> None:
    h5_path = _build_training_h5(tmp_path / "train_radioml.h5")
    split_bundle = create_stratified_splits_from_h5(h5_path, seed=42)
    split_path = save_split_bundle(split_bundle, tmp_path / "splits.npz")
    out_dir = tmp_path / "run"

    config = TrainerConfig(
        task="amc",
        model_name="cnn1d",
        num_classes=4,
        epochs=5,
        batch_size=16,
        lr=1e-3,
        optimizer="adamw",
        weight_decay=1e-4,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_clip=1.0,
        early_stopping_patience=8,
        device="cpu",
        dropout=0.1,
        classifier_hidden_dim=64,
        channels=(16, 32, 64),
        kernel_sizes=(7, 5, 3),
        save_every=5,
        scan_chunk_size=256,
    )
    trainer = RFMLTrainer(
        config,
        h5_path=h5_path,
        split_path=split_path,
        out_dir=out_dir,
    )
    result = trainer.fit()

    history = result["history"]
    assert len(history) == 5
    assert history[-1]["train_loss"] <= history[0]["train_loss"]
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "last.pt").exists()
    assert (out_dir / "train_log.csv").exists()
    assert (out_dir / "history.json").exists()
    assert (out_dir / "live_status.json").exists()
    parsed = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    assert len(parsed) == 5
    live = json.loads((out_dir / "live_status.json").read_text(encoding="utf-8"))
    assert live["status"] == "completed"
    assert live["phase"] in {"done", "stopped_early"}


def test_accuracy_vs_snr_trend_on_synthetic_predictions() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1], dtype=np.int64)
    snrs = np.array([-20.0, -20.0, -20.0, -20.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    df = compute_accuracy_vs_snr(y_true, y_pred, snrs)
    low = float(df.loc[df["snr"] == -20.0, "accuracy"].iloc[0])
    high = float(df.loc[df["snr"] == 10.0, "accuracy"].iloc[0])
    assert high >= low


def test_weighted_cross_entropy_emphasizes_weighted_samples() -> None:
    logits = torch.tensor([[2.5, 0.1], [0.2, 2.2]], dtype=torch.float32)
    targets = torch.tensor([1, 0], dtype=torch.int64)
    unweighted = compute_weighted_cross_entropy(logits, targets)
    weighted = compute_weighted_cross_entropy(logits, targets, sample_weights=torch.tensor([3.0, 1.0]))
    assert weighted > unweighted


def test_resume_rejects_when_target_epochs_not_greater_than_checkpoint_epoch(tmp_path: Path) -> None:
    h5_path = _build_training_h5(tmp_path / "resume_radioml.h5")
    split_bundle = create_stratified_splits_from_h5(h5_path, seed=42)
    split_path = save_split_bundle(split_bundle, tmp_path / "splits_resume.npz")
    out_dir = tmp_path / "resume_run"

    config = TrainerConfig(
        task="amc",
        model_name="cnn1d",
        num_classes=4,
        epochs=3,
        batch_size=16,
        lr=1e-3,
        optimizer="adamw",
        weight_decay=1e-4,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_clip=1.0,
        early_stopping_patience=8,
        device="cpu",
        dropout=0.1,
        classifier_hidden_dim=64,
        channels=(16, 32, 64),
        kernel_sizes=(7, 5, 3),
        save_every=5,
        scan_chunk_size=256,
    )
    trainer = RFMLTrainer(config, h5_path=h5_path, split_path=split_path, out_dir=out_dir)
    trainer.fit()
    with pytest.raises(ValueError, match="checkpoint epoch is already >= configured total epochs"):
        RFMLTrainer(
            config,
            h5_path=h5_path,
            split_path=split_path,
            out_dir=tmp_path / "resume_reject",
            resume_ckpt=out_dir / "last.pt",
        )


def test_resume_uses_current_config_optimizer_hparams(tmp_path: Path) -> None:
    h5_path = _build_training_h5(tmp_path / "resume_opt_radioml.h5")
    split_bundle = create_stratified_splits_from_h5(h5_path, seed=42)
    split_path = save_split_bundle(split_bundle, tmp_path / "splits_resume_opt.npz")
    base_out_dir = tmp_path / "resume_opt_base"

    base_config = TrainerConfig(
        task="amc",
        model_name="cnn1d",
        num_classes=4,
        epochs=2,
        batch_size=16,
        lr=1e-3,
        optimizer="adamw",
        weight_decay=1e-4,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_clip=1.0,
        early_stopping_patience=8,
        device="cpu",
        dropout=0.1,
        classifier_hidden_dim=64,
        channels=(16, 32, 64),
        kernel_sizes=(7, 5, 3),
        save_every=5,
        scan_chunk_size=256,
    )
    RFMLTrainer(base_config, h5_path=h5_path, split_path=split_path, out_dir=base_out_dir).fit()

    resumed_config = TrainerConfig(
        task="amc",
        model_name="cnn1d",
        num_classes=4,
        epochs=4,
        batch_size=16,
        lr=5e-4,
        optimizer="adamw",
        weight_decay=5e-5,
        amp=False,
        num_workers=0,
        pin_memory=False,
        grad_clip=1.0,
        early_stopping_patience=8,
        device="cpu",
        dropout=0.1,
        classifier_hidden_dim=64,
        channels=(16, 32, 64),
        kernel_sizes=(7, 5, 3),
        save_every=5,
        scan_chunk_size=256,
    )
    resumed = RFMLTrainer(
        resumed_config,
        h5_path=h5_path,
        split_path=split_path,
        out_dir=tmp_path / "resume_opt_new",
        resume_ckpt=base_out_dir / "best.pt",
    )
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert resumed.optimizer.param_groups[0]["weight_decay"] == pytest.approx(5e-5)


def test_monitor_snapshot_and_dashboard_render(tmp_path: Path) -> None:
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir(parents=True)
    eval_dir = tmp_path / "demo_run_eval"
    eval_dir.mkdir(parents=True)
    (run_dir / "train_log.csv").write_text(
        "epoch,train_loss,train_acc,val_loss,val_acc,lr\n"
        "1,2.0,0.3,1.8,0.35,0.001\n"
        "2,1.7,0.4,1.5,0.45,0.001\n",
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(
        json.dumps(
            [
                {"epoch": 1, "train_loss": 2.0, "train_acc": 0.3, "val_loss": 1.8, "val_acc": 0.35, "lr": 0.001},
                {"epoch": 2, "train_loss": 1.7, "train_acc": 0.4, "val_loss": 1.5, "val_acc": 0.45, "lr": 0.001},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "live_status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "phase": "train",
                "epoch": 3,
                "num_epochs": 8,
                "updated_at": "2026-05-08T23:59:59+0800",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (eval_dir / "summary.json").write_text(json.dumps({"task": "amc", "overall_accuracy": 0.52}, indent=2), encoding="utf-8")
    (eval_dir / "accuracy_vs_snr.csv").write_text(
        "snr,accuracy\n-20,0.1\n0,0.3\n20,0.8\n",
        encoding="utf-8",
    )
    (run_dir / "best.pt").write_bytes(b"pt")
    stft_round1 = tmp_path / "stft_cnn_round1_seed42"
    stft_round1.mkdir(parents=True)
    stft_round1_eval = tmp_path / "stft_cnn_round1_seed42_eval"
    stft_round1_eval.mkdir(parents=True)
    (stft_round1 / "train_log.csv").write_text(
        "epoch,train_loss,train_acc,val_loss,val_acc,lr\n"
        "1,2.1,0.2,1.9,0.30,0.001\n",
        encoding="utf-8",
    )
    (stft_round1 / "live_status.json").write_text(
        json.dumps({"status": "completed", "phase": "eval", "epoch": 1, "updated_at": "2026-05-08T20:00:00+0800"}, indent=2),
        encoding="utf-8",
    )
    (stft_round1_eval / "summary.json").write_text(
        json.dumps({"task": "amc", "overall_accuracy": 0.31}, indent=2),
        encoding="utf-8",
    )
    stft_round2 = tmp_path / "stft_cnn_round2_nfft128_hop16_deep"
    stft_round2.mkdir(parents=True)
    stft_round2_eval = tmp_path / "stft_cnn_round2_nfft128_hop16_deep_eval"
    stft_round2_eval.mkdir(parents=True)
    (stft_round2 / "train_log.csv").write_text(
        "epoch,train_loss,train_acc,val_loss,val_acc,lr\n"
        "1,1.9,0.3,1.6,0.40,0.001\n",
        encoding="utf-8",
    )
    (stft_round2 / "live_status.json").write_text(
        json.dumps({"status": "completed", "phase": "eval", "epoch": 2, "updated_at": "2026-05-08T21:00:00+0800"}, indent=2),
        encoding="utf-8",
    )
    (stft_round2_eval / "summary.json").write_text(
        json.dumps({"task": "amc", "overall_accuracy": 0.47}, indent=2),
        encoding="utf-8",
    )

    snapshot = load_run_snapshot(run_dir)
    assert snapshot.eval_dir == eval_dir
    assert len(snapshot.train_log) == 2
    assert snapshot.live_status is not None
    assert snapshot.summary is not None
    stft_snapshot_1 = load_run_snapshot(stft_round1)
    stft_snapshot_2 = load_run_snapshot(stft_round2)
    html = render_dashboard_html(
        [snapshot, stft_snapshot_1, stft_snapshot_2],
        root=tmp_path,
        gpu_stats=collect_gpu_stats(),
        refreshed_at="2026-05-08T23:59:59+0800",
        refresh_seconds=5.0,
    )
    assert "RFML Training Monitor" in html
    assert "Experiment Overview" in html
    assert "Task Leaderboard" in html
    assert "Sweep Families" in html
    assert "Filters" in html
    assert "Family Trends" in html
    assert "Recent Runs" in html
    assert "demo_run" in html
    assert "demo_run_eval" in html
    assert "overall_accuracy" in html
    assert "stft_cnn" in html
    assert "round1" in html
    assert "round2" in html

    filtered_html = render_dashboard_html(
        [snapshot, stft_snapshot_1, stft_snapshot_2],
        root=tmp_path,
        gpu_stats=[],
        refreshed_at="2026-05-08T23:59:59+0800",
        refresh_seconds=5.0,
        filters=DashboardFilters(task="amc", status="completed", family="stft_cnn"),
    )
    assert "stft_cnn_round1_seed42" in filtered_html
    assert "stft_cnn_round2_nfft128_hop16_deep" in filtered_html
    assert "demo_run_eval" not in filtered_html
    assert "2 / 3 runs shown" in filtered_html


def _build_training_h5(path: Path) -> Path:
    class_defs = [
        ("BPSK", 1.0),
        ("QPSK", 2.0),
        ("QAM16", 3.0),
        ("QAM64", 4.0),
    ]
    snrs = [-20.0, -10.0, 0.0, 10.0]
    repeats = 6
    seq_len = 1024
    num_samples = len(class_defs) * len(snrs) * repeats

    x = np.zeros((num_samples, seq_len, 2), dtype=np.float32)
    y = np.zeros((num_samples, len(class_defs)), dtype=np.float32)
    z = np.zeros((num_samples, 1), dtype=np.float32)

    idx = 0
    for label, (_, freq_scale) in enumerate(class_defs):
        for snr in snrs:
            amp = 0.3 + (snr + 20.0) / 20.0
            noise_std = max(0.05, 0.35 - (snr + 20.0) / 120.0)
            for repeat in range(repeats):
                t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
                phase = repeat * 0.3
                i = amp * np.sin(2.0 * np.pi * freq_scale * t + phase)
                q = amp * np.cos(2.0 * np.pi * freq_scale * t + phase)
                rng = np.random.default_rng(1000 + idx)
                x[idx, :, 0] = i + rng.normal(0.0, noise_std, size=seq_len)
                x[idx, :, 1] = q + rng.normal(0.0, noise_std, size=seq_len)
                y[idx, label] = 1.0
                z[idx, 0] = snr
                idx += 1

    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("X", data=x)
        h5f.create_dataset("Y", data=y)
        h5f.create_dataset("Z", data=z)
        h5f.create_dataset(
            "classes",
            data=np.array([name.encode("utf-8") for name, _ in class_defs]),
        )
    return path

"""Generic trainer for RFML models."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rfml.data.radioml2018 import RadioML2018Dataset
from rfml.data.spectrum_sensing import SpectrumSensingDataset
from rfml.data.splits import SplitBundle, load_split_bundle, resolve_split_indices
from rfml.data.transforms import STFTTransform
from rfml.models.cnn1d import CNN1D
from rfml.models.resnet1d import build_resnet1d
from rfml.models.stft_cnn import STFTCNN
from rfml.training.losses import build_classification_loss
from rfml.training.metrics import compute_accuracy


@dataclass(frozen=True)
class TrainerConfig:
    task: str
    model_name: str
    num_classes: int
    epochs: int
    batch_size: int
    lr: float
    optimizer: str
    weight_decay: float
    amp: bool
    num_workers: int
    pin_memory: bool
    grad_clip: float | None
    early_stopping_patience: int
    device: str
    dropout: float
    classifier_hidden_dim: int
    channels: tuple[int, int, int]
    kernel_sizes: tuple[int, int, int]
    save_every: int
    scan_chunk_size: int
    stft_n_fft: int | None = None
    stft_hop_length: int | None = None
    stft_window: str | None = None
    stft_output: str | None = None
    stft_backend: str | None = None
    sensing_positive_ratio: float | None = None
    sensing_noise_power: float | None = None
    sensing_seed: int = 42


@dataclass(frozen=True)
class EvaluationOutputs:
    loss: float
    accuracy: float
    labels: np.ndarray
    preds: np.ndarray
    snrs: np.ndarray


class RFMLTrainer:
    def __init__(
        self,
        config: TrainerConfig,
        *,
        h5_path: str | Path,
        split_path: str | Path,
        out_dir: str | Path,
        resume_ckpt: str | Path | None = None,
    ) -> None:
        self.config = config
        self.h5_path = Path(h5_path).expanduser().resolve()
        self.split_path = Path(split_path).expanduser().resolve()
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.out_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = self.out_dir / "tensorboard"
        self.csv_log_path = self.out_dir / "train_log.csv"
        self.history_json_path = self.out_dir / "history.json"
        self.last_ckpt_path = self.out_dir / "last.pt"
        self.best_ckpt_path = self.out_dir / "best.pt"
        self.resume_ckpt = Path(resume_ckpt).expanduser().resolve() if resume_ckpt is not None else None

        self.device = torch.device(config.device)
        self.split_bundle = load_split_bundle(self.split_path)
        self.model = self._build_model().to(self.device)
        self.criterion = build_classification_loss("cross_entropy")
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.history: list[dict[str, float | int]] = []

        self.train_loader = self._build_loader("train", shuffle=True)
        self.val_loader = self._build_loader("val", shuffle=False)

        if self.resume_ckpt is not None and self.resume_ckpt.exists():
            self._load_checkpoint(self.resume_ckpt)

    def fit(self) -> dict[str, Any]:
        patience_counter = 0
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss, train_acc = self._run_epoch(epoch, training=True)
            val_outputs = self.evaluate_loader(self.val_loader)
            val_loss = val_outputs.loss
            val_acc = val_outputs.accuracy

            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            self.history.append(row)
            self._append_csv_row(row)
            self._log_tensorboard(epoch, row)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            self._save_checkpoint(epoch, is_best=is_best)

            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch, is_best=False, named=True)

            if patience_counter >= self.config.early_stopping_patience:
                break

        self.writer.flush()
        self.writer.close()
        self.history_json_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        return {
            "best_ckpt": str(self.best_ckpt_path),
            "last_ckpt": str(self.last_ckpt_path),
            "history": self.history,
        }

    def evaluate_loader(self, loader: DataLoader) -> EvaluationOutputs:
        self.model.eval()
        losses: list[float] = []
        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        snrs_list: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                x = batch["iq"].to(self.device, non_blocking=self.config.pin_memory)
                y = batch["label"].to(self.device, non_blocking=self.config.pin_memory)
                snr = batch["snr"]
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.config.amp and self.device.type == "cuda",
                ):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                losses.append(float(loss.item()))
                logits_list.append(logits.detach().cpu())
                labels_list.append(y.detach().cpu())
                snrs_list.append(snr.detach().cpu())

        all_logits = torch.cat(logits_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0).numpy()
        all_snrs = torch.cat(snrs_list, dim=0).numpy()
        all_preds = torch.argmax(all_logits, dim=1).numpy()

        return EvaluationOutputs(
            loss=float(np.mean(losses)) if losses else float("nan"),
            accuracy=compute_accuracy(all_labels, all_preds),
            labels=all_labels,
            preds=all_preds,
            snrs=all_snrs,
        )

    def _run_epoch(self, epoch: int, *, training: bool) -> tuple[float, float]:
        loader = self.train_loader if training else self.val_loader
        self.model.train(training)
        losses: list[float] = []
        preds_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{self.config.epochs}", leave=False)
        for batch in progress:
            x = batch["iq"].to(self.device, non_blocking=self.config.pin_memory)
            y = batch["label"].to(self.device, non_blocking=self.config.pin_memory)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.config.amp and self.device.type == "cuda",
            ):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            if training:
                self.scaler.scale(loss).backward()
                if self.config.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            losses.append(float(loss.item()))
            preds_list.append(torch.argmax(logits.detach(), dim=1).cpu())
            labels_list.append(y.detach().cpu())
            progress.set_postfix(loss=f"{np.mean(losses):.4f}")

        y_pred = torch.cat(preds_list, dim=0).numpy()
        y_true = torch.cat(labels_list, dim=0).numpy()
        return float(np.mean(losses)), compute_accuracy(y_true, y_pred)

    def _build_model(self) -> nn.Module:
        if self.config.model_name == "cnn1d":
            return CNN1D(
                num_classes=self.config.num_classes,
                dropout=self.config.dropout,
                classifier_hidden_dim=self.config.classifier_hidden_dim,
                channels=self.config.channels,
                kernel_sizes=self.config.kernel_sizes,
            )
        if self.config.model_name in {"resnet1d-small", "resnet1d-medium"}:
            return build_resnet1d(
                self.config.model_name,
                num_classes=self.config.num_classes,
                dropout=self.config.dropout,
                classifier_hidden_dim=self.config.classifier_hidden_dim,
            )
        if self.config.model_name == "stft_cnn":
            return STFTCNN(
                num_classes=self.config.num_classes,
                channels=self.config.channels,
                dropout=self.config.dropout,
                classifier_hidden_dim=self.config.classifier_hidden_dim,
            )
        raise ValueError(f"Unsupported model_name: {self.config.model_name}")

    def _build_optimizer(self) -> Optimizer:
        optimizer_name = self.config.optimizer.lower()
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _build_loader(self, split_name: str, *, shuffle: bool) -> DataLoader:
        transform = None
        if self.config.model_name == "stft_cnn":
            transform = STFTTransform(
                n_fft=int(self.config.stft_n_fft or 128),
                hop_length=int(self.config.stft_hop_length or 32),
                window=str(self.config.stft_window or "hann"),
                output=str(self.config.stft_output or "log_power"),
                backend=str(self.config.stft_backend or "torch"),
            )
        split_indices = resolve_split_indices(self.split_bundle, split_name)
        if self.config.task == "spectrum_sensing":
            dataset = SpectrumSensingDataset(
                self.h5_path,
                split_indices=split_indices,
                class_names=self.split_bundle.class_names,
                scan_chunk_size=self.config.scan_chunk_size,
                transform=transform,
                positive_ratio=float(self.config.sensing_positive_ratio or 0.5),
                noise_power=self.config.sensing_noise_power,
                seed=self.config.sensing_seed,
            )
        else:
            dataset = RadioML2018Dataset(
                self.h5_path,
                split_indices=split_indices,
                class_names=self.split_bundle.class_names,
                scan_chunk_size=self.config.scan_chunk_size,
                transform=transform,
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )

    def _append_csv_row(self, row: dict[str, float | int]) -> None:
        exists = self.csv_log_path.exists()
        with self.csv_log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _log_tensorboard(self, epoch: int, row: dict[str, float | int]) -> None:
        for key, value in row.items():
            if key == "epoch":
                continue
            self.writer.add_scalar(key, float(value), epoch + 1)

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
            "split_path": str(self.split_path),
            "h5_path": str(self.h5_path),
        }

    def _save_checkpoint(self, epoch: int, *, is_best: bool, named: bool = False) -> None:
        payload = self._checkpoint_payload(epoch)
        torch.save(payload, self.last_ckpt_path)
        if is_best:
            torch.save(payload, self.best_ckpt_path)
        if named:
            torch.save(payload, self.checkpoints_dir / f"epoch_{epoch + 1:03d}.pt")

    def _load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scaler.load_state_dict(payload["scaler_state_dict"])
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        self.start_epoch = int(payload.get("epoch", 0))

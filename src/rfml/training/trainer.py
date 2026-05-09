"""Generic trainer for RFML models."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rfml.data.radioml2018 import RadioML2018Dataset
from rfml.data.multitask import MultiTaskRadioMLDataset
from rfml.data.spectrum_sensing import SpectrumSensingDataset
from rfml.data.splits import SplitBundle, load_split_bundle, resolve_split_indices
from rfml.data.transforms import STFTTransform
from rfml.models.cnn1d import CNN1D
from rfml.models.multitask import MultiTaskRFModel
from rfml.models.resnet1d import build_resnet1d
from rfml.models.stft_cnn import STFTCNN
from rfml.training.losses import build_classification_loss, compute_multitask_loss, compute_weighted_cross_entropy
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
    modulation_num_classes: int | None = None
    sensing_num_classes: int | None = None
    stft_n_fft: int | None = None
    stft_hop_length: int | None = None
    stft_window: str | None = None
    stft_output: str | None = None
    stft_backend: str | None = None
    sensing_positive_ratio: float | None = None
    sensing_noise_power: float | None = None
    sensing_seed: int = 42
    lambda_sensing: float = 1.0
    stft_backbone: str = "basic"
    best_metric: str = "val_loss"
    low_snr_threshold: float | None = None
    low_snr_weight: float = 1.0
    low_snr_oversample_factor: float = 1.0


@dataclass(frozen=True)
class EvaluationOutputs:
    loss: float
    accuracy: float
    labels: np.ndarray
    preds: np.ndarray
    snrs: np.ndarray
    extra: dict[str, Any] | None = None


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
        self.live_status_path = self.out_dir / "live_status.json"
        self.last_ckpt_path = self.out_dir / "last.pt"
        self.best_ckpt_path = self.out_dir / "best.pt"
        self.resume_ckpt = Path(resume_ckpt).expanduser().resolve() if resume_ckpt is not None else None

        self.device = torch.device(config.device)
        self.split_bundle = load_split_bundle(self.split_path)
        self.stft_transform = self._build_stft_transform()
        self.model = self._build_model().to(self.device)
        self.criterion = build_classification_loss("cross_entropy")
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self.best_val_loss = float("inf")
        self.best_metric_value = float("inf") if self._best_metric_mode() == "min" else float("-inf")
        self.start_epoch = 0
        self.history: list[dict[str, float | int]] = []
        self.fit_started_at = time.time()

        self.train_loader = self._build_loader("train", shuffle=True)
        self.val_loader = self._build_loader("val", shuffle=False)

        if self.resume_ckpt is not None and self.resume_ckpt.exists():
            self._load_checkpoint(self.resume_ckpt)
        if self.start_epoch >= self.config.epochs:
            raise ValueError(
                "resume checkpoint epoch is already >= configured total epochs; "
                f"checkpoint_epoch={self.start_epoch}, configured_epochs={self.config.epochs}. "
                "Increase training.epochs to continue fine-tuning."
            )

    def fit(self) -> dict[str, Any]:
        patience_counter = 0
        self.fit_started_at = time.time()
        self._write_live_status(
            status="running",
            phase="train",
            epoch=self.start_epoch + 1,
            num_epochs=self.config.epochs,
            history_length=len(self.history),
            best_metric_value=self.best_metric_value,
        )
        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                train_loss, train_acc = self._run_epoch(epoch, training=True)
                val_outputs = self.evaluate_loader(self.val_loader, epoch=epoch, phase="val")
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
                self._flush_history()

                current_metric = self._metric_value_from_row(row)
                is_best = self._is_better_metric(current_metric)
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_metric_value = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1

                self._save_checkpoint(epoch, is_best=is_best)

                if (epoch + 1) % self.config.save_every == 0:
                    self._save_checkpoint(epoch, is_best=False, named=True)

                self._write_live_status(
                    status="running",
                    phase="epoch_end",
                    epoch=epoch + 1,
                    num_epochs=self.config.epochs,
                    history_length=len(self.history),
                    latest_row=row,
                    patience_counter=patience_counter,
                    best_metric_value=self.best_metric_value,
                    is_best_epoch=is_best,
                )

                if patience_counter >= self.config.early_stopping_patience:
                    self._write_live_status(
                        status="completed",
                        phase="stopped_early",
                        epoch=epoch + 1,
                        num_epochs=self.config.epochs,
                        history_length=len(self.history),
                        latest_row=row,
                        best_metric_value=self.best_metric_value,
                        stopped_early=True,
                    )
                    break
        except KeyboardInterrupt:
            self._flush_history()
            self._write_live_status(
                status="interrupted",
                phase="train",
                epoch=min(self.config.epochs, self.start_epoch + len(self.history) + 1),
                num_epochs=self.config.epochs,
                history_length=len(self.history),
                best_metric_value=self.best_metric_value,
            )
            raise
        except Exception as exc:
            self._flush_history()
            self._write_live_status(
                status="failed",
                phase="train",
                epoch=min(self.config.epochs, self.start_epoch + len(self.history) + 1),
                num_epochs=self.config.epochs,
                history_length=len(self.history),
                best_metric_value=self.best_metric_value,
                error=repr(exc),
            )
            raise
        finally:
            self.writer.flush()
            self.writer.close()
            self._flush_history()

        self._write_live_status(
            status="completed",
            phase="done",
            epoch=self.history[-1]["epoch"] if self.history else self.start_epoch,
            num_epochs=self.config.epochs,
            history_length=len(self.history),
            latest_row=self.history[-1] if self.history else None,
            best_metric_value=self.best_metric_value,
        )
        return {
            "best_ckpt": str(self.best_ckpt_path),
            "last_ckpt": str(self.last_ckpt_path),
            "history": self.history,
        }

    def evaluate_loader(self, loader: DataLoader, *, epoch: int | None = None, phase: str = "val") -> EvaluationOutputs:
        if self.config.task == "multitask":
            return self._evaluate_multitask_loader(loader, epoch=epoch, phase=phase)

        self.model.eval()
        losses: list[float] = []
        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        snrs_list: list[torch.Tensor] = []
        total_batches = len(loader)
        update_interval = self._live_update_interval(total_batches)
        running_correct = 0
        running_total = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
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
                preds = torch.argmax(logits.detach(), dim=1)
                running_correct += int((preds == y).sum().item())
                running_total += int(y.numel())
                logits_list.append(logits.detach().cpu())
                labels_list.append(y.detach().cpu())
                snrs_list.append(snr.detach().cpu())
                if epoch is not None and self._should_update_live_status(batch_idx, total_batches, update_interval):
                    self._write_live_status(
                        status="running",
                        phase=phase,
                        epoch=epoch + 1,
                        num_epochs=self.config.epochs,
                        batch=batch_idx + 1,
                        num_batches=total_batches,
                        running_loss=float(np.mean(losses)),
                        running_acc=float(running_correct / max(1, running_total)),
                        best_metric_value=self.best_metric_value,
                        history_length=len(self.history),
                    )

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
        if self.config.task == "multitask":
            return self._run_multitask_epoch(epoch, training=training)

        loader = self.train_loader if training else self.val_loader
        self.model.train(training)
        losses: list[float] = []
        preds_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        total_batches = len(loader)
        update_interval = self._live_update_interval(total_batches)
        running_correct = 0
        running_total = 0

        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{self.config.epochs}", leave=False)
        for batch_idx, batch in enumerate(progress):
            x = batch["iq"].to(self.device, non_blocking=self.config.pin_memory)
            y = batch["label"].to(self.device, non_blocking=self.config.pin_memory)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.config.amp and self.device.type == "cuda",
            ):
                logits = self.model(x)
                if training and self.config.low_snr_threshold is not None and self.config.low_snr_weight != 1.0:
                    loss = compute_weighted_cross_entropy(
                        logits,
                        y,
                        sample_weights=self._build_low_snr_sample_weights(batch["snr"]),
                    )
                else:
                    loss = self.criterion(logits, y)

            if training:
                self.scaler.scale(loss).backward()
                if self.config.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            losses.append(float(loss.item()))
            preds = torch.argmax(logits.detach(), dim=1)
            running_correct += int((preds == y).sum().item())
            running_total += int(y.numel())
            preds_list.append(preds.cpu())
            labels_list.append(y.detach().cpu())
            progress.set_postfix(loss=f"{np.mean(losses):.4f}")
            if self._should_update_live_status(batch_idx, total_batches, update_interval):
                self._write_live_status(
                    status="running",
                    phase="train" if training else "val",
                    epoch=epoch + 1,
                    num_epochs=self.config.epochs,
                    batch=batch_idx + 1,
                    num_batches=total_batches,
                    running_loss=float(np.mean(losses)),
                    running_acc=float(running_correct / max(1, running_total)),
                    best_metric_value=self.best_metric_value,
                    history_length=len(self.history),
                )

        y_pred = torch.cat(preds_list, dim=0).numpy()
        y_true = torch.cat(labels_list, dim=0).numpy()
        return float(np.mean(losses)), compute_accuracy(y_true, y_pred)

    def _run_multitask_epoch(self, epoch: int, *, training: bool) -> tuple[float, float]:
        loader = self.train_loader if training else self.val_loader
        self.model.train(training)
        losses: list[float] = []
        mod_preds_list: list[torch.Tensor] = []
        mod_labels_list: list[torch.Tensor] = []
        mod_masks_list: list[torch.Tensor] = []
        total_batches = len(loader)
        update_interval = self._live_update_interval(total_batches)
        running_correct = 0
        running_total = 0

        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{self.config.epochs}", leave=False)
        for batch_idx, batch in enumerate(progress):
            x = batch["iq"].to(self.device, non_blocking=self.config.pin_memory)
            mod_y = batch["modulation_label"].to(self.device, non_blocking=self.config.pin_memory)
            sense_y = batch["sensing_label"].to(self.device, non_blocking=self.config.pin_memory)
            mod_mask = batch["mod_mask"].to(self.device, non_blocking=self.config.pin_memory)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.config.amp and self.device.type == "cuda",
            ):
                outputs = self.model(x)
                loss_output = compute_multitask_loss(
                    outputs["modulation_logits"],
                    mod_y,
                    mod_mask,
                    outputs["sensing_logits"],
                    sense_y,
                    lambda_sensing=self.config.lambda_sensing,
                )
                loss = loss_output.total_loss

            if training:
                self.scaler.scale(loss).backward()
                if self.config.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            losses.append(float(loss.item()))
            mod_preds = torch.argmax(outputs["modulation_logits"].detach(), dim=1)
            valid_mask = mod_mask > 0.5
            if torch.any(valid_mask):
                running_correct += int((mod_preds[valid_mask] == mod_y[valid_mask]).sum().item())
                running_total += int(valid_mask.sum().item())
            mod_preds_list.append(mod_preds.cpu())
            mod_labels_list.append(mod_y.detach().cpu())
            mod_masks_list.append(mod_mask.detach().cpu())
            progress.set_postfix(loss=f"{np.mean(losses):.4f}")
            if self._should_update_live_status(batch_idx, total_batches, update_interval):
                self._write_live_status(
                    status="running",
                    phase="train" if training else "val",
                    epoch=epoch + 1,
                    num_epochs=self.config.epochs,
                    batch=batch_idx + 1,
                    num_batches=total_batches,
                    running_loss=float(np.mean(losses)),
                    running_acc=float(running_correct / max(1, running_total)) if running_total > 0 else float("nan"),
                    best_metric_value=self.best_metric_value,
                    history_length=len(self.history),
                )

        mod_preds = torch.cat(mod_preds_list, dim=0).numpy()
        mod_labels = torch.cat(mod_labels_list, dim=0).numpy()
        mod_masks = torch.cat(mod_masks_list, dim=0).numpy() > 0.5
        if np.any(mod_masks):
            mod_acc = compute_accuracy(mod_labels[mod_masks], mod_preds[mod_masks])
        else:
            mod_acc = float("nan")
        return float(np.mean(losses)), mod_acc

    def _build_model(self) -> nn.Module:
        if self.config.task == "multitask":
            modulation_num_classes = int(self.config.modulation_num_classes or self.config.num_classes)
            sensing_num_classes = int(self.config.sensing_num_classes or 2)
            return MultiTaskRFModel(
                backbone=self.config.model_name,
                modulation_num_classes=modulation_num_classes,
                sensing_num_classes=sensing_num_classes,
                channels=self.config.channels,
                kernel_sizes=self.config.kernel_sizes,
                classifier_hidden_dim=self.config.classifier_hidden_dim,
                dropout=self.config.dropout,
            )
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
                in_channels=self._stft_input_channels(),
                num_classes=self.config.num_classes,
                channels=self.config.channels,
                dropout=self.config.dropout,
                classifier_hidden_dim=self.config.classifier_hidden_dim,
                backbone=self.config.stft_backbone,
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
        transform = self.stft_transform
        split_indices = resolve_split_indices(self.split_bundle, split_name)
        if self.config.task == "multitask":
            dataset = MultiTaskRadioMLDataset(
                self.h5_path,
                split_indices=split_indices,
                class_names=self.split_bundle.class_names,
                scan_chunk_size=self.config.scan_chunk_size,
                transform=transform,
                positive_ratio=float(self.config.sensing_positive_ratio or 0.5),
                noise_power=self.config.sensing_noise_power,
                seed=self.config.sensing_seed,
            )
        elif self.config.task == "spectrum_sensing":
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
        sampler = None
        effective_shuffle = shuffle
        if (
            split_name == "train"
            and self.config.task == "amc"
            and self.config.low_snr_threshold is not None
            and self.config.low_snr_oversample_factor > 1.0
        ):
            sampler = self._build_low_snr_sampler(dataset)
            effective_shuffle = False
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=effective_shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )

    def _build_stft_transform(self) -> STFTTransform | None:
        if self.config.model_name != "stft_cnn":
            return None
        return STFTTransform(
            n_fft=int(self.config.stft_n_fft or 128),
            hop_length=int(self.config.stft_hop_length or 32),
            window=str(self.config.stft_window or "hann"),
            output=str(self.config.stft_output or "log_power"),
            backend=str(self.config.stft_backend or "torch"),
        )

    def _stft_input_channels(self) -> int:
        return self.stft_transform.num_channels if self.stft_transform is not None else 1

    def _build_low_snr_sample_weights(self, snr: torch.Tensor) -> torch.Tensor:
        threshold = float(self.config.low_snr_threshold if self.config.low_snr_threshold is not None else 0.0)
        base = torch.ones_like(snr, dtype=torch.float32)
        boosted = torch.full_like(base, float(self.config.low_snr_weight))
        return torch.where(snr <= threshold, boosted, base)

    def _build_low_snr_sampler(self, dataset: RadioML2018Dataset) -> WeightedRandomSampler | None:
        threshold = float(self.config.low_snr_threshold if self.config.low_snr_threshold is not None else 0.0)
        factor = float(self.config.low_snr_oversample_factor)
        if factor <= 1.0:
            return None
        weights = np.ones(len(dataset), dtype=np.float64)
        batch_size = max(1, int(self.config.scan_chunk_size))
        with dataset.open_h5() as h5f:
            for start in range(0, len(dataset), batch_size):
                stop = min(start + batch_size, len(dataset))
                indices = dataset.indices[start:stop]
                snr_values = np.asarray(h5f["Z"][indices]).reshape(len(indices), -1)[:, 0]
                weights[start:stop] = np.where(snr_values <= threshold, factor, 1.0)
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )

    def _evaluate_multitask_loader(
        self,
        loader: DataLoader,
        *,
        epoch: int | None = None,
        phase: str = "val",
    ) -> EvaluationOutputs:
        self.model.eval()
        losses: list[float] = []
        mod_logits_list: list[torch.Tensor] = []
        mod_labels_list: list[torch.Tensor] = []
        mod_masks_list: list[torch.Tensor] = []
        sensing_logits_list: list[torch.Tensor] = []
        sensing_labels_list: list[torch.Tensor] = []
        snrs_list: list[torch.Tensor] = []
        total_batches = len(loader)
        update_interval = self._live_update_interval(total_batches)
        running_correct = 0
        running_total = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                x = batch["iq"].to(self.device, non_blocking=self.config.pin_memory)
                mod_y = batch["modulation_label"].to(self.device, non_blocking=self.config.pin_memory)
                sense_y = batch["sensing_label"].to(self.device, non_blocking=self.config.pin_memory)
                mod_mask = batch["mod_mask"].to(self.device, non_blocking=self.config.pin_memory)
                snr = batch["snr"]
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.config.amp and self.device.type == "cuda",
                ):
                    outputs = self.model(x)
                    loss_output = compute_multitask_loss(
                        outputs["modulation_logits"],
                        mod_y,
                        mod_mask,
                        outputs["sensing_logits"],
                        sense_y,
                        lambda_sensing=self.config.lambda_sensing,
                )
                losses.append(float(loss_output.total_loss.item()))
                modulation_logits = outputs["modulation_logits"].detach()
                modulation_preds = torch.argmax(modulation_logits, dim=1)
                valid_mask = mod_mask > 0.5
                if torch.any(valid_mask):
                    running_correct += int((modulation_preds[valid_mask] == mod_y[valid_mask]).sum().item())
                    running_total += int(valid_mask.sum().item())
                mod_logits_list.append(modulation_logits.cpu())
                mod_labels_list.append(mod_y.detach().cpu())
                mod_masks_list.append(mod_mask.detach().cpu())
                sensing_logits_list.append(outputs["sensing_logits"].detach().cpu())
                sensing_labels_list.append(sense_y.detach().cpu())
                snrs_list.append(snr.detach().cpu())
                if epoch is not None and self._should_update_live_status(batch_idx, total_batches, update_interval):
                    self._write_live_status(
                        status="running",
                        phase=phase,
                        epoch=epoch + 1,
                        num_epochs=self.config.epochs,
                        batch=batch_idx + 1,
                        num_batches=total_batches,
                        running_loss=float(np.mean(losses)),
                        running_acc=float(running_correct / max(1, running_total)) if running_total > 0 else float("nan"),
                        best_metric_value=self.best_metric_value,
                        history_length=len(self.history),
                    )

        mod_logits = torch.cat(mod_logits_list, dim=0)
        mod_labels = torch.cat(mod_labels_list, dim=0).numpy()
        mod_masks = torch.cat(mod_masks_list, dim=0).numpy() > 0.5
        sensing_logits = torch.cat(sensing_logits_list, dim=0)
        sensing_labels = torch.cat(sensing_labels_list, dim=0).numpy()
        snrs = torch.cat(snrs_list, dim=0).numpy()

        mod_preds = torch.argmax(mod_logits, dim=1).numpy()
        if np.any(mod_masks):
            mod_accuracy = compute_accuracy(mod_labels[mod_masks], mod_preds[mod_masks])
        else:
            mod_accuracy = float("nan")

        sensing_preds = torch.argmax(sensing_logits, dim=1).numpy()
        sensing_scores = torch.softmax(sensing_logits, dim=1)[:, 1].numpy()

        return EvaluationOutputs(
            loss=float(np.mean(losses)) if losses else float("nan"),
            accuracy=mod_accuracy,
            labels=mod_labels,
            preds=mod_preds,
            snrs=snrs,
            extra={
                "modulation_masks": mod_masks,
                "sensing_labels": sensing_labels,
                "sensing_preds": sensing_preds,
                "sensing_scores": sensing_scores,
            },
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

    def _flush_history(self) -> None:
        self.history_json_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def _live_update_interval(self, total_batches: int) -> int:
        return max(1, total_batches // 25)

    def _should_update_live_status(self, batch_idx: int, total_batches: int, update_interval: int) -> bool:
        return batch_idx == 0 or (batch_idx + 1) % update_interval == 0 or (batch_idx + 1) == total_batches

    def _write_live_status(self, *, status: str, phase: str, **payload: Any) -> None:
        data = {
            "status": status,
            "phase": phase,
            "task": self.config.task,
            "model_name": self.config.model_name,
            "device": str(self.device),
            "out_dir": str(self.out_dir),
            "best_metric": self.config.best_metric,
            "best_metric_value": None if not np.isfinite(self.best_metric_value) else float(self.best_metric_value),
            "resume_ckpt": str(self.resume_ckpt) if self.resume_ckpt is not None else None,
            "resume_start_epoch": self.start_epoch,
            "elapsed_seconds": float(max(0.0, time.time() - self.fit_started_at)),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        }
        data.update(payload)
        temp_path = self.live_status_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        temp_path.replace(self.live_status_path)

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        return {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_metric": self.config.best_metric,
            "best_metric_value": self.best_metric_value,
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
        # Resume model weights and optimizer moments, but keep the current config's
        # scheduled optimizer hyperparameters so fine-tune sweeps actually take effect.
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.config.lr)
            group["weight_decay"] = float(self.config.weight_decay)
        self.scaler.load_state_dict(payload["scaler_state_dict"])
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        default_metric_value = float("inf") if self._best_metric_mode() == "min" else float("-inf")
        self.best_metric_value = float(payload.get("best_metric_value", default_metric_value))
        self.start_epoch = int(payload.get("epoch", 0))

    def _best_metric_mode(self) -> str:
        metric_name = self.config.best_metric.lower()
        if metric_name == "val_loss":
            return "min"
        if metric_name == "val_acc":
            return "max"
        raise ValueError(f"Unsupported best_metric: {self.config.best_metric}")

    def _metric_value_from_row(self, row: dict[str, float | int]) -> float:
        metric_name = self.config.best_metric.lower()
        return float(row[metric_name])

    def _is_better_metric(self, current_metric: float) -> bool:
        if self._best_metric_mode() == "min":
            return current_metric < self.best_metric_value
        return current_metric > self.best_metric_value

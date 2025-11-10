"""
Lightweight trainer to keep parity with the Segment Anything repo style.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from CoralMonSter import CoralMonSter
from CoralMonSter.configs.hkcoral_config import HKCoralConfig
from CoralMonSter.utils import (
    SegmentationMeter,
    save_checkpoint,
    save_segmentation_comparison,
    save_training_curves,
)


class CoralTrainer:
    def __init__(self, model: CoralMonSter, cfg: HKCoralConfig, enable_profile: bool = False) -> None:
        self.model = model
        self.cfg = cfg
        self.enable_profile = enable_profile
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.optimization.learning_rate,
            weight_decay=cfg.optimization.weight_decay,
        )
        self.scheduler = None
        if self.cfg.optimization.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=self._lr_schedule,
            )
        self.log_dir = Path(self.cfg.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
        self.best_miou = 0.0
        self.best_model_path = self.cfg.save_dir / f"{self.cfg.model_type}_coralmonster.pth"
        self._setup_logger()
        self.last_val_loss = 0.0
        self.last_val_miou = 0.0
        self.last_val_pix = 0.0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> Path:
        device = next(self.model.parameters()).device
        profiled = False
        for epoch in range(self.cfg.optimization.max_epochs):
            momentum = self._compute_ema_momentum(epoch)
            if hasattr(self.model, "set_momentum"):
                self.model.set_momentum(momentum)
            teacher_temp = self._compute_teacher_temperature(epoch)
            if hasattr(self.model, "set_teacher_temperature"):
                self.model.set_teacher_temperature(teacher_temp)
            self.model.train()
            start = time.time()
            train_loss, train_miou, train_pix_acc = self._train_one_epoch(
                train_loader,
                device,
                epoch,
                profile_batch=(self.enable_profile and not profiled),
            )
            profiled = profiled or self.enable_profile
            duration = time.time() - start
            val_metrics = {"loss": 0.0, "miou": 0.0, "pix_acc": 0.0, "preview": None}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, device)
                self.last_val_loss = val_metrics["loss"]
                self.last_val_miou = val_metrics["miou"]
                self.last_val_pix = val_metrics["pix_acc"]
                if val_metrics["preview"] is not None:
                    preview = val_metrics["preview"]
                    save_segmentation_comparison(
                        preview["image"],
                        preview["prediction"],
                        preview["mask"],
                        mean=self.cfg.image_mean,
                        std=self.cfg.image_std,
                        palette=self.cfg.class_palette,
                        path=self.log_dir / f"epoch_{epoch+1:03d}_comparison.png",
                        title=preview["file_name"],
                    )
                if val_metrics["miou"] > self.best_miou:
                    self.best_miou = val_metrics["miou"]
                    save_checkpoint({"model": self.model.state_dict()}, self.best_model_path)
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "train_miou": train_miou,
                "val_miou": val_metrics["miou"],
                "train_pix_acc": train_pix_acc,
                "val_pix_acc": val_metrics["pix_acc"],
                "duration_sec": duration,
            }
            self.history.append(metrics)
            self._log_metrics(metrics)
            save_training_curves(self.history, self.log_dir / "training_curves.png")
            message = (
                f"Epoch {epoch+1}/{self.cfg.optimization.max_epochs} "
                f"- train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
                f"train_mIoU={train_miou:.4f}, val_mIoU={val_metrics['miou']:.4f}, "
                f"train_pixAcc={train_pix_acc:.4f}, val_pixAcc={val_metrics['pix_acc']:.4f}"
            )
            print(message)
            if self.scheduler is not None:
                self.scheduler.step()

        if test_loader is not None:
            test_metrics = self.evaluate(test_loader, device)
            self.logger.info(
                "Test | loss=%.4f | mIoU=%.4f | pixAcc=%.4f",
                test_metrics["loss"],
                test_metrics["miou"],
                test_metrics["pix_acc"],
            )
            print(
                f"Test set - loss={test_metrics['loss']:.4f}, "
                f"mIoU={test_metrics['miou']:.4f}, pixAcc={test_metrics['pix_acc']:.4f}"
            )
        return self.best_model_path if self.best_miou > 0 else None

    def _train_one_epoch(
        self,
        loader: DataLoader,
        device: torch.device,
        epoch: int,
        profile_batch: bool = False,
    ) -> tuple[float, float, float]:
        loss_sum = 0.0
        count = 0
        meter = SegmentationMeter(self.cfg.num_classes, self.cfg.ignore_label)
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{self.cfg.optimization.max_epochs}",
            leave=False,
        )
        profiled = False
        for batch in progress:
            io_start = time.time()
            batch = self._to_device(batch, device)
            data_io = time.time() - io_start
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = self.model(batch)
            loss = outputs["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.optimization.grad_clip_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            if epoch != 0:
                self.model.update_teacher()

            batch_size = batch["images"].shape[0]
            loss_sum += loss.item() * batch_size
            count += batch_size
            preds = outputs["logits"].argmax(dim=1).detach().cpu()
            masks = batch.get("masks")
            if masks is not None:
                meter.update(preds, masks.detach().cpu())
            train_loss = loss_sum / max(count, 1)
            loss_postfix = {"loss_total": f"{train_loss:.4f}"}
            for key in ("seg_loss", "mask_kd_loss", "token_kd_loss"):
                if key in outputs:
                    label = key.replace("_loss", "")
                    loss_postfix[f"loss_{label}"] = f"{outputs[key].item():.4f}"
            progress.set_postfix(**loss_postfix)

            if profile_batch and not profiled:
                teacher_time = outputs.get("teacher_time", 0.0)
                encoder_time = outputs.get("encoder_time", 0.0)
                student_time = outputs.get("student_time", 0.0)
                print(
                    f"[PROFILE] Data IO={data_io*1000:.2f} ms | "
                    f"Image Encoder={encoder_time*1000:.2f} ms | "
                    f"Student Decoder={student_time*1000:.2f} ms | "
                    f"Teacher Decoder={teacher_time*1000:.2f} ms"
                )
                profiled = True
        return loss_sum / max(count, 1), meter.mean_iou(), meter.pixel_accuracy()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, device: torch.device) -> Dict[str, object]:
        self.model.eval()
        loss_sum = 0.0
        count = 0
        meter = SegmentationMeter(self.cfg.num_classes, self.cfg.ignore_label)
        preview = None
        for batch in loader:
            batch = self._to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = self.model.forward(batch)
            batch_size = batch["images"].shape[0]
            loss_sum += outputs["total_loss"].item() * batch_size
            count += batch_size
            preds = outputs["logits"].argmax(dim=1).detach().cpu()
            masks = batch.get("masks")
            if masks is not None:
                meter.update(preds, masks.detach().cpu())
                if preview is None:
                    preview = {
                        "image": batch["images"][0].detach().cpu(),
                        "mask": masks[0].detach().cpu(),
                        "prediction": preds[0].detach().cpu(),
                        "file_name": batch["file_names"][0],
                    }
        return {
            "loss": loss_sum / max(count, 1),
            "miou": meter.mean_iou(),
            "pix_acc": meter.pixel_accuracy(),
            "preview": preview,
        }

    @staticmethod
    def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        moved = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                moved[k] = v.to(device)
            elif isinstance(v, list):
                moved[k] = [item.to(device) if torch.is_tensor(item) else item for item in v]
            else:
                moved[k] = v
        return moved

    def _compute_ema_momentum(self, epoch_idx: int) -> float:
        min_m = self.cfg.optimization.ema_momentum_min
        max_m = self.cfg.optimization.ema_momentum_max
        if max_m < min_m:
            min_m, max_m = max_m, min_m
        total = max(1, self.cfg.optimization.max_epochs - 1)
        progress = min(epoch_idx / total, 1.0)
        scaled = math.sin(0.5 * math.pi * progress)
        return min_m + (max_m - min_m) * scaled

    def _compute_teacher_temperature(self, epoch_idx: int) -> float:
        start = self.cfg.distillation.teacher_temperature_start
        end = self.cfg.distillation.teacher_temperature_end
        warmup = max(1, self.cfg.distillation.teacher_temperature_warmup_epochs)
        progress = min((epoch_idx + 1) / warmup, 1.0)
        return start + (end - start) * progress

    def _lr_schedule(self, current_step: int) -> float:
        total_epochs = max(1, self.cfg.optimization.max_epochs)
        warmup_epochs = max(1, self.cfg.optimization.lr_warmup_epochs)
        epoch = current_step

        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        progress = (epoch - warmup_epochs + 1) / max(1, total_epochs - warmup_epochs)
        min_factor = self.cfg.optimization.lr_min_factor
        cosine = 0.5 * (1 + math.sin(math.pi * (progress - 0.5)))
        return min_factor + (1 - min_factor) * cosine

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger(f"CoralTrainer_{self.cfg.scenario_name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_dir / "training.log")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        self.logger.info(
            "Epoch %03d | train_loss=%.4f | val_loss=%.4f | train_mIoU=%.4f | val_mIoU=%.4f | "
            "train_pixAcc=%.4f | val_pixAcc=%.4f",
            metrics["epoch"],
            metrics["train_loss"],
            metrics["val_loss"],
            metrics["train_miou"],
            metrics["val_miou"],
            metrics["train_pix_acc"],
            metrics["val_pix_acc"],
        )
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

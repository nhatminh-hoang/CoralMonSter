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
    save_pca_overlay,
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
        self.pca_counts = {"train": 0, "val": 0, "test": 0}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> Path:
        device = next(self.model.parameters()).device
        profiled = False
        skip_epochs = getattr(self.cfg.optimization, "momentum_skip_epochs", 0)
        for epoch in range(self.cfg.optimization.max_epochs):
            teacher_active = epoch >= skip_epochs
            self.model.set_distillation_enabled(teacher_active)
            momentum = self._compute_ema_momentum(epoch)
            if hasattr(self.model, "set_momentum"):
                self.model.set_momentum(momentum)
            teacher_temp = self._compute_teacher_temperature(epoch)
            if hasattr(self.model, "set_teacher_temperature"):
                self.model.set_teacher_temperature(teacher_temp)
            self.model.train()
            start = time.time()
            (
                train_loss,
                train_miou,
                train_pix_acc,
                train_components,
                train_class_iou,
                train_confusion,
            ) = self._train_one_epoch(
                train_loader,
                device,
                epoch,
                profile_batch=(self.enable_profile and not profiled),
            )
            profiled = profiled or self.enable_profile
            duration = time.time() - start
            val_metrics = {"loss": 0.0, "miou": 0.0, "pix_acc": 0.0, "preview": None, "components": {}}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, device, stage="val", epoch_idx=epoch)
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
                        class_names=self.cfg.class_names,
                        path=self.log_dir / f"epoch_{epoch+1:03d}_comparison.png",
                        title=preview["file_name"],
                        teacher_mask=preview.get("teacher_mask"),
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
                "train_seg_loss": train_components.get("seg_loss", 0.0),
                "train_mask_kd_loss": train_components.get("mask_kd_loss", 0.0),
                "train_token_kd_loss": train_components.get("token_kd_loss", 0.0),
                "train_token_cls_loss": train_components.get("token_cls_loss", 0.0),
                "train_dice_loss": train_components.get("dice_loss", 0.0),
                "train_ce_loss": train_components.get("ce_loss", 0.0),
                "val_seg_loss": val_metrics["components"].get("seg_loss", 0.0),
                "val_mask_kd_loss": val_metrics["components"].get("mask_kd_loss", 0.0),
                "val_token_kd_loss": val_metrics["components"].get("token_kd_loss", 0.0),
                "val_token_cls_loss": val_metrics["components"].get("token_cls_loss", 0.0),
                "val_dice_loss": val_metrics["components"].get("dice_loss", 0.0),
                "val_ce_loss": val_metrics["components"].get("ce_loss", 0.0),
                "train_class_iou": train_class_iou,
                "val_class_iou": val_metrics["class_iou"],
                "train_confusion": train_confusion.tolist(),
                "val_confusion": val_metrics["confusion_matrix"],
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
            test_vis_dir = self.log_dir / "test_visuals"
            test_metrics = self.evaluate(
                test_loader,
                device,
                stage="test",
                epoch_idx=self.cfg.optimization.max_epochs - 1,
                visualize_dir=test_vis_dir,
            )
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
    ) -> tuple[float, float, float, Dict[str, float], list[float], torch.Tensor]:
        loss_sum = 0.0
        count = 0
        meter = SegmentationMeter(
            self.cfg.num_classes,
            self.cfg.ignore_label,
            getattr(self.cfg, "eval_ignore_classes", None),
        )
        self.pca_counts["train"] = 0
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{self.cfg.optimization.max_epochs}",
            leave=False,
        )
        profiled = False
        component_sums: Dict[str, float] = {}
        timing_totals = {
            "encoder_time": 0.0,
            "student_time": 0.0,
            "teacher_time": 0.0,
            "backward_time": 0.0,
            "cpu_loader": 0.0,
        }
        timing_steps = 0
        for batch in progress:
            loader_timings = batch.pop("timings", None)
            if loader_timings:
                cpu_total = float(loader_timings.get("total", 0.0))
                if cpu_total <= 0.0:
                    cpu_total = sum(float(value) for value in loader_timings.values())
                timing_totals["cpu_loader"] += cpu_total
            batch = self._to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = self.model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            loss = outputs["total_loss"]
            backward_start = time.time()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.optimization.grad_clip_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            timing_totals["backward_time"] += time.time() - backward_start
            if (
                self.cfg.optimization.use_teacher_momentum
                and getattr(self.model, "distillation_enabled", True)
                and epoch != 0
            ):
                self.model.update_teacher()

            self._maybe_log_pca("train", epoch, batch, outputs)

            batch_size = batch["images"].shape[0]
            loss_sum += loss.item() * batch_size
            count += batch_size
            preds = outputs["logits"].argmax(dim=1).detach().cpu()
            masks = batch.get("masks")
            if masks is not None:
                meter.update(preds, masks.detach().cpu())
            train_loss = loss_sum / max(count, 1)
            loss_postfix = {"loss_total": f"{train_loss:.4f}"}
            for key in ("seg_loss", "mask_kd_loss", "token_kd_loss", "token_cls_loss", "dice_loss", "ce_loss"):
                if key in outputs:
                    label = key.replace("_loss", "")
                    loss_postfix[f"loss_{label}"] = f"{outputs[key].item():.4f}"
                    component_sums[key] = component_sums.get(key, 0.0) + outputs[key].item() * batch_size
            for key in ("encoder_time", "student_time", "teacher_time"):
                if key in outputs:
                    timing_totals[key] += outputs[key]
            timing_steps += 1
            avg_postfix = {}
            if timing_steps > 0:
                avg_postfix = {
                    "enc_ms": f"{(timing_totals['encoder_time'] / timing_steps) * 1000:.1f}",
                    "stu_ms": f"{(timing_totals['student_time'] / timing_steps) * 1000:.1f}",
                    "teach_ms": f"{(timing_totals['teacher_time'] / timing_steps) * 1000:.1f}",
                    "bwd_ms": f"{(timing_totals['backward_time'] / timing_steps) * 1000:.1f}",
                    "cpu_ms": f"{(timing_totals['cpu_loader'] / timing_steps) * 1000:.1f}",
                }
            progress.set_postfix({**loss_postfix, **avg_postfix})

            if profile_batch and not profiled:
                teacher_time = outputs.get("teacher_time", 0.0)
                encoder_time = outputs.get("encoder_time", 0.0)
                student_time = outputs.get("student_time", 0.0)
                print(
                    f"[PROFILE] Image Encoder={encoder_time*1000:.2f} ms | "
                    f"Student Decoder={student_time*1000:.2f} ms | "
                    f"Teacher Decoder={teacher_time*1000:.2f} ms"
                )
                profiled = True
        component_avgs = {k: v / max(count, 1) for k, v in component_sums.items()}
        return (
            loss_sum / max(count, 1),
            meter.mean_iou(),
            meter.pixel_accuracy(),
            component_avgs,
            meter.per_class_iou(),
            meter.confusion_matrix(),
        )

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        device: torch.device,
        stage: str = "val",
        epoch_idx: Optional[int] = None,
        visualize_dir: Optional[Path] = None,
    ) -> Dict[str, object]:
        self.model.eval()
        loss_sum = 0.0
        count = 0
        meter = SegmentationMeter(
            self.cfg.num_classes,
            self.cfg.ignore_label,
            getattr(self.cfg, "eval_ignore_classes", None),
        )
        preview = None
        if visualize_dir is not None:
            visualize_dir.mkdir(parents=True, exist_ok=True)
        component_sums: Dict[str, float] = {}
        self.pca_counts[stage] = 0
        enable_distillation = stage != "test"
        for batch in tqdm(loader):
            batch = self._to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = self.model.forward(batch, compute_distillation=enable_distillation)
            self._maybe_log_pca(stage, epoch_idx, batch, outputs)
            batch_size = batch["images"].shape[0]
            loss_sum += outputs["total_loss"].item() * batch_size
            count += batch_size
            preds = outputs["logits"].argmax(dim=1).detach().cpu()
            masks = batch.get("masks")
            teacher_masks = None
            # Teacher predictions are optional and only computed when we plan to visualize them.
            if (visualize_dir is not None or preview is None) and batch.get("prompt_sets"):
                teacher_pred = self.model.teacher_predict(batch)
                if teacher_pred is not None:
                    teacher_masks = teacher_pred.detach().cpu()
            if masks is not None:
                meter.update(preds, masks.detach().cpu())
                if preview is None:
                    preview = {
                        "image": batch["images"][0].detach().cpu(),
                        "mask": masks[0].detach().cpu(),
                        "prediction": preds[0].detach().cpu(),
                        "teacher_mask": teacher_masks[0] if teacher_masks is not None else None,
                        "file_name": batch["file_names"][0],
                    }
                if visualize_dir is not None:
                    for i in range(batch_size):
                        file_stem = Path(batch["file_names"][i]).stem
                        save_segmentation_comparison(
                            batch["images"][i].detach().cpu(),
                            preds[i],
                            masks[i],
                            mean=self.cfg.image_mean,
                            std=self.cfg.image_std,
                            palette=self.cfg.class_palette,
                            class_names=self.cfg.class_names,
                            path=visualize_dir / f"{file_stem}.png",
                            title=batch["file_names"][i],
                            teacher_mask=teacher_masks[i] if teacher_masks is not None else None,
                        )
                for key in ("seg_loss", "mask_kd_loss", "token_kd_loss", "token_cls_loss", "dice_loss", "ce_loss"):
                    if key in outputs:
                        component_sums[key] = component_sums.get(key, 0.0) + outputs[key].item() * batch_size
        return {
            "loss": loss_sum / max(count, 1),
            "miou": meter.mean_iou(),
            "pix_acc": meter.pixel_accuracy(),
            "preview": preview,
            "components": {k: v / max(count, 1) for k, v in component_sums.items()},
            "class_iou": meter.per_class_iou(),
            "confusion_matrix": meter.confusion_matrix().tolist(),
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
        base_lr = self.cfg.optimization.learning_rate
        target_lr = 1e-6
        target_factor = target_lr / max(base_lr, 1e-12)

        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        progress = (epoch - warmup_epochs + 1) / max(1, total_epochs - warmup_epochs)
        min_factor = min(self.cfg.optimization.lr_min_factor, target_factor)
        cosine = 0.5 * (1 + math.sin(math.pi * (progress - 0.5)))
        if epoch >= total_epochs - 1:
            return target_factor
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

    def _maybe_log_pca(
        self,
        stage: str,
        epoch_idx: Optional[int],
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        if not getattr(self.cfg, "enable_pca_logging", False):
            return
        limit = max(0, getattr(self.cfg, "pca_samples_per_epoch", 0))
        if limit == 0:
            return
        if stage not in self.pca_counts:
            self.pca_counts[stage] = 0
        if self.pca_counts[stage] >= limit:
            return
        embeddings = outputs.get("encoder_embeddings")
        if embeddings is None:
            return
        images = batch.get("images")
        if images is None:
            return
        file_names = batch.get("file_names") or []
        remaining = limit - self.pca_counts[stage]
        save_dir = self.log_dir / getattr(self.cfg, "pca_log_dirname", "encoder_pca") / stage
        if epoch_idx is not None:
            save_dir = save_dir / f"epoch_{epoch_idx + 1:03d}"
        for idx in range(min(remaining, embeddings.shape[0])):
            name = file_names[idx] if idx < len(file_names) else f"sample_{self.pca_counts[stage] + 1}"
            stem = Path(name).stem
            out_path = save_dir / f"{stem}_pca.png"
            title = f"{stage.upper()} | {name}"
            image_tensor = images[idx].detach().cpu()
            embedding_tensor = embeddings[idx]
            try:
                save_pca_overlay(
                    image_tensor,
                    embedding_tensor,
                    self.cfg.image_mean,
                    self.cfg.image_std,
                    out_path,
                    title,
                )
            except Exception as exc:  # pragma: no cover - visualization failures shouldn't break training
                self.logger.warning("Failed to save PCA overlay for %s: %s", name, exc)
            self.pca_counts[stage] += 1
            if self.pca_counts[stage] >= limit:
                break

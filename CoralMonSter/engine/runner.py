"""
Unified Runner for CoralMonSter training, evaluation, and inference.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from CoralMonSter import CoralTrainer
from CoralMonSter.core.config import HKCoralConfig
from CoralMonSter.core.factory import build_model, build_dataset, prepare_dataloader
from CoralMonSter.utils import save_checkpoint
from CoralMonSter.utils.common import infer_model_type_from_checkpoint
from CoralMonSter.utils.inference import preprocess_image, masks_to_json
from CoralMonSter.utils.visualize import save_confusion_figure


class CoralRunner:
    def __init__(self, args, mode: str = "train"):
        self.args = args
        self.mode = mode
        self._setup_device()
        
        # Infer model type if not provided
        if hasattr(args, "sam_checkpoint") and args.sam_checkpoint:
             self._infer_model_type(Path(args.sam_checkpoint))
        elif hasattr(args, "checkpoint_root") and mode == "eval":
             # Eval mode handles model type per checkpoint usually, but we set a default
             pass

    def _setup_device(self):
        if self.args.gpu_devices and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_devices
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.args.gpu)
            torch.set_float32_matmul_precision("high")

    def _infer_model_type(self, checkpoint_path: Path):
        inferred = infer_model_type_from_checkpoint(checkpoint_path)
        if inferred and self.args.model_type and inferred != self.args.model_type:
            print(
                f"[Warning] Checkpoint '{checkpoint_path}' looks like '{inferred}' "
                f"but '--model_type' was set to '{self.args.model_type}'. Using '{self.args.model_type}'."
            )
        elif self.args.model_type is None and inferred:
            print(f"[Info] Detected backbone '{inferred}' from checkpoint name.")
            self.args.model_type = inferred
        elif self.args.model_type is None and inferred is None:
            print("[Info] Could not infer backbone from checkpoint name; defaulting to 'vit_h'.")
            self.args.model_type = "vit_h"

    def train(self, cfg: HKCoralConfig):
        # Build Model
        model = build_model(cfg, self.device)
        
        if self.args.resume:
            state = torch.load(self.args.resume, map_location="cpu")
            model.load_state_dict(state.get("model", state), strict=False)

        # Build Datasets & Loaders
        train_loader = self._build_loader(cfg, "train", self.args.dataset.lower(), shuffle=True, limit=self.args.train_subset)
        val_loader = self._build_loader(cfg, "val", self.args.dataset.lower(), shuffle=False, limit=self.args.val_subset)
        test_loader = self._build_loader(cfg, "test", self.args.dataset.lower(), shuffle=False, limit=self.args.test_subset)

        # Train
        trainer = CoralTrainer(model, cfg, enable_profile=self.args.profile)
        best_path = trainer.fit(train_loader, val_loader, test_loader)

        final_path = cfg.save_dir / f"{cfg.model_type}_coralmonster_last.pth"
        save_checkpoint({"model": model.state_dict()}, final_path)
        print(f"Final checkpoint saved to {final_path}")
        if best_path:
            print(f"Best mIoU checkpoint saved to {best_path}")

    def evaluate(self, cfg: HKCoralConfig, ckpt_path: Path, visualize_dir: Optional[Path] = None) -> Dict:
        model = build_model(cfg, self.device)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
        
        dataset = build_dataset(cfg, self.args.split, self.args.dataset)
        loader = prepare_dataloader(dataset, self.args.batch_size, self.args.num_workers)
        
        trainer = CoralTrainer(model, cfg)
        metrics = trainer.evaluate(loader, self.device, stage=cfg.split, visualize_dir=visualize_dir)
        
        return {
            "checkpoint": str(ckpt_path),
            "scenario": cfg.scenario_name,
            "split": cfg.split,
            "loss": metrics["loss"],
            "miou": metrics["miou"],
            "pixel_accuracy": metrics["pix_acc"],
            "class_miou": metrics["class_iou"],
            "confusion_matrix": metrics["confusion_matrix"],
        }

    def predict(self, cfg: HKCoralConfig, student_weights: Optional[str] = None):
        model = build_model(cfg, self.device)
        
        if student_weights:
            state = torch.load(student_weights, map_location="cpu")
            model_state = state.get("model", state)
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            if missing or unexpected:
                print(f"[Warning] Missing keys: {missing} | Unexpected keys: {unexpected}")
                
        model.eval()
        output_dir = Path(self.args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = sorted(glob.glob(os.path.join(self.args.test_img_path, "*.*")))
        for path in image_paths:
            path = Path(path)
            if path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue
            target_json = output_dir / f"{path.stem}.json"
            if target_json.exists():
                continue
            
            sample = preprocess_image(path, self.args.image_size, model.cfg.image_mean, model.cfg.image_std)
            image = sample["image"].unsqueeze(0).to(self.device)
            original_size = sample["original_size"].unsqueeze(0)
            
            with torch.no_grad():
                logits = model.predict(image, original_size)[0]
            
            json_dict = masks_to_json(path.name, logits, self.args.prob_threshold, cfg.class_names)
            with open(target_json, "w") as fp:
                json.dump(json_dict, fp)
            print(f"Saved predictions for {path.name}")

    def _build_loader(self, cfg, split, dataset_name, shuffle, limit):
        dataset = build_dataset(cfg, split, dataset_name)
        return prepare_dataloader(
            dataset, 
            cfg.optimization.batch_size, 
            cfg.optimization.num_workers, 
            shuffle=shuffle, 
            limit=limit
        )

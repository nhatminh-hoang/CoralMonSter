"""
Visualize CoralMonSter predictions for test samples that have prompt points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from CoralMonSter import CoralMonSter as CoralModel
from CoralMonSter import HKCoralConfig
from CoralMonSter.data import HKCoralDataset, hkcoral_collate_fn
from CoralMonSter.utils import save_segmentation_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CoralMonSter predictions with prompts.")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Base SAM checkpoint path.")
    parser.add_argument("--student_weights", type=str, required=True, help="Fine-tuned CoralMonSter weights.")
    parser.add_argument("--dataset_root", type=str, default="datasets/HKCoral")
    parser.add_argument("--scenario_name", type=str, default="viz_prompts")
    parser.add_argument("--model_type", type=str, default=None, choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="visualizations/prompts")
    parser.add_argument("--max_samples", type=int, default=50)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = HKCoralConfig(
        dataset_root=Path(args.dataset_root),
        split="test",
        image_size=args.image_size,
        model_type=args.model_type or "vit_h",
        sam_checkpoint=Path(args.sam_checkpoint),
        scenario_name=args.scenario_name,
    )
    cfg.resolve_paths()

    model = CoralModel(cfg).to(device)
    weights = torch.load(args.student_weights, map_location="cpu")
    model.load_state_dict(weights.get("model", weights), strict=False)
    model.eval()

    dataset = HKCoralDataset(
        cfg.dataset_root,
        "test",
        cfg.image_size,
        cfg.num_classes,
        cfg.ignore_label,
        prompt_points=cfg.distillation.prompt_points,
        prompt_bins=cfg.prompt_bins,
        mean=cfg.image_mean,
        std=cfg.image_std,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=hkcoral_collate_fn)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for batch in loader:
        coords = batch["point_coords"][0]
        if coords.numel() == 0:
            continue
        images = batch["images"].to(device)
        original_size = batch["original_sizes"]
        preds = model.predict(images, original_size)[0]
        save_segmentation_comparison(
            batch["images"][0],
            preds.argmax(dim=0).cpu(),
            batch["masks"][0],
            mean=cfg.image_mean,
            std=cfg.image_std,
            palette=cfg.class_palette,
            class_names=cfg.class_names,
            prompt_coords=coords,
            prompt_labels=batch["point_labels"][0],
            path=output_dir / f"{Path(batch['file_names'][0]).stem}.png",
            title=batch["file_names"][0],
        )
        saved += 1
        if saved >= args.max_samples:
            break

    print(f"Saved {saved} visualizations to {output_dir}")


if __name__ == "__main__":
    main()

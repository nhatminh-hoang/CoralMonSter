import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pycocotools.mask as mask_util
import torch
from PIL import Image
from torchvision import transforms

from CoralMonSter import CoralMonSter, HKCoralConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-free CoralMonSter inference")
    parser.add_argument("--model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to the base SAM checkpoint")
    parser.add_argument("--student_weights", type=str, default=None, help="Fine-tuned CoralMonSter weights")
    parser.add_argument("--test_img_path", type=str, required=True, help="Directory with test images")
    parser.add_argument("--output_path", type=str, required=True, help="Directory where JSON predictions are stored")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> CoralMonSter:
    cfg = HKCoralConfig(
        dataset_root=Path(args.test_img_path),
        split="test",
        image_size=args.image_size,
        model_type=args.model_type,
        sam_checkpoint=Path(args.sam_checkpoint),
    )
    model = CoralMonSter(cfg)
    if args.student_weights:
        state = torch.load(args.student_weights, map_location="cpu")
        model_state = state.get("model", state)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing or unexpected:
            print(f"[Warning] Missing keys: {missing} | Unexpected keys: {unexpected}")
    return model


def preprocess_image(path: Path, image_size: int, mean, std) -> Dict[str, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    original_size = torch.tensor((image.height, image.width), dtype=torch.long)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return {
        "image": transform(image),
        "original_size": original_size,
    }


def masks_to_json(
    file_name: str,
    logits: torch.Tensor,
    threshold: float,
    class_names: List[str],
) -> Dict:
    probs = torch.softmax(logits, dim=0)
    height, width = probs.shape[1:]
    annotations = []
    anno_id = 0
    for class_id in range(1, probs.shape[0]):
        mask = probs[class_id] > threshold
        if mask.sum() == 0:
            continue
        mask_np = mask.cpu().numpy().astype(np.uint8)
        rle = mask_util.encode(np.asfortranarray(mask_np))
        rle["counts"] = rle["counts"].decode("utf-8")
        ys, xs = np.where(mask_np == 1)
        bbox = [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() - xs.min() + 1),
            float(ys.max() - ys.min() + 1),
        ]
        annotations.append(
            {
                "id": anno_id,
                "category_id": class_id,
                "category_name": class_names[class_id],
                "segmentation": rle,
                "area": float(mask_np.sum()),
                "bbox": bbox,
                "predicted_iou": float(probs[class_id][mask].mean().item()),
            }
        )
        anno_id += 1

    return {
        "image": {
            "image_id": 0,
            "width": int(width),
            "height": int(height),
            "file_name": file_name,
        },
        "annotations": annotations,
    }


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    model.eval()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = [
        "background",
        "massive",
        "encrusting",
        "branching",
        "laminar",
        "folliaceous",
        "columnar",
    ]

    image_paths = sorted(glob.glob(os.path.join(args.test_img_path, "*.*")))
    for path in image_paths:
        path = Path(path)
        if path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        target_json = output_dir / f"{path.stem}.json"
        if target_json.exists():
            continue
        sample = preprocess_image(path, args.image_size, model.cfg.image_mean, model.cfg.image_std)
        image = sample["image"].unsqueeze(0).to(device)
        original_size = sample["original_size"].unsqueeze(0)
        with torch.no_grad():
            logits = model.predict(image, original_size)[0]
        json_dict = masks_to_json(path.name, logits, args.prob_threshold, class_names)
        with open(target_json, "w") as fp:
            json.dump(json_dict, fp)
        print(f"Saved predictions for {path.name}")


if __name__ == "__main__":
    main()

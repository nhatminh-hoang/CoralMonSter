import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# CoralMonSter Imports
from coralmonster.arch import CoralMonSter
from coralmonster.encoder import build_sam_encoder
from data.hkcoral import HKCoralDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser(description="CoralMonSter Evaluation")
    parser.add_argument("--config", type=str, default="configs/hkcoral_vit_b.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    print("Building model...")
    sam_backbone = build_sam_encoder(
        model_type=cfg['model']['type'],
        checkpoint_path=cfg['model']['checkpoint'],
        image_size=cfg['data']['image_size'],
    )
    model = CoralMonSter(
        sam_backbone,
        num_classes=cfg['model']['num_classes'],
        image_size=cfg['data']['image_size'],
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Dataset
    dataset = HKCoralDataset(
        root=cfg['data']['root'],
        split="test", # Assuming test split exists or use 'val'
        image_size=cfg['data']['image_size'],
        num_classes=cfg['model']['num_classes']
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting evaluation...")
    for idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        original_size = batch['original_size'] # (B, 2)
        
        # Inference
        outputs = model(images, compute_distillation=False)
        logits = outputs["student_logits"] # 1024x1024
        
        # Resize to original
        h, w = original_size[0]
        pred_mask = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred_mask = pred_mask.argmax(dim=1).float() / (cfg['model']['num_classes'] - 1)
        
        save_image(pred_mask, output_dir / f"pred_{idx:04d}.png")
        print(f"Saved pred_{idx:04d}.png")

if __name__ == "__main__":
    evaluate()

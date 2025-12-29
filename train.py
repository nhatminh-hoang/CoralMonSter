import argparse
import yaml
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW

# CoralMonSter Imports
from coralmonster.arch import CoralMonSter
from coralmonster.encoder import build_sam_encoder
from coralmonster.loss import CoralMonsterLoss
from data.hkcoral import HKCoralDataset
from utils.scheduling import get_params_schedule_with_warmup, cosine_scheduler

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="CoralMonSter Training")
    parser.add_argument("--config", type=str, default="configs/hkcoral_vit_b.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # 1. Dataset
    print(f"Loading dataset: {cfg['data']['dataset']}")
    # Assuming HKCoral for now
    dataset_root = cfg['data']['root']
    train_dataset = HKCoralDataset(
        root=dataset_root,
        split="train",
        image_size=cfg['data']['image_size'],
        num_classes=cfg['model']['num_classes']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        drop_last=True
    )
    
    # 2. Model
    print(f"Building SAM backbone: {cfg['model']['type']}")
    sam_backbone = build_sam_encoder(
        model_type=cfg['model']['type'],
        checkpoint_path=cfg['model']['checkpoint'],
        image_size=cfg['data']['image_size'],
    )
    
    print("Initializing CoralMonSter...")
    model = CoralMonSter(
        sam_backbone,
        num_classes=cfg['model']['num_classes'],
        image_size=cfg['data']['image_size'],
        freeze_image_encoder=cfg['model']['freeze_image_encoder'],
        initial_momentum=cfg['distillation']['momentum_start']
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.set_distillation_enabled(True)
    
    # 3. Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        params,
        lr=float(cfg['optimizer']['lr']),
        weight_decay=float(cfg['optimizer']['weight_decay']),
        betas=tuple(cfg['optimizer']['betas'])
    )
    
    # 4. Schedulers
    total_steps = len(train_loader) * cfg['training']['max_epochs']
    warmup_steps = cfg['distillation']['warmup_steps']
    
    # LR Scheduler (Cosine with Warmup)
    lr_scheduler = get_params_schedule_with_warmup(
        optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=warmup_steps,
    )
    
    # Momentum Schedule
    momentum_schedule = cosine_scheduler(
        base_value=cfg['distillation']['momentum_start'],
        final_value=cfg['distillation']['momentum_end'],
        total_steps=total_steps,
    )
    
    # Teacher Temp Schedule
    teacher_temp_schedule = cosine_scheduler(
        base_value=cfg['distillation']['teacher_temp_start'],
        final_value=cfg['distillation']['teacher_temp_end'],
        total_steps=total_steps,
    )
    
    # 5. Loss
    criterion = CoralMonsterLoss(cfg['distillation']) # Config dict passed directly
    
    # 6. Training Loop
    print("Starting training...")
    global_step = 0
    save_dir = Path(cfg['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(cfg['training']['max_epochs']):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            gt_points = batch.get('gt_points')
            if gt_points is not None:
                gt_points = gt_points.to(device)
            
            # Update Schedules
            current_momentum = momentum_schedule[global_step]
            current_teacher_temp = teacher_temp_schedule[global_step]
            
            model.set_momentum(current_momentum)
            
            # Forward
            outputs = model(
                images=images,
                gt_points=gt_points,
                compute_distillation=True # Always True for clean training config
            )
            
            # Loss
            loss_dict = criterion(
                outputs,
                masks,
                current_step=global_step,
                teacher_temp=current_teacher_temp
            )
            
            loss = loss_dict["total_loss"]
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Teacher Update
            model.update_teacher()
            
            # Logging
            if global_step % cfg['training']['log_interval'] == 0:
                print(
                    f"Epoch [{epoch + 1}/{cfg['training']['max_epochs']}] "
                    f"Step [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(MaskKD: {loss_dict.get('mask_kd', 0):.4f}, TokenKD: {loss_dict.get('token_kd', 0):.4f}) "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} "
                    f"Mom: {current_momentum:.4f}"
                )
            
            global_step += 1
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, save_dir / f"checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()

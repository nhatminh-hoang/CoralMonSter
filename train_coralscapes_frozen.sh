#!/bin/bash

# Training script for CoralMonSter on CoralScapes
# Configuration:
# - Checkpoint: checkpoints/vit_b_coralscop.pth
# - Learning Rate: 1e-4
# - Scenario: frozen_encoder (Frozen Image Encoder)
# - Epochs: 60
# - Batch Size: 2
# - Num Workers: 4
# - Prompt Bins: 10
# - Scenario Name: coralscapes_CoralMonSter_freeze_encoder_10points

echo "Starting training..."

python -m CoralMonSter.tools.train \
  --dataset coralscapes \
  --dataset_root data_storage/CoralScapes \
  --sam_checkpoint checkpoints/vit_b_coralscop.pth \
  --scenario_preset frozen_encoder \
  --scenario_name "coralscapes_CoralMonSter_freeze_encoder_10points" \
  --max_epochs 60 \
  --batch_size 1 \
  --num_workers 4 \
  --learning_rate 1e-4 \
  --prompt_bins 10

echo "Training complete."

#!/bin/bash

# Train
python train.py --config configs/hkcoral_vit_b.yaml --gpu 0 --seed 42

# Evaluate on test set
python evaluate.py --config configs/hkcoral_vit_b.yaml --checkpoint logs/hkcoral_vit_b/checkpoints/best.pth --split test --gpu 0

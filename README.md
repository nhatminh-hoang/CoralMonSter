# CoralMonSter: Coral Image Momentum Segmenter

## Abstract

**CoralMonSter** is a premium, prompt-free coral reef segmentation framework built on top of Segment Anything (SAM). It leverages a novel **Teacher-Student Distillation** architecture and **Learnable Class Tokens** to achieve state-of-the-art semantic segmentation without requiring user prompts.

By introducing a **Semantic Query Decoder (SQD)** and a **Momentum Distillation** framework, CoralMonSter effectively bridges the gap between SAM's class-agnostic segmentation and the need for semantic understanding in coral ecology. Our method achieves **57.28% mIoU** on the HKCoral benchmark, significantly outperforming frozen baselines.

## Repository Structure

The code is organized to be clean and publication-ready:

- `configs/`: YAML configuration files for different datasets/models.
- `coralmonster/`: The main Python package containing the model architecture.
  - `arch.py`: Parameter definitions and forward pass logic.
  - `decoder.py`: Semantic Query Decoder implementation.
  - `encoder.py`: Wrapper for SAM backbone.
  - `loss.py`: Unified CoralMonsterLoss (Dice + Distillation).
- `data/`: Dataset loaders for HKCoral and CoralScapes.
- `utils/`: Training utilities (scheduling, EMA).
- `train.py`: Main training script.
- `evaluate.py`: Evaluation script.

## Installation

```bash
pip install -r requirements.txt
# Ensure you have torch and torchvision installed compatible with your CUDA version.
```

## Usage

### Training

To reproduce the **HKCoral SOTA (57.28% mIoU)** results using the recommended specific configuration (Unfrozen Encoder, AdamW, Distillation):

```bash
python train.py --config configs/hkcoral_vit_b.yaml
```

### Evaluation

To evaluate a trained checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/hkcoral_clean/checkpoint_epoch_50.pth --config configs/hkcoral_vit_b.yaml
```

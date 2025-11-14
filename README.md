# CoralMonSter

Teacher–student prompt-free coral segmentation built on SAM.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation

### HKCoral
Place the HKCoral dataset under `datasets/HKCoral/` with the following layout:

```
datasets/HKCoral/
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*_labelTrainIds.png
```

### CoralScapes
CoralScapes is pulled directly from the Hugging Face hub. Run the helper once to warm the cache and download the label metadata:

```bash
python datasets/CoralScapes/setup.py
```

This populates `datasets/CoralScapes/cache/` and stores `id2label` / `label2color` files used for visualization.

## Training

### HKCoral
```bash
python CoralMonSter/train.py \
  --dataset hkcoral \
  --dataset_root datasets/HKCoral \
  --sam_checkpoint checkpoints/vit_b_coralscop.pth \
  --model_type vit_b \
  --batch_size 2 \
  --max_epochs 40
```

### CoralScapes
```bash
python CoralMonSter/train.py \
  --dataset coralscapes \
  --dataset_root datasets/CoralScapes \
  --dataset_cache_dir datasets/CoralScapes/cache \
  --sam_checkpoint checkpoints/vit_b_coralscop.pth \
  --model_type vit_b \
  --batch_size 2 \
  --max_epochs 40
```

Use `--train_subset`, `--val_subset`, or `--test_subset` during debugging to limit the number of samples processed per epoch.

## Outputs
Checkpoints are written to `checkpoints/<scenario_name>/` and logs/visualizations live under `logs/<scenario_name>/`.

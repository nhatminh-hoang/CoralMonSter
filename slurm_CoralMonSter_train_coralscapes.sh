#!/bin/bash

#SBATCH --job-name=CoralMonSter_CoralScapes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --output=train_outs/gpu/out/%x.%j.out
#SBATCH --error=train_outs/gpu/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=21013299@st.phenikaa-uni.edu.vn

conda init bash
source ~/.bashrc
conda activate gpu_11.8

CHECKPOINT="checkpoints/vit_b_coralscop.pth"
DATASET_ROOT="datasets/CoralScapes"
DATASET_CACHE_DIR="${DATASET_ROOT}/cache"
MAX_EPOCHS=60
BATCH_SIZE=2
NUM_WORKERS=4
GPU_DEVICES="0"
LR=1e-4
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"

HF_TOKEN_ARG=()
if [[ -n "${HF_TOKEN}" ]]; then
  HF_TOKEN_ARG+=(--hf_token "${HF_TOKEN}")
fi

# echo "===> Running scenario: no_momentum"
# python -m CoralMonSter.train \
#   --dataset coralscapes \
#   --dataset_root "${DATASET_ROOT}" \
#   --dataset_cache_dir "${DATASET_CACHE_DIR}" \
#   --scenario_name "coralscapes_no_momentum" \
#   --scenario_preset "no_momentum" \
#   --sam_checkpoint "${CHECKPOINT}" \
#   --max_epochs "${MAX_EPOCHS}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --learning_rate "${LR}"

# echo "===> Running scenario: momentum_no_skip"
# python -m CoralMonSter.train \
#   --dataset coralscapes \
#   --dataset_root "${DATASET_ROOT}" \
#   --dataset_cache_dir "${DATASET_CACHE_DIR}" \
#   --scenario_name "coralscapes_momentum_no_skip" \
#   --scenario_preset "momentum_no_skip" \
#   --sam_checkpoint "${CHECKPOINT}" \
#   --max_epochs "${MAX_EPOCHS}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --learning_rate "${LR}"

# echo "===> Running scenario: frozen_encoder"
# python -m CoralMonSter.train \
#   --dataset coralscapes \
#   --dataset_root "${DATASET_ROOT}" \
#   --dataset_cache_dir "${DATASET_CACHE_DIR}" \
#   --scenario_name "coralscapes_frozen_encoder" \
#   --scenario_preset "frozen_encoder" \
#   --sam_checkpoint "${CHECKPOINT}" \
#   --max_epochs "${MAX_EPOCHS}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --learning_rate "${LR}"

# echo "===> Running scenario: no_scheduler"
# python -m CoralMonSter.train \
#   --dataset coralscapes \
#   --dataset_root "${DATASET_ROOT}" \
#   --dataset_cache_dir "${DATASET_CACHE_DIR}" \
#   --scenario_name "coralscapes_no_scheduler" \
#   --scenario_preset "no_scheduler" \
#   --sam_checkpoint "${CHECKPOINT}" \
#   --max_epochs "${MAX_EPOCHS}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --learning_rate "${LR}"

# echo "===> Running scenario: coralscapes_CoralMonSter"
# python -m CoralMonSter.train \
#   --dataset coralscapes \
#   --dataset_root "${DATASET_ROOT}" \
#   --dataset_cache_dir "${DATASET_CACHE_DIR}" \
#   --scenario_name "coralscapes_CoralMonSter" \
#   --scenario_preset "full" \
#   --sam_checkpoint "${CHECKPOINT}" \
#   --max_epochs "${MAX_EPOCHS}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --profile \
#   --learning_rate "${LR}"

# echo "===> Running scenario: CoralMonSter_10points"
# python -m CoralMonSter.train \
#   --dataset coralscapes \
#   --dataset_root "${DATASET_ROOT}" \
#   --dataset_cache_dir "${DATASET_CACHE_DIR}" \
#   --scenario_name "coralscapes_CoralMonSter_10points" \
#   --scenario_preset "full" \
#   --sam_checkpoint "${CHECKPOINT}" \
#   --max_epochs "${MAX_EPOCHS}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --profile \
#   --learning_rate "${LR}" \
#   --prompt_bins 10 

# echo "===> Running scenario: CoralMonSter_4points"
# python -m CoralMonSter.train \
#   --dataset coralscapes \
#   --dataset_root "${DATASET_ROOT}" \
#   --dataset_cache_dir "${DATASET_CACHE_DIR}" \
#   --scenario_name "coralscapes_CoralMonSter_4points" \
#   --scenario_preset "full" \
#   --sam_checkpoint "${CHECKPOINT}" \
#   --max_epochs "${MAX_EPOCHS}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --profile \
#   --learning_rate "${LR}" \
#   --prompt_bins 4

echo "===> Running scenario: CoralMonSter_freeze_4points_OthoInit"
python -m CoralMonSter.train \
  --dataset coralscapes \
  --dataset_root "${DATASET_ROOT}" \
  --dataset_cache_dir "${DATASET_CACHE_DIR}" \
  --scenario_name "coralscapes_CoralMonSter_freeze_4points_OthoInit" \
  --scenario_preset "frozen_encoder" \
  --sam_checkpoint "${CHECKPOINT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --learning_rate "${LR}" \
  --prompt_bins 4

python evaluate_checkpoints.py \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --dataset_cache_dir "${DATASET_CACHE_DIR}" \
  --checkpoint_root checkpoints \
  --output coralscapes_checkpoint_evaluations.json \
  --dataset_keyword coralscapes \

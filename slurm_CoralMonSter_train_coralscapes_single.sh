#!/bin/bash

#SBATCH --job-name=CoralMonSter_CoralScapes_pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx-small
#SBATCH --cpus-per-task=8
#SBATCH --output=train_outs/small/out/%x.%j.out
#SBATCH --error=train_outs/small/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=21013299@st.phenikaa-uni.edu.vn

conda init bash
source ~/.bashrc
conda activate minh_ml

CHECKPOINT="checkpoints/vit_b_coralscop.pth"
DATASET_ROOT="datasets/CoralScapes"
DATASET_CACHE_DIR="${DATASET_ROOT}/cache"
MAX_EPOCHS=60
BATCH_SIZE=2
NUM_WORKERS=2
LR=1e-5
HF_TOKEN_ARG=()
if [[ -n "${HF_TOKEN}" ]]; then
  HF_TOKEN_ARG+=(--hf_token "${HF_TOKEN}")
fi

# echo "===> Running scenario: coralscapes_no_scheduler"
# CUDA_VISIBLE_DEVICES=4 python -m CoralMonSter.train \
#   --dataset coralscapes \
#   --dataset_root "${DATASET_ROOT}" \
#   --dataset_cache_dir "${DATASET_CACHE_DIR}" \
#   --scenario_name "coralscapes_no_scheduler" \
#   --scenario_preset "no_scheduler" \
#   --sam_checkpoint "${CHECKPOINT}" \
#   --max_epochs "${MAX_EPOCHS}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --profile \
#   --learning_rate "${LR}" \
#   "${HF_TOKEN_ARG[@]}" &
# PID_NOSCHED=$!

# echo "===> Running scenario: coralscapes_no_centering"
# CUDA_VISIBLE_DEVICES=5 python -m CoralMonSter.train \
#   --dataset coralscapes \
#   --dataset_root "${DATASET_ROOT}" \
#   --dataset_cache_dir "${DATASET_CACHE_DIR}" \
#   --scenario_name "coralscapes_no_centering" \
#   --scenario_preset "no_centering" \
#   --sam_checkpoint "${CHECKPOINT}" \
#   --max_epochs "${MAX_EPOCHS}" \
#   --batch_size "${BATCH_SIZE}" \
#   --num_workers "${NUM_WORKERS}" \
#   --profile \
#   --learning_rate "${LR}" \
#   "${HF_TOKEN_ARG[@]}" &
# PID_NOCENTER=$!

# echo "===> Running scenario: CoralMonSter"
# CUDA_VISIBLE_DEVICES=6 python -m CoralMonSter.train \
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
#   --learning_rate "${LR}" \
#   "${HF_TOKEN_ARG[@]}" &
# PID_FULL=$!

echo "===> Running scenario: CoralMonSter_4points"
CUDA_VISIBLE_DEVICES=2 python -m CoralMonSter.train \
  --dataset coralscapes \
  --dataset_root "${DATASET_ROOT}" \
  --dataset_cache_dir "${DATASET_CACHE_DIR}" \
  --scenario_name "coralscapes_CoralMonSter_4points" \
  --scenario_preset "full" \
  --sam_checkpoint "${CHECKPOINT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --profile \
  --learning_rate "${LR}" \
  --prompt_bins 4 \
  "${HF_TOKEN_ARG[@]}" &
PID_FULL_4POINTS=$!

# echo "===> Running scenario: CoralMonSter_10points"
# CUDA_VISIBLE_DEVICES=7 python -m CoralMonSter.train \
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
#   --prompt_bins 10 \
#   "${HF_TOKEN_ARG[@]}" &
# PID_FULL_10POINTS=$!

wait $PID_NOSCHED $PID_NOCENTER $PID_FULL $PID_FULL_4POINTS $PID_FULL_10POINTS
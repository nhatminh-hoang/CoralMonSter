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
MAX_EPOCHS=40
BATCH_SIZE=2
NUM_WORKERS=8
GPU_DEVICES="0"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"

HF_TOKEN_ARG=()
if [[ -n "${HF_TOKEN}" ]]; then
  HF_TOKEN_ARG+=(--hf_token "${HF_TOKEN}")
fi

SCENARIOS=(
  "coralscapes_full:full"
  "coralscapes_no_scheduler:no_scheduler"
  "coralscapes_no_centering:no_centering"
  "coralscapes_unfrozen_encoder:unfrozen_encoder"
  "coralscapes_no_momentum:no_momentum"
  "coralscapes_momentum_no_skip:momentum_no_skip"
  "coralscapes_token_kd_kl:token_kd_kl"
)

for entry in "${SCENARIOS[@]}"; do
  IFS=":" read -r SCENARIO_NAME SCENARIO_PRESET <<< "${entry}"
  echo "===> Running scenario: ${SCENARIO_NAME} (${SCENARIO_PRESET})"
  python -m CoralMonSter.train \
    --dataset coralscapes \
    --dataset_root "${DATASET_ROOT}" \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --gpu_devices "${GPU_DEVICES}" \
    --scenario_name "${SCENARIO_NAME}" \
    --scenario_preset "${SCENARIO_PRESET}" \
    --sam_checkpoint "${CHECKPOINT}" \
    --max_epochs "${MAX_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    "${HF_TOKEN_ARG[@]}"

done

python evaluate_checkpoints.py \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --checkpoint_root checkpoints

#!/bin/bash

#SBATCH --job-name=SSP_train_encoder_decoder_ADReSS2020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --nodelist=hpc24
#SBATCH --cpus-per-task=32
#SBATCH --output=train_outs/gpu/out/%x.%j.out
#SBATCH --error=train_outs/gpu/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=21013299@st.phenikaa-uni.edu.vn

conda init bash
source ~/.bashrc
conda activate gpu_11.8

CHECKPOINT="checkpoints/vit_b_coralscop.pth"
DATASET_ROOT="datasets/HKCoral"
MAX_EPOCHS=40
BATCH_SIZE=2
GPU_DEVICES="0"
export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}

echo "===> Running scenario: unfrozen_encoder"
python -m CoralMonSter.train \
  --gpu_devices "${GPU_DEVICES}" \
  --scenario_name "scen_unfrozen_encoder" \
  --scenario_preset "unfrozen_encoder" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}"

echo "===> Running scenario: no_momentum"
python -m CoralMonSter.train \
  --gpu_devices "${GPU_DEVICES}" \
  --scenario_name "scen_no_momentum" \
  --scenario_preset "no_momentum" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}"

echo "===> Running scenario: momentum_no_skip"
python -m CoralMonSter.train \
  --gpu_devices "${GPU_DEVICES}" \
  --scenario_name "scen_momentum_no_skip" \
  --scenario_preset "momentum_no_skip" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}"

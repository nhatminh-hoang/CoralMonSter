#!/bin/bash

#SBATCH --job-name=SSP_train_encoder_decoder_ADReSS2020
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
DATASET_ROOT="datasets/HKCoral"
MAX_EPOCHS=40
BATCH_SIZE=8

echo "===> Running scenario: full"
CUDA_VISIBLE_DEVICES=0 python -m CoralMonSter.train \
  --scenario_name "scen_full" \
  --scenario_preset "full" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" &
PID_FULL=$!

echo "===> Running scenario: no_scheduler"
CUDA_VISIBLE_DEVICES=1 python -m CoralMonSter.train \
  --scenario_name "scen_no_scheduler" \
  --scenario_preset "no_scheduler" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" &
PID_NOSCHED=$!

echo "===> Running scenario: no_centering"
CUDA_VISIBLE_DEVICES=2 python -m CoralMonSter.train \
  --scenario_name "scen_no_centering" \
  --scenario_preset "no_centering" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" &
PID_NOCENTER=$!

echo "===> Running scenario: unfrozen_encoder"
CUDA_VISIBLE_DEVICES=3 python -m CoralMonSter.train \
  --scenario_name "scen_unfrozen_encoder" \
  --scenario_preset "unfrozen_encoder" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" &
PID_UNFROZEN=$!

wait $PID_FULL $PID_NOSCHED $PID_NOCENTER $PID_UNFROZEN

echo "===> Running scenario: no_momentum"
CUDA_VISIBLE_DEVICES=0 python -m CoralMonSter.train \
  --scenario_name "scen_no_momentum" \
  --scenario_preset "no_momentum" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}"

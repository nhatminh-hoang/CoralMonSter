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
NUM_GPUS=3
GPU_DEVICES="5,6,7"
export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}

echo "===> Running scenario: full"
torchrun --nproc_per_node=${NUM_GPUS} --master_port=12355 python -m CoralMonSter.train \
  --distributed \
  --gpu_devices "${GPU_DEVICES}" \
  --scenario_name "scen_full" \
  --scenario_preset "full" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}"

echo "===> Running scenario: no_scheduler"
torchrun --nproc_per_node=${NUM_GPUS} --master_port=12356 python -m CoralMonSter.train \
  --distributed \
  --gpu_devices "${GPU_DEVICES}" \
  --scenario_name "scen_no_scheduler" \
  --scenario_preset "no_scheduler" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}"

echo "===> Running scenario: no_centering"
torchrun --nproc_per_node=${NUM_GPUS} --master_port=12357 python -m CoralMonSter.train \
  --distributed \
  --gpu_devices "${GPU_DEVICES}" \
  --scenario_name "scen_no_centering" \
  --scenario_preset "no_centering" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}"

echo "===> Running scenario: unfrozen_encoder"
torchrun --nproc_per_node=${NUM_GPUS} --master_port=12358 python -m CoralMonSter.train \
  --distributed \
  --gpu_devices "${GPU_DEVICES}" \
  --scenario_name "scen_unfrozen_encoder" \
  --scenario_preset "unfrozen_encoder" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}"

echo "===> Running scenario: no_momentum"
torchrun --nproc_per_node=${NUM_GPUS} --master_port=12359 python -m CoralMonSter.train \
  --distributed \
  --gpu_devices "${GPU_DEVICES}" \
  --scenario_name "scen_no_momentum" \
  --scenario_preset "no_momentum" \
  --sam_checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}"

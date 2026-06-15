#!/usr/bin/env bash
set -euo pipefail

GRIB_PATH=data/era5_2019.grib
MODEL_TYPE=prior_permutation # or mt_icl
NIC=0
BATCH_SIZE=64
LEARNING_RATE=5e-4 # tune this based on the val performance
SEED=0
SPLIT_STRATEGY=iid # or ood
MAX_EPOCHS=1000
LOGGER=wandb
WANDB_PROJECT=era5-multitask-icl

RUN_NAME="${MODEL_TYPE}_seed${SEED}_${SPLIT_STRATEGY}_k=${NIC}_lr${LEARNING_RATE}"
OUTPUT_DIR="outputs/${RUN_NAME}"

echo "Starting ${RUN_NAME}"

python train_era5_multitask.py \
  --grib_path "${GRIB_PATH}" \
  --nic "${NIC}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --model "${MODEL_TYPE}" \
  --seed "${SEED}" \
  --learning_rate "${LEARNING_RATE}" \
  --split_strategy "${SPLIT_STRATEGY}" \
  --logger "${LOGGER}" \
  --max_epochs "${MAX_EPOCHS}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${RUN_NAME}"

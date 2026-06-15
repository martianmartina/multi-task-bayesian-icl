#!/usr/bin/env bash
set -euo pipefail

TEST_GRIB_PATH=data/era5_2019-2020.grib
STATS_GRIB_PATH=data/era5_2019.grib
TIME_START=2020-01-01T00:00:00
TIME_END=2020-12-31T23:59:59
NUM_TEST_SAMPLES=2000
SEED=0
BATCH_SIZE=64
NUM_WORKERS=8
DEVICES=1
PRECISION=16-mixed
EVAL_OUTPUT_NAME=test_eval_2020only_2019stats_2000_summary.json

# modify the RUN_DIRS array to include your actual checkpoints
RUN_DIRS=(
  outputs/old_outperform_icicl_version_prior_permutation_seed0_iid_val_lr5e-4_noise0.0
  outputs/old_outperform_icicl_version_prior_permutation_seed0_iid_val_k=0_lr5e-4_noise0.0
  outputs/old_outperform_icicl_version_mt_icl_seed0_iid_val_lr1e-3_noise0.0
  outputs/old_outperform_icicl_version_mt_icl_seed0_iid_val_k=0_lr1e-3_noise0.0
)

for run_dir in "${RUN_DIRS[@]}"; do
  echo "Evaluating ${run_dir}/checkpoints/best.ckpt"
  python eval_era5_multitask.py \
    --checkpoint "${run_dir}/checkpoints/best.ckpt" \
    --eval_output "${run_dir}/${EVAL_OUTPUT_NAME}" \
    --test_grib_path "${TEST_GRIB_PATH}" \
    --stats_grib_path "${STATS_GRIB_PATH}" \
    --time_start "${TIME_START}" \
    --time_end "${TIME_END}" \
    --num_test_samples "${NUM_TEST_SAMPLES}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}"
done
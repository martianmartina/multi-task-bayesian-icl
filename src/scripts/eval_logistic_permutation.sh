#!/usr/bin/env bash
set -euo pipefail

PRIOR_MEAN=0
NUM_PRIOR_TASKS=20
SEQUENCE_LENGTH=50
NUM_QUERY_POINTS=100
NUM_TEST_DRAWS=60
SEED=0

OUTPUT_ROOT=results/logistic_results_normal_query
MODEL_NAME=gpt_best_config_d128h8B4096_normal_prior
CHECKPOINT_FILENAME=best-model-epoch=96-val_loss=0.22.ckpt

PERMUTE_TRIALS=10

for CONTEXT_LEN_FOR_TARGET in 20; do
  python src/eval_logistic.py \
    --mode eval_neural \
    --prior-dist normal \
    --prior-mean "${PRIOR_MEAN}" \
    --prior-scale 1.0 \
    --transform-name identity \
    --output-root "${OUTPUT_ROOT}" \
    --num-prior-tasks "${NUM_PRIOR_TASKS}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --baseline-context-len "${CONTEXT_LEN_FOR_TARGET}" \
    --num-query-points "${NUM_QUERY_POINTS}" \
    --num-test-draws "${NUM_TEST_DRAWS}" \
    --seed "${SEED}" \
    --model-name "${MODEL_NAME}" \
    --checkpoint-filename "${CHECKPOINT_FILENAME}" \
    --permute-prior-datasets \
    --permute-prior-points \
    --permute-trials "${PERMUTE_TRIALS}"
done

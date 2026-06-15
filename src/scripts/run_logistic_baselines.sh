#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <draw_start> <draw_end>"
  exit 1
fi

DRAW_START="$1"
DRAW_END="$2"

NUM_DRAWS="${NUM_DRAWS:-60}"
PRIOR_MEAN="${PRIOR_MEAN:-0}"
NUM_PRIOR_TASKS="${NUM_PRIOR_TASKS:-30}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-50}"
BASELINE_CONTEXT_LEN="${BASELINE_CONTEXT_LEN:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/logistic_results_normal_query_K30}"

BASELINE_TYPES=(
  "mcmc"
  "mcmc_hier"
  "svi"
  "svi_hier"
)

for BASELINE_TYPE in "${BASELINE_TYPES[@]}"; do
  echo "Running logistic baseline: ${BASELINE_TYPE} draws=[${DRAW_START},${DRAW_END})"
  python -u src/eval_logistic.py \
    --mode eval_baseline \
    --device cpu \
    --draw-start "${DRAW_START}" \
    --draw-end "${DRAW_END}" \
    --num-test-draws "${NUM_DRAWS}" \
    --baseline-types "${BASELINE_TYPE}" \
    --prior-dist normal \
    --prior-mean "${PRIOR_MEAN}" \
    --prior-scale 1.0 \
    --num-prior-tasks "${NUM_PRIOR_TASKS}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --baseline-context-len "${BASELINE_CONTEXT_LEN}" \
    --transform-name identity \
    --seed 0 \
    --output-root "${OUTPUT_ROOT}"
done

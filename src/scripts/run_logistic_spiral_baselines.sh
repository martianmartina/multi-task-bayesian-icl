#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <draw_start> <draw_end>"
  exit 1
fi

DRAW_START="$1"
DRAW_END="$2"

NUM_DRAWS="${NUM_DRAWS:-60}"
PRIOR_MEAN="${PRIOR_MEAN:-8}"
CONTEXT_LEN="${CONTEXT_LEN:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
SUFFIX="${SUFFIX:-warmup10_final}"
OUTPUT_ROOT="results/logistic_results_spiral_flow_${SUFFIX}"

BASELINE_TYPES=(
  "mcmc_hier_spiral"
  "svi_hier_spiral"
)

for BASELINE_TYPE in "${BASELINE_TYPES[@]}"; do
  if [[ "${BASELINE_TYPE}" == "mcmc_hier_spiral" ]]; then
    K_LIST="1 2 3 4 5 6 7 8 9 10 20 30 40 50 80 100 200 300 500 800 1000"
  elif [[ "${BASELINE_TYPE}" == "svi_hier_spiral" ]]; then
    K_LIST="1 2 3 4 5 6 7 8 9 10 20 30 40 50 80 100 200 300 500 800 1000 1500 2000 2500 3000"
  else
    echo "Unknown BASELINE_TYPE=${BASELINE_TYPE}"
    exit 1
  fi

  echo "Running spiral logistic baseline: ${BASELINE_TYPE} draws=[${DRAW_START},${DRAW_END})"
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
    --transform-name spiral_flow \
    --baseline-context-len "${CONTEXT_LEN}" \
    --k-list ${K_LIST} \
    --mcmc-warmup-steps "${WARMUP_STEPS}" \
    --mcmc-thinning 1 \
    --output-root "${OUTPUT_ROOT}"
done

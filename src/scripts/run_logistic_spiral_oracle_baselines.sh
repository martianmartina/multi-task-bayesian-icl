#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <draw_start> <draw_end>"
  exit 1
fi

DRAW_START="$1"
DRAW_END="$2"

NUM_DRAWS="${NUM_DRAWS:-60}"
PRIOR_MEAN="${PRIOR_MEAN:-8.0}"
BASELINE_CONTEXT_LEN="${BASELINE_CONTEXT_LEN:-50}"
SUFFIX="${SUFFIX:-warmup10_final}"
OUTPUT_ROOT="results/logistic_results_spiral_flow_${SUFFIX}"

BASELINE_TYPES=(
  "mcmc_spiral_oracle"
)

for BASELINE_TYPE in "${BASELINE_TYPES[@]}"; do
  echo "Running spiral oracle baseline: ${BASELINE_TYPE} draws=[${DRAW_START},${DRAW_END})"
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
    --output-root "${OUTPUT_ROOT}" \
    --baseline-context-len "${BASELINE_CONTEXT_LEN}"
done

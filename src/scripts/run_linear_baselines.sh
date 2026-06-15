#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <draw_start> <draw_end>"
  exit 1
fi

DRAW_START="$1"
DRAW_END="$2"

BASELINE_TYPES=(
  "mcmc_hier"
  "svi_hier"
)

for BASELINE_TYPE in "${BASELINE_TYPES[@]}"; do
  echo "Running linear baseline: ${BASELINE_TYPE} draws=[${DRAW_START},${DRAW_END})"
  python -u src/eval_linear.py \
    --mode eval_baseline \
    --device cpu \
    --draw-start "${DRAW_START}" \
    --draw-end "${DRAW_END}" \
    --baseline-types "${BASELINE_TYPE}" \
    --seed 0
done

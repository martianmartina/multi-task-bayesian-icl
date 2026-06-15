NUM_PRIOR_TASKS=20
SEQUENCE_LENGTH=50
CONTEXT_LEN_FOR_TARGET=20
OUTPUT_ROOT=results/logistic_results_normal_query

PRIOR_MEAN_RANGE=(0 4 8 10)

for PRIOR_MEAN in ${PRIOR_MEAN_RANGE[@]}; do
  # step 1: generate data
  python src/eval_logistic.py \
  --mode generate_data \
  --prior-dist normal \
  --prior-mean ${PRIOR_MEAN} \
  --prior-scale 1.0 \
  --output-root ${OUTPUT_ROOT} \
  --transform-name identity \
  --num-prior-tasks ${NUM_PRIOR_TASKS} \
  --sequence-length ${SEQUENCE_LENGTH} \
  --baseline-context-len ${CONTEXT_LEN_FOR_TARGET} \
  --num-query-points 100 \
  --num-test-draws 60 

  # step 2: evaluate neural model
  python src/eval_logistic.py \
  --mode eval_neural \
  --prior-dist normal \
  --prior-mean ${PRIOR_MEAN} \
  --prior-scale 1.0 \
  --num-prior-tasks ${NUM_PRIOR_TASKS} \
  --baseline-context-len ${CONTEXT_LEN_FOR_TARGET} \
  --output-root ${OUTPUT_ROOT} \
  --transform-name identity \
  --model-name gpt_best_config_d128h8B4096_normal_prior \
  --checkpoint-filename best-model-epoch=96-val_loss=0.22.ckpt \
  --neural-key "with prefix" \
  --model-name2 gpt_best_config_d128h8B4096_normal_prior_no_prior \
  --checkpoint-filename2 best-model-epoch=76-val_loss=0.25.ckpt \
  --neural-key2 "no prefix" \
  --neural2-no-prior-tasks
done


# step 3: evaluate baselines (time consuming on cpu, so inside sbatch scripts)


for PRIOR_MEAN in ${PRIOR_MEAN_RANGE[@]}; do
  # step 4: merge baselines
  for BASELINE_TYPE in mcmc mcmc_hier svi svi_hier; do
    python src/eval_logistic.py --mode merge_baseline_shards \
        --output-root $OUTPUT_ROOT \
        --prior-dist normal --transform-name identity --prior-mean ${PRIOR_MEAN} \
        --num-test-draws 60 --num-query-points 100 --seed 0 \
        --baseline-context-len $CONTEXT_LEN_FOR_TARGET \
        --baseline-types ${BASELINE_TYPE}
  done

  # step 5: merge neural and baselines
  python src/eval_logistic.py --mode merge \
    --output-root ${OUTPUT_ROOT} \
    --prior-dist normal --transform-name identity --prior-mean ${PRIOR_MEAN} \
    --num-test-draws 60 --num-query-points 100 --seed 0 \
    --baseline-types mcmc mcmc_hier svi svi_hier \
    --model-name gpt_best_config_d128h8B4096_normal_prior \
    --model-name2 gpt_best_config_d128h8B4096_normal_prior_no_prior \
    --baseline-context-len $CONTEXT_LEN_FOR_TARGET \
    --neural-key "with prefix" \
    --neural-key2 "no prefix" \

  # step 6: report results as tables (rebuttal)
  RESULTS_PATH=${OUTPUT_ROOT}/normal_prior/wmean${PRIOR_MEAN}_ctx${CONTEXT_LEN_FOR_TARGET}_60draws_100queries_seed0/final_results_ctx${CONTEXT_LEN_FOR_TARGET}.pt
  for METRIC in kl tv; do
    python src/plot_logistic.py \
      --results-path $RESULTS_PATH \
      --oracle mcmc --metric ${METRIC} \
      --methods mcmc mcmc_hier svi svi_hier\
      --neural-keys "with prefix" "no prefix" \
      --neural-band \
      --skip-plot \
      --table-only-k 1000
  done
done
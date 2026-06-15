SUFFIX="warmup10_final"
NUM_PRIOR_TASKS=20
SEQUENCE_LENGTH=50
CONTEXT_LEN=50

PRIOR_MEAN=12

python src/eval_logistic.py --mode generate_data \
--output-root results/logistic_results_spiral_flow_${SUFFIX} \
--prior-dist normal --transform-name spiral_flow --transform-seed 1_500_000_000 \
--prior-mean ${PRIOR_MEAN} \
--num-test-draws 60 --num-query-points 100 --seed 0 \
--k-list 1 2 3 4 5 6 7 8 9 10 20 30 40 50 80 100 200 300 500 800 1000 \
--num-prior-tasks ${NUM_PRIOR_TASKS} \
--sequence-length ${SEQUENCE_LENGTH} \
--baseline-context-len ${CONTEXT_LEN} \
--mcmc-thinning 1

python src/eval_logistic.py \
--mode eval_neural \
--prior-dist normal \
--transform-name spiral_flow \
--transform-seed 1_500_000_000 \
--prior-mean ${PRIOR_MEAN} \
--prior-scale 1.0 \
--baseline-context-len ${CONTEXT_LEN} \
--output-root results/logistic_results_spiral_flow_${SUFFIX} \
--model-name gpt_best_config_large_spiral_flow_per_batch \
--checkpoint-filename best-model-epoch=75-val_loss=0.38.ckpt

# # run eval_baseline for each baseline type on cpu

python src/eval_logistic.py --mode merge_baseline_shards \
  --output-root results/logistic_results_spiral_flow_${SUFFIX} \
  --prior-dist normal --transform-name spiral_flow --transform-seed 1_500_000_000 \
  --prior-mean ${PRIOR_MEAN} \
  --num-test-draws 60 --num-query-points 100 --seed 0 \
  --baseline-types mcmc_spiral_oracle \
  --baseline-context-len ${CONTEXT_LEN}

python src/eval_logistic.py --mode merge_baseline_shards \
    --output-root results/logistic_results_spiral_flow_${SUFFIX} \
    --prior-dist normal --transform-name spiral_flow --transform-seed 1_500_000_000 \
    --prior-mean ${PRIOR_MEAN} \
    --num-test-draws 60 --num-query-points 100 --seed 0 \
    --baseline-types svi_hier_spiral \
    --baseline-context-len ${CONTEXT_LEN}

python src/eval_logistic.py --mode merge_baseline_shards \
    --output-root results/logistic_results_spiral_flow_${SUFFIX} \
    --prior-dist normal --transform-name spiral_flow --transform-seed 1_500_000_000 \
    --prior-mean ${PRIOR_MEAN} \
    --num-test-draws 60 --num-query-points 100 --seed 0 \
    --baseline-types mcmc_hier_spiral \
    --baseline-context-len ${CONTEXT_LEN}

python src/eval_logistic.py --mode merge \
  --output-root results/logistic_results_spiral_flow_${SUFFIX} \
  --prior-dist normal --transform-name spiral_flow --transform-seed 1_500_000_000 \
  --prior-mean ${PRIOR_MEAN} \
  --num-test-draws 60 --num-query-points 100 --seed 0 \
  --baseline-types mcmc_spiral_oracle mcmc_hier_spiral svi_hier_spiral \
  --model-name gpt_best_config_large_spiral_flow_per_batch \
  --mcmc-thinning 1 \
  --baseline-context-len ${CONTEXT_LEN}

python src/plot_logistic.py \
  --results-path results/logistic_results_spiral_flow_${SUFFIX}/spiral_flow_seed1500000000_normal_prior/wmean${PRIOR_MEAN}_ctx50_60draws_100queries_seed0/final_results_ctx50.pt \
  --oracle mcmc_spiral_oracle --metric kl \
  --methods mcmc_hier_spiral svi_hier_spiral \
  --neural-keys "neural"

python src/plot_logistic.py \
  --results-path results/logistic_results_spiral_flow_${SUFFIX}/spiral_flow_seed1500000000_normal_prior/wmean${PRIOR_MEAN}_ctx50_60draws_100queries_seed0/final_results_ctx50.pt \
  --oracle mcmc_spiral_oracle --metric tv \
  --methods mcmc_hier_spiral svi_hier_spiral \
  --neural-keys "neural"
python src/eval_linear.py \
--mode generate_data \
--output-root results/linear_results_new \
--model-name multi_task_implicit_learner_large_-8to8_training_lr \
--checkpoint-filename best-model-t0-epoch=52-val_loss=0.2283.ckpt \
--num-prior-tasks 20 \
--prior-task-len 50 \
--permute-prior-datasets \
--permute-prior-points \
--permute-trials 10


python src/eval_linear.py \
--mode eval_neural \
--output-root results/linear_results_new \
--model-name multi_task_implicit_learner_large_-8to8_training_lr \
--checkpoint-filename best-model-t0-epoch=52-val_loss=0.2283.ckpt \
--num-prior-tasks 20 \
--prior-task-len 50 \
--permute-prior-datasets \
--permute-prior-points \
--permute-trials 10

python src/eval_linear.py \
--mode merge_baseline_shards \
--output-root results/linear_results_new \
--baseline-types mcmc_hier \
--num-prior-tasks 20 \
--prior-task-len 50 \
--baseline-context-lens 5 10 20 30 50 


python src/eval_linear.py \
--mode merge \
--output-root results/linear_results_new \
--num-prior-tasks 20 \
--prior-task-len 50 \
--baseline-context-lens 5 10 20 30 50 \
--neural-results "with prefix=results/linear_results_new/60draws_100queries_noise0.5_seed0/multi_task_implicit_learner_large_-8to8_training_lr_results.pt" "no prefix=results/linear_results_new/60draws_100queries_noise0.5_seed0/multi_task_implicit_learner_large_linear_-8to8_noprefix_long_results.pt" \
--baseline-types mcmc_hier



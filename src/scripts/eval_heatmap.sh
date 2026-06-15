python src/heatmap/eval_heatmap.py \
--config configs/logistic/student_t_ood_heatmap/eval_extreme_var_mean.yaml \
--mode neural \
--draws_path results/logistic_results/student_t_ood_heatmap_extreme_var_mean/draws.pt \
--device cuda \
--df_shard_id 0 --df_num_shards 1 \
--out results/logistic_results/student_t_ood_heatmap_extreme_var_mean/shards/neural_0of1.pt

python src/heatmap/eval_heatmap_merge.py \
--config configs/logistic/student_t_ood_heatmap/eval_extreme_var_mean.yaml \
--neural_glob "results/logistic_results/student_t_ood_heatmap_extreme_var_mean/shards/neural_*.pt" \
--baseline_glob "results/logistic_results/student_t_ood_heatmap_extreme_var_mean/shards/baseline_*.pt" \
--out results/logistic_results/student_t_ood_heatmap_extreme_var_mean/final_merged.pt


python src/heatmap/eval_heatmap_plot.py \
--in_path results/logistic_results/student_t_ood_heatmap_extreme_var_mean/final_merged.pt \
--out_dir results/logistic_results/student_t_ood_heatmap_extreme_var_mean/plots
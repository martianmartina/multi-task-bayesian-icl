# eval_logistic.py
#!/usr/bin/env python
import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple
import random

import torch
import numpy as np
from tqdm import tqdm

# Project-specific Imports
from src.models.multi_task_implicit_model import MultiTaskImplicitInContextLearner
from data.logistic_helpers import *
from utils.eval_helpers import (
    sample_normal_vec,
    sample_laplace_vec,
    sample_student_t_vec,
    generate_logistic_task,
    bernoulli_kl,
    bernoulli_tv,
    run_oracle_mcmc_predictive,
    run_oracle_mcmc_predictive_spiral_known_theta,
    predict_with_bootstrap,
    _mcmc_predict_proba_xquery,
    run_oracle_hierarchical_mcmc_predictive,
    run_oracle_hierarchical_mcmc_predictive_spiral,
    run_svi_curve,
    run_hier_svi_curve,
    run_hier_svi_curve_spiral,
)
import pyro

# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Modular Evaluation Script")

    # Mode Control
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["generate_data", "eval_neural", "eval_baseline", "merge_baseline_shards", "merge"],
        help="Step to run: data generation, neural eval, baseline eval, or merging results."
    )
    parser.add_argument(
        "--baseline-types",
        nargs="+",
        default=["mcmc", "mcmc_hier", "svi", "svi_hier", "mcmc_hier_oracle", "mcmc_spiral_oracle","mcmc_hier_spiral", "svi_hier_spiral"],
        help="Specific baselines to run in eval_baseline mode."
    )

    # Prior settings
    parser.add_argument("--prior-dist", type=str, default="normal", choices=["normal", "laplace", "student_t"])
    parser.add_argument("--transform-name", type=str, default="spiral_flow")
    parser.add_argument("--transform-seed", type=int, default=1_500_000_000)
    parser.add_argument("--student-t-df", type=float, default=1.0)
    parser.add_argument("--x-dim", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--prior-mean", type=float, default=4.0)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--x-mean", type=float, default=0.0)
    parser.add_argument("--x-std", type=float, default=1.0)
    
    # Data settings
    parser.add_argument("--num-prior-tasks", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--baseline-context-len", type=int, default=50)
    parser.add_argument("--num-query-points", type=int, default=100)
    parser.add_argument("--num-test-draws", type=int, default=60)

    # Paths
    parser.add_argument("--output-root", type=str, default="results/logistic_results_spiral_flow")
    
    # Neural model settings
    parser.add_argument("--model-name", type=str, default="gpt_best_config_large_spiral_flow_per_batch")
    parser.add_argument("--checkpoint-filename", type=str, default="best-model-epoch=75-val_loss=0.38.ckpt")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--num-bootstraps", type=int, default=1)
    parser.add_argument("--neural-key", type=str, default="neural",
                        help="Key used to store neural outputs inside the saved results (default: neural).")
    parser.add_argument("--neural-no-prior-tasks", action="store_true", default=False,
                        help="If set, neural eval does NOT pass prior_tasks_data (ablation).")
    parser.add_argument(
        "--permute-prior-datasets",
        action="store_true",
        help="For eval_neural: run ablation over random permutations of prior dataset order.",
    )
    parser.add_argument(
        "--permute-prior-points",
        action="store_true",
        help="For eval_neural: run ablation over random permutations of points within each prior dataset.",
    )
    parser.add_argument(
        "--permute-trials",
        type=int,
        default=0,
        help="For eval_neural permutation ablations: number of random permutations per example.",
    )
    parser.add_argument(
        "--permute-seed",
        type=int,
        default=None,
        help="Optional base seed for permutation ablations. Defaults to --seed when omitted.",
    )
    parser.add_argument(
        "--resample-prior-prefixes",
        action="store_true",
        help=(
            "For eval_neural: for each fixed target context, resample N fresh prior "
            "prefixes from the same prior distribution and summarize KL-to-oracle and "
            "pairwise symmetric KL across the resulting neural PPDs."
        ),
    )
    parser.add_argument(
        "--num-prefix-resamples",
        type=int,
        default=0,
        help="Number of fresh prior prefixes to sample per fixed target context.",
    )
    parser.add_argument(
        "--prefix-resample-seed",
        type=int,
        default=None,
        help="Optional base seed for prior-prefix resampling. Defaults to --seed.",
    )

    # Optional second neural model
    parser.add_argument("--model-name2", type=str, default=None,
                        help="If set, evaluate a second neural model in eval_neural and merge it in merge.")
    parser.add_argument("--checkpoint-filename2", type=str, default=None,
                        help="Checkpoint filename for the second model (used if --checkpoint-path2 is not set).")
    parser.add_argument("--checkpoint-path2", type=str, default=None,
                        help="Full checkpoint path for the second model. If set, overrides --checkpoint-filename2.")
    parser.add_argument("--neural-key2", type=str, default="neural2",
                        help="Key used to store second neural outputs (default: neural2).")
    parser.add_argument("--neural2-no-prior-tasks", action="store_true", default=False,
                        help="If set, second neural eval does NOT pass prior_tasks_data (ablation).")

    # Baseline settings
    parser.add_argument("--k-list", type=int, nargs="+", default=[50, 100, 150, 200, 300, 500, 800, 1000])
    parser.add_argument("--mcmc-oracle-num-samples", type=int, default=10_000)
    parser.add_argument("--svi-posterior-samples", type=int, default=200)
    parser.add_argument("--hier-prior-mean-min", type=float, default=-8.0)
    parser.add_argument("--hier-prior-mean-max", type=float, default=8.0)
    parser.add_argument("--mcmc-warmup-steps", type=int, default=1000)
    parser.add_argument("--mcmc-thinning", type=int, default=10)
    parser.add_argument("--mcmc-num-chains", type=int, default=1)
    
    parser.add_argument("--draw-start", type=int, default=0)
    parser.add_argument("--draw-end", type=int, default=None)  # exclusive

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()

# -------------------------
# Helpers
# -------------------------

def get_save_paths(args):
    transform_prefix = f"{args.transform_name}_seed{args.transform_seed}_" if args.transform_name != "identity" else ""
    if args.prior_dist == "student_t":
        base = os.path.join(args.output_root, f"{transform_prefix}{args.prior_dist}_prior_df{args.student_t_df}")
    else:
        base = os.path.join(args.output_root, f"{transform_prefix}{args.prior_dist}_prior")
    
    tag = (
        f"wmean{args.prior_mean:.0f}_ctx{int(args.baseline_context_len)}_"
        f"{args.num_test_draws}draws_{args.num_query_points}queries_seed{args.seed}"
    )
    dir_path = os.path.join(base, tag)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def sample_weight_vec(args):
    """Unified weight sampling based on prior-dist argument."""
    if args.prior_dist == "student_t":
        return sample_student_t_vec(args.student_t_df, args.prior_mean, args.prior_scale, args.x_dim)
    elif args.prior_dist == "laplace":
        return sample_laplace_vec(args.prior_mean, args.prior_scale, args.x_dim)
    elif args.prior_dist == "normal":
        return sample_normal_vec(args.prior_mean, args.prior_scale, args.x_dim)
    else:
        raise ValueError(f"Unknown prior distribution: {args.prior_dist}")

def apply_transforms(w: torch.Tensor, transform: Optional[torch.nn.Module]) -> torch.Tensor:
    # w: [D, 1]
    if transform is None:
        return w
    w2 = w.squeeze(1).unsqueeze(0)   # [1, D]
    w2 = transform(w2)               # [1, D]
    return w2.squeeze(0).unsqueeze(1)  # [D, 1]

def draw_seed(base_seed: int, draw_idx: int) -> int:
    return int(base_seed + draw_idx) 


def _permute_prior_task_order(
    prior_tasks: List[Tuple[torch.Tensor, torch.Tensor]],
    rng: np.random.Generator,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if len(prior_tasks) <= 1:
        return list(prior_tasks)
    perm = rng.permutation(len(prior_tasks))
    return [prior_tasks[int(i)] for i in perm]


def _permute_points_within_prior_tasks(
    prior_tasks: List[Tuple[torch.Tensor, torch.Tensor]],
    rng: np.random.Generator,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for xp, yp in prior_tasks:
        n = int(xp.shape[0])
        if n <= 1:
            out.append((xp, yp))
            continue
        idx = torch.as_tensor(rng.permutation(n), device=xp.device, dtype=torch.long)
        out.append((xp.index_select(0, idx), yp.index_select(0, idx)))
    return out


def _perm_trial_seed(base_seed: int, global_draw_id: int, trial_idx: int, kind_offset: int) -> int:
    # Deterministic seed per (example, trial, ablation kind) for reproducibility.
    return int(base_seed + 1_000_003 * global_draw_id + 10_007 * trial_idx + kind_offset)

def _sample_fresh_prior_prefix_for_draw(
    args,
    *,
    draw: Dict[str, Any],
    resample_idx: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    seed_i = int(draw["transform_seed"])
    g = torch.Generator(device="cpu").manual_seed(seed_i)
    transform = make_transforms(args.transform_name, args.x_dim, generator=g)
    if transform is not None:
        transform.eval()

    # independent RNG state per resample
    base = int(args.seed if args.prefix_resample_seed is None else args.prefix_resample_seed)
    local_seed = int((base + 10_000_019 * resample_idx + 1_000_003 * seed_i) % (2**32))
    torch.manual_seed(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    pyro.set_rng_seed(local_seed)

    prior_tasks = []
    for _ in range(args.num_prior_tasks):
        w_p = sample_weight_vec(args)
        with torch.no_grad():
            w_p = apply_transforms(w_p, transform)
        x_p, y_p = generate_logistic_task(
            w_p,
            args.sequence_length,
            args.noise_std,
            args.x_mean,
            args.x_std,
        )
        prior_tasks.append((x_p, y_p))
    return prior_tasks

def _summarize_prediction_set_against_oracle(
    trial_preds: List[Dict[str, Any]],
    oracle_probs: np.ndarray,
) -> Dict[str, float]:
    """
    For one fixed target context, summarize variability across multiple neural PPDs
    obtained from different sampled prior prefixes.

    Returns:
      - oracle_kl_mean / oracle_kl_std:
          KL(Bern(oracle_probs) || Bern(pred_probs)) across resampled prefixes
      - pairwise_sym_kl_mean / pairwise_sym_kl_std:
          average pairwise symmetric KL among the neural Bernoulli PPDs
    """
    out: Dict[str, float] = {"num_trials": int(len(trial_preds))}
    if len(trial_preds) == 0:
        out["oracle_kl_mean"] = float("nan")
        out["oracle_kl_std"] = float("nan")
        out["pairwise_sym_kl_mean"] = float("nan")
        out["pairwise_sym_kl_std"] = float("nan")
        return out

    oracle = torch.as_tensor(oracle_probs, dtype=torch.float32).view(-1, 1)

    oracle_kls: List[float] = []
    q_tensors: List[torch.Tensor] = []

    for pred in trial_preds:
        q = torch.as_tensor(pred["q_probs"], dtype=torch.float32).view(-1, 1)
        q_tensors.append(q)
        oracle_kls.append(float(bernoulli_kl(oracle, q)))

    oracle_kls_np = np.asarray(oracle_kls, dtype=np.float64)
    out["oracle_kl_mean"] = float(oracle_kls_np.mean())
    out["oracle_kl_std"] = (
        float(oracle_kls_np.std(ddof=1)) if len(oracle_kls_np) > 1 else float("nan")
    )

    pairwise_sym_kls: List[float] = []
    T = len(q_tensors)
    for i in range(T):
        for j in range(i + 1, T):
            kl_ij = float(bernoulli_kl(q_tensors[i], q_tensors[j]))
            kl_ji = float(bernoulli_kl(q_tensors[j], q_tensors[i]))
            pairwise_sym_kls.append(0.5 * (kl_ij + kl_ji))

    if len(pairwise_sym_kls) == 0:
        out["pairwise_sym_kl_mean"] = float("nan")
        out["pairwise_sym_kl_std"] = float("nan")
    else:
        pairwise_sym_kls_np = np.asarray(pairwise_sym_kls, dtype=np.float64)
        out["pairwise_sym_kl_mean"] = float(pairwise_sym_kls_np.mean())
        out["pairwise_sym_kl_std"] = (
            float(pairwise_sym_kls_np.std(ddof=1))
            if len(pairwise_sym_kls_np) > 1 else float("nan")
        )

    return out

def load_test_data(args, dir_path: str):
    """
    Load test data saved by run_generate_data().
    Expects test_data.pt to be a dict with:
      - draws: list[draw_dict]
      - x_query: torch.Tensor [num_query_points, x_dim] sampled from Normal
    """
    data_path = os.path.join(dir_path, "test_data.pt")
    obj = torch.load(data_path)
    if not isinstance(obj, dict) or "draws" not in obj or "x_query" not in obj:
        raise FileNotFoundError(
            f"{data_path} is not in the expected format. Re-run with --mode generate_data to regenerate it."
        )
    draws = obj["draws"]
    x_query = obj["x_query"]
    
    return draws, x_query


def _load_mcmc_oracle_probs_for_permutation(
    dir_path: str,
    expected_num_draws: int,
) -> Optional[List[np.ndarray]]:
    """
    Load per-draw oracle posterior predictive probabilities from existing MCMC artifacts.
    """
    candidate_paths = [os.path.join(dir_path, "baseline_mcmc_merged.pt")]
    candidate_paths.extend(
        sorted(
            [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.startswith("final_results_ctx") and f.endswith(".pt")
            ]
        )
    )

    def _extract_refs(obj: Any, source_path: str) -> Optional[List[np.ndarray]]:
        if not isinstance(obj, list) or len(obj) != expected_num_draws:
            print(
                f"[warn] Oracle source {source_path} has unexpected structure/length "
                f"(expected list of {expected_num_draws} draws)."
            )
            return None

        refs: List[np.ndarray] = []
        for i, item in enumerate(obj):
            if not isinstance(item, dict):
                print(f"[warn] Oracle source {source_path}, draw {i}: expected dict.")
                return None

            mcmc_entry = item.get("mcmc", None)
            if not isinstance(mcmc_entry, dict) or ("q_final" not in mcmc_entry):
                print(
                    f"[warn] Oracle source {source_path}, draw {i}: "
                    "missing mcmc.q_final."
                )
                return None

            q_final = torch.as_tensor(mcmc_entry["q_final"], dtype=torch.float32).view(-1)
            refs.append(q_final.detach().cpu().numpy())
        return refs

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"[warn] Failed to load oracle source {path}: {e}")
            continue
        refs = _extract_refs(obj, path)
        if refs is not None:
            print(f"✓ Loaded permutation oracle (MCMC q_final) from {path}")
            return refs

    return None
# -------------------------
# Modules
# -------------------------

def run_generate_data(args):
    """Step 1: Generate and store test data."""
    print(f"Generating {args.num_test_draws} test draws using {args.prior_dist} prior...")
    test_draws_data = []

    # Shared x_query sampled from Normal(args.x_mean, args.x_std) for all draws and methods
    gq = torch.Generator(device="cpu").manual_seed(int(args.seed) + 12_345)
    x_query = torch.randn(int(args.num_query_points), int(args.x_dim), generator=gq) * args.x_std + args.x_mean

    for draw_idx in tqdm(range(args.num_test_draws)):
        # Create a per-draw transform
        seed_i = draw_seed(args.transform_seed, draw_idx)
        g = torch.Generator(device="cpu").manual_seed(seed_i)
        transform = make_transforms(args.transform_name, args.x_dim, generator=g)
        if transform is not None:
            transform.eval()
            
        # Target Task
        w_target = sample_weight_vec(args)
        with torch.no_grad():
            w_target = apply_transforms(w_target, transform)
        x_pool, y_pool = generate_logistic_task(w_target, args.baseline_context_len, args.noise_std, args.x_mean, args.x_std)

        # Prior tasks
        prior_tasks = []
        for _ in range(args.num_prior_tasks):
            w_p = sample_weight_vec(args)
            with torch.no_grad():
                w_p = apply_transforms(w_p, transform)
            x_p, y_p = generate_logistic_task(w_p, args.sequence_length, args.noise_std, args.x_mean, args.x_std)
            prior_tasks.append((x_p, y_p))

        test_draws_data.append({
            "w_target": w_target, "x_pool": x_pool, "y_pool": y_pool, "prior_tasks_data": prior_tasks,
            "transform_name": args.transform_name,
            "transform_seed": seed_i,
        })

    save_path = os.path.join(get_save_paths(args), "test_data.pt")
    torch.save({"draws": test_draws_data, "x_query": x_query}, save_path)
    print(f"✓ Data saved to {save_path}")

def run_eval_neural(args, device):
    """Step 2: Neural model evaluation (GPU)."""
    dir_path = get_save_paths(args)
    data_path = os.path.join(dir_path, "test_data.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} not found! Generate data first!")

    test_draws_data, x_query = load_test_data(args, dir_path)

    # -----------------------------
    # Analysis mode flags
    # -----------------------------
    do_perm_dataset_order = bool(args.permute_prior_datasets)
    do_perm_points = bool(args.permute_prior_points)
    do_any_perm = do_perm_dataset_order or do_perm_points

    do_prefix_resampling = bool(args.resample_prior_prefixes)

    if do_any_perm and int(args.permute_trials) <= 0:
        raise ValueError("Permutation ablations requested, but --permute-trials must be >= 1.")

    if do_prefix_resampling and int(args.num_prefix_resamples) <= 0:
        raise ValueError(
            "Prefix resampling requested, but --num-prefix-resamples must be >= 1."
        )

    if do_any_perm and do_prefix_resampling:
        raise ValueError(
            "Use either permutation ablations or prior-prefix resampling in one run, not both."
        )

    perm_seed_base = int(args.seed if args.permute_seed is None else args.permute_seed)

    # -----------------------------
    # Oracle refs for comparison
    # Needed for both permutation and prefix-resampling analyses
    # -----------------------------
    perm_oracle_probs_by_draw: Optional[List[np.ndarray]] = None
    if do_any_perm or do_prefix_resampling:
        perm_oracle_probs_by_draw = _load_mcmc_oracle_probs_for_permutation(
            dir_path=dir_path,
            expected_num_draws=len(test_draws_data),
        )
        if perm_oracle_probs_by_draw is None:
            raise RuntimeError(
                "Analysis requested (permutation or prior-prefix resampling), but no "
                "existing MCMC oracle predictions were found "
                "(baseline_mcmc_merged.pt / final_results_ctx*.pt)."
            )

    def _analysis_run_suffix() -> str:
        if do_prefix_resampling:
            return f"prefix_resample_{int(args.num_prefix_resamples)}trials"

        if do_perm_dataset_order and do_perm_points:
            mode = "perm_both"
        elif do_perm_dataset_order:
            mode = "perm_dataset"
        elif do_perm_points:
            mode = "perm_points"
        else:
            mode = "no_perm"

        trials = int(args.permute_trials) if do_any_perm else 0
        return f"{mode}_{trials}trials"

    def _eval_one_model(*, model_name: str, ckpt_path: str, neural_key: str, use_prior_tasks: bool) -> str:
        model = MultiTaskImplicitInContextLearner.load_from_checkpoint(
            ckpt_path, map_location="cpu"
        ).to(device).eval()

        xq = x_query.to(device)

        results = []
        analysis_agg = defaultdict(lambda: defaultdict(list))

        for draw_idx, draw in enumerate(tqdm(test_draws_data, desc=f"Neural Eval ({model_name})")):
            x_ctx = draw["x_pool"][:args.baseline_context_len].to(device)
            y_ctx = draw["y_pool"][:args.baseline_context_len].to(device)
            prior_tasks_data = draw["prior_tasks_data"] if use_prior_tasks else []

            print(f"Current model name: {model_name}")

            logits = predict_with_bootstrap(
                model,
                prior_tasks_data,
                x_ctx,
                y_ctx,
                xq,
                use_task_ids=(getattr(model.hparams, "identity_dim", 0) > 0) and use_prior_tasks,
                num_bootstraps=args.num_bootstraps,
            )
            q_probs = torch.sigmoid(logits).mean(dim=0)

            result_item: Dict[str, Any] = {neural_key: {"q_probs": q_probs.cpu()}}

            # ============================================================
            # resample fresh prior prefixes while fixing target ctx
            # ============================================================
            if do_prefix_resampling and use_prior_tasks:
                if perm_oracle_probs_by_draw is None:
                    raise RuntimeError("Internal error: missing oracle probabilities for prefix resampling.")

                p_oracle = perm_oracle_probs_by_draw[draw_idx]
                if int(np.asarray(p_oracle).reshape(-1).shape[0]) != int(q_probs.numel()):
                    raise RuntimeError(
                        f"Draw {draw_idx}: MCMC oracle size mismatch "
                        f"(oracle={int(np.asarray(p_oracle).reshape(-1).shape[0])}, "
                        f"neural={int(q_probs.numel())})."
                    )

                trial_preds: List[Dict[str, Any]] = []

                for trial_idx in range(int(args.num_prefix_resamples)):
                    sampled_prior_tasks = _sample_fresh_prior_prefix_for_draw(
                        args,
                        draw=draw,
                        resample_idx=(draw_idx * int(args.num_prefix_resamples) + trial_idx),
                    )

                    logits_r = predict_with_bootstrap(
                        model,
                        sampled_prior_tasks,
                        x_ctx,
                        y_ctx,
                        xq,
                        use_task_ids=(getattr(model.hparams, "identity_dim", 0) > 0),
                        num_bootstraps=args.num_bootstraps,
                    )
                    q_probs_r = (
                        torch.sigmoid(logits_r)
                        .mean(dim=0)
                        .detach()
                        .cpu()
                        .view(-1)
                        .numpy()
                    )

                    trial_preds.append(
                        {
                            "trial_idx": int(trial_idx),
                            "q_probs": q_probs_r,
                        }
                    )

                resample_summary = _summarize_prediction_set_against_oracle(
                    trial_preds=trial_preds,
                    oracle_probs=p_oracle,
                )

                result_item[neural_key]["prefix_resampling"] = {
                    "num_trials": int(args.num_prefix_resamples),
                    "summary": resample_summary,
                }

                for metric_name in (
                    "oracle_kl_mean",
                    "oracle_kl_std",
                    "pairwise_sym_kl_mean",
                    "pairwise_sym_kl_std",
                ):
                    v = resample_summary.get(metric_name, float("nan"))
                    if np.isfinite(v):
                        analysis_agg["prefix_resampling"][metric_name].append(float(v))

            # ============================================================
            # permutation sensitivity
            # ============================================================
            elif do_any_perm and use_prior_tasks and len(prior_tasks_data) > 0:
                if perm_oracle_probs_by_draw is None:
                    raise RuntimeError(
                        "Internal error: permutation mode enabled without loaded MCMC oracle refs."
                    )

                p_oracle = perm_oracle_probs_by_draw[draw_idx]
                if int(np.asarray(p_oracle).reshape(-1).shape[0]) != int(q_probs.numel()):
                    raise RuntimeError(
                        f"Draw {draw_idx}: MCMC oracle size mismatch "
                        f"(oracle={int(np.asarray(p_oracle).reshape(-1).shape[0])}, "
                        f"neural={int(q_probs.numel())})."
                    )

                perm_info: Dict[str, Any] = {}

                if do_perm_dataset_order:
                    dataset_trial_preds: List[Dict[str, Any]] = []
                    for trial_idx in range(int(args.permute_trials)):
                        s = _perm_trial_seed(perm_seed_base, draw_idx, trial_idx, kind_offset=17)
                        rng = np.random.default_rng(s)
                        prior_perm = _permute_prior_task_order(prior_tasks_data, rng)

                        logits_p = predict_with_bootstrap(
                            model,
                            prior_perm,
                            x_ctx,
                            y_ctx,
                            xq,
                            use_task_ids=(getattr(model.hparams, "identity_dim", 0) > 0),
                            num_bootstraps=args.num_bootstraps,
                        )
                        q_probs_p = (
                            torch.sigmoid(logits_p)
                            .mean(dim=0)
                            .detach()
                            .cpu()
                            .view(-1)
                            .numpy()
                        )
                        dataset_trial_preds.append(
                            {
                                "trial_idx": int(trial_idx),
                                "seed": int(s),
                                "q_probs": q_probs_p,
                            }
                        )

                    dataset_summary = _summarize_prediction_set_against_oracle(
                        trial_preds=dataset_trial_preds,
                        oracle_probs=p_oracle,
                    )
                    perm_info["prior_dataset_order"] = {
                        "trials": [{"trial_idx": t["trial_idx"], "seed": t["seed"]} for t in dataset_trial_preds],
                        "summary": dataset_summary,
                    }

                    for metric_name in (
                        "oracle_kl_mean",
                        "oracle_kl_std",
                        "pairwise_sym_kl_mean",
                        "pairwise_sym_kl_std",
                    ):
                        v = dataset_summary.get(metric_name, float("nan"))
                        if np.isfinite(v):
                            analysis_agg["prior_dataset_order"][metric_name].append(float(v))

                if do_perm_points:
                    point_trial_preds: List[Dict[str, Any]] = []
                    for trial_idx in range(int(args.permute_trials)):
                        s = _perm_trial_seed(perm_seed_base, draw_idx, trial_idx, kind_offset=71)
                        rng = np.random.default_rng(s)
                        prior_perm = _permute_points_within_prior_tasks(prior_tasks_data, rng)

                        logits_p = predict_with_bootstrap(
                            model,
                            prior_perm,
                            x_ctx,
                            y_ctx,
                            xq,
                            use_task_ids=(getattr(model.hparams, "identity_dim", 0) > 0),
                            num_bootstraps=args.num_bootstraps,
                        )
                        q_probs_p = (
                            torch.sigmoid(logits_p)
                            .mean(dim=0)
                            .detach()
                            .cpu()
                            .view(-1)
                            .numpy()
                        )
                        point_trial_preds.append(
                            {
                                "trial_idx": int(trial_idx),
                                "seed": int(s),
                                "q_probs": q_probs_p,
                            }
                        )

                    point_summary = _summarize_prediction_set_against_oracle(
                        trial_preds=point_trial_preds,
                        oracle_probs=p_oracle,
                    )
                    perm_info["within_prior_dataset_points"] = {
                        "trials": [{"trial_idx": t["trial_idx"], "seed": t["seed"]} for t in point_trial_preds],
                        "summary": point_summary,
                    }

                    for metric_name in (
                        "oracle_kl_mean",
                        "oracle_kl_std",
                        "pairwise_sym_kl_mean",
                        "pairwise_sym_kl_std",
                    ):
                        v = point_summary.get(metric_name, float("nan"))
                        if np.isfinite(v):
                            analysis_agg["within_prior_dataset_points"][metric_name].append(float(v))

                if perm_info:
                    result_item[neural_key]["perm_sensitivity"] = perm_info

            results.append(result_item)

        run_suffix = _analysis_run_suffix()
        out_file = os.path.join(dir_path, f"{model_name}_{run_suffix}_results.pt")
        torch.save(results, out_file)
        print("✓ Neural evaluation complete, saved to ", out_file)

        # Backward-compat: preserve original filename when no extra analysis is requested.
        if not do_any_perm and not do_prefix_resampling:
            legacy_file = os.path.join(dir_path, f"{model_name}_results.pt")
            if legacy_file != out_file:
                torch.save(results, legacy_file)
                print("✓ Backward-compatible neural file saved to ", legacy_file)

        if (do_any_perm or do_prefix_resampling) and use_prior_tasks:
            analysis_summary: Dict[str, Any] = {}
            for analysis_kind, by_metric in analysis_agg.items():
                analysis_summary[analysis_kind] = {}
                for metric_name, vals in by_metric.items():
                    vals_np = np.asarray(vals, dtype=np.float64)
                    if len(vals_np) == 0:
                        analysis_summary[analysis_kind][metric_name] = {
                            "mean": float("nan"),
                            "std": float("nan"),
                            "n": 0,
                        }
                    else:
                        analysis_summary[analysis_kind][metric_name] = {
                            "mean": float(vals_np.mean()),
                            "std": float(vals_np.std(ddof=1)) if len(vals_np) > 1 else float("nan"),
                            "n": int(len(vals_np)),
                        }

            analysis_out_file = os.path.join(
                dir_path,
                f"{model_name}_{run_suffix}_analysis_summary.pt",
            )
            torch.save(
                {
                    "meta": {
                        "model_name": model_name,
                        "checkpoint": ckpt_path,
                        "resample_prior_prefixes": bool(args.resample_prior_prefixes),
                        "num_prefix_resamples": int(args.num_prefix_resamples),
                        "prefix_resample_seed": int(
                            args.seed if args.prefix_resample_seed is None else args.prefix_resample_seed
                        ),
                        "permute_prior_datasets": bool(args.permute_prior_datasets),
                        "permute_prior_points": bool(args.permute_prior_points),
                        "permute_trials": int(args.permute_trials),
                        "permute_seed": int(perm_seed_base),
                    },
                    "summary": analysis_summary,
                },
                analysis_out_file,
            )
            print("✓ Analysis summary saved to ", analysis_out_file)

        return out_file

    # Model 1
    ckpt1 = args.checkpoint_path or os.path.join(
        f"checkpoints/multi_task_implicit_learner_logistic_{args.model_name}",
        args.checkpoint_filename
    )
    use_prior_1 = not bool(args.neural_no_prior_tasks)
    _eval_one_model(
        model_name=args.model_name,
        ckpt_path=ckpt1,
        neural_key=args.neural_key,
        use_prior_tasks=use_prior_1,
    )

    # Optional Model 2
    has_model2 = (
        (args.model_name2 is not None)
        or (args.checkpoint_path2 is not None)
        or (args.checkpoint_filename2 is not None)
    )
    if has_model2:
        if args.model_name2 is None:
            raise ValueError(
                "For a second neural eval, please set --model-name2 "
                "(used for naming the output file)."
            )

        ckpt2 = args.checkpoint_path2 or os.path.join(
            f"checkpoints/multi_task_implicit_learner_logistic_{args.model_name2}",
            args.checkpoint_filename2
        )
        use_prior_2 = not bool(args.neural2_no_prior_tasks)
        _eval_one_model(
            model_name=args.model_name2,
            ckpt_path=ckpt2,
            neural_key=args.neural_key2,
            use_prior_tasks=use_prior_2,
        )
    
def run_eval_baseline(args, device):
    """Step 3: Baseline evaluation (CPU)."""
    def _is_mcmc_oracle(args):
        if args.prior_dist == "normal" and args.transform_name == "identity":
            return True
        return False
    
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    device = torch.device("cpu")
    
    dir_path = get_save_paths(args)
    test_draws_data, x_query = load_test_data(args, dir_path)
    
    draw_start = int(args.draw_start)
    draw_end = len(test_draws_data) if args.draw_end is None else int(args.draw_end)
    assert 0 <= draw_start < draw_end <= len(test_draws_data)
    
    x_query = x_query.to(device)
    
    results = [{} for _ in range(draw_end - draw_start)]

    for local_i, global_i in enumerate(tqdm(range(draw_start, draw_end), desc=f"Baseline Eval [{draw_start},{draw_end})")):
        draw = test_draws_data[global_i]
        x_ctx = draw["x_pool"][:args.baseline_context_len].to(device)
        y_ctx = draw["y_pool"][:args.baseline_context_len].to(device)
        p_gt = torch.sigmoid(x_query @ draw["w_target"].to(device))
        
        # --- MCMC Simple ---
        if "mcmc" in args.baseline_types:
            if _is_mcmc_oracle(args):
                num_thinned = args.mcmc_oracle_num_samples//args.mcmc_thinning
            else:
                num_thinned = max(args.k_list)//args.mcmc_thinning
            w_mcmc, q_final = run_oracle_mcmc_predictive(x_ctx, y_ctx, x_query, args.prior_mean, args.prior_scale, 
                                                 num_thinned, args.mcmc_warmup_steps, args.mcmc_thinning)
            q_probs = []
            for K in args.k_list:
                qp = _mcmc_predict_proba_xquery(x_query, w_mcmc[:K//args.mcmc_thinning])
                q_probs.append(qp)
            results[local_i]["mcmc"] = {"q_probs": q_probs, "K": list(args.k_list), "q_final": q_final}

        # --- MCMC Spiral Oracle (known theta from saved transform_seed) ---
        if "mcmc_spiral_oracle" in args.baseline_types:
            w_samps, q_final = run_oracle_mcmc_predictive_spiral_known_theta(
                x_ctx,
                y_ctx,
                x_query,
                base_mean=float(args.prior_mean),
                base_scale=float(args.prior_scale),
                transform_seed=int(draw["transform_seed"]),
                num_thinned=int(args.mcmc_oracle_num_samples // args.mcmc_thinning),
                warmup=int(args.mcmc_warmup_steps),
                thinning=int(args.mcmc_thinning),
                num_chains=int(args.mcmc_num_chains),
                speed=1.0,
                random_initial=False,
                initial_scale=0.1,
                jit_compile=False,
            )
            q_list = []
            for K in args.k_list:
                q_list.append(_mcmc_predict_proba_xquery(x_query, w_samps[: K // args.mcmc_thinning]))
            results[local_i]["mcmc_spiral_oracle"] = {"q_probs": q_list, "K": list(args.k_list), "q_final": q_final}

        # --- MCMC Hier ---
        if "mcmc_hier" in args.baseline_types:
            num_thinned = max(args.k_list)//args.mcmc_thinning
            w_h, q_final = run_oracle_hierarchical_mcmc_predictive(
                draw["prior_tasks_data"],
                x_ctx,
                y_ctx,
                x_query,
                prior_mean_min=float(args.hier_prior_mean_min),
                prior_mean_max=float(args.hier_prior_mean_max),
                num_thinned=num_thinned,
                warmup=int(args.mcmc_warmup_steps),
                thinning=int(args.mcmc_thinning),
            )
            q_probs = []
            for K in args.k_list:
                qp = _mcmc_predict_proba_xquery(x_query, w_h[:K//args.mcmc_thinning])
                q_probs.append(qp)
            results[local_i]["mcmc_hier"] = {"q_probs": q_probs, "K": list(args.k_list), "q_final": q_final}
            
        # --- SVI simple ---
        if "svi" in args.baseline_types:
            qp_list = run_svi_curve(x_ctx, y_ctx, x_query, args.prior_mean, args.prior_scale, set(args.k_list), args.svi_posterior_samples)
            results[local_i]["svi"] = {"q_probs": qp_list, "K": list(args.k_list)}
            
        # --- SVI partial progress ---
        if "svi_hier" in args.baseline_types:
            qp_list = run_hier_svi_curve(x_ctx, y_ctx, x_query, draw["prior_tasks_data"], set(args.k_list), 
                                         args.hier_prior_mean_min, args.hier_prior_mean_max, args.svi_posterior_samples)
            results[local_i]["svi_hier"] = {"q_probs": qp_list, "K": list(args.k_list)}

        if "mcmc_hier_spiral" in args.baseline_types:
            w_samps, q_probs = run_oracle_hierarchical_mcmc_predictive_spiral(
                draw["prior_tasks_data"],
                x_ctx, y_ctx, x_query,
                prior_mean_min=float(args.hier_prior_mean_min),
                prior_mean_max=float(args.hier_prior_mean_max),
                num_thinned=max(args.k_list)//args.mcmc_thinning,
                warmup=int(args.mcmc_warmup_steps),
                thinning=int(args.mcmc_thinning),
                A_scale=1.0,      
                speed=1.0,
                jit_compile=False,
            )
            q_list = []
            for K in args.k_list:
                q_list.append(_mcmc_predict_proba_xquery(x_query, w_samps[:K // args.mcmc_thinning]))
            results[local_i]["mcmc_hier_spiral"] = {"q_probs": q_list, "K": list(args.k_list)}
        
        if "svi_hier_spiral" in args.baseline_types:
            qp_list = run_hier_svi_curve_spiral(
                x_ctx, y_ctx, x_query, draw["prior_tasks_data"],
                steps_set=set(args.k_list),
                hier_prior_mean_min=float(args.hier_prior_mean_min),
                hier_prior_mean_max=float(args.hier_prior_mean_max),
                svi_posterior_samples=args.svi_posterior_samples,
                lr=1e-2,
                A_scale=1.0,
                speed=1.0,
            )
            results[local_i]["svi_hier_spiral"] = {"q_probs": qp_list, "K": list(args.k_list)}
                
    # save shard
    shard_tag = f"{draw_start}_{draw_end}"
    assert len(args.baseline_types) == 1, "Only one baseline type is supported for now"
    suffix = args.baseline_types[0]
    out_path = os.path.join(dir_path, f"baseline_{suffix}_shard_{shard_tag}.pt")
    torch.save({"draw_start": draw_start, "draw_end": draw_end, "results": results}, out_path)
    print(f"✓ Saved shard to {out_path}")

# for maximal parallelism
def run_merge_baseline_shards(args):
    """Step 3.5: Merge baseline shards."""
    dir_path = get_save_paths(args)
    assert len(args.baseline_types) == 1, "Only one baseline type is supported for now"
    suffix = args.baseline_types[0]

    shard_files = sorted([
        f for f in os.listdir(dir_path)
        if f.startswith(f"baseline_{suffix}_shard_") and f.endswith(".pt")
    ])
    if not shard_files:
        raise FileNotFoundError("No shard files found.")

    # allocate full list
    test_draws_data, _ = load_test_data(args, dir_path)
    full = [{} for _ in range(len(test_draws_data))]

    for sf in shard_files:
        shard = torch.load(os.path.join(dir_path, sf))
        s, e = shard["draw_start"], shard["draw_end"]
        res = shard["results"]
        assert len(res) == (e - s)
        for i in range(s, e):
            full[i].update(res[i - s])

    out_path = os.path.join(dir_path, f"baseline_{suffix}_merged.pt")
    torch.save(full, out_path)
    print(f"✓ Merged baselines saved to {out_path}")

def run_merge(args):
    """Step 4: Merge results from neural and all baselines."""
    dir_path = get_save_paths(args)
    final_results = []
    
    for bl in args.baseline_types:
        print(f"Merging {bl}...")
        
        if os.path.exists(os.path.join(dir_path, f"baseline_{bl}_merged.pt")):
            print(f"✓ {bl} merged file found")
        else:
            print(f"✗ {bl} merged file not found")
            print(dir_path, f"baseline_{bl}_merged.pt")
    print(f"Merging neural results from {args.model_name}...")
    neural_res = torch.load(os.path.join(dir_path, f"{args.model_name}_results.pt")) if os.path.exists(os.path.join(dir_path, f"{args.model_name}_results.pt")) else None

    neural_res2 = None
    if args.model_name2 is not None:
        print(f"Merging second neural results from {args.model_name2}...")
        p2 = os.path.join(dir_path, f"{args.model_name2}_results.pt")
        neural_res2 = torch.load(p2) if os.path.exists(p2) else None
    baseline_files = [f"baseline_{suffix}_merged.pt" for suffix in args.baseline_types if os.path.exists(os.path.join(dir_path, f"baseline_{suffix}_merged.pt"))]
    if not baseline_files or not neural_res:
        raise FileNotFoundError("Missing merged baseline file or neural results.")
    if args.model_name2 is not None and neural_res2 is None:
        raise FileNotFoundError(f"Missing second neural results file for {args.model_name2}. Run --mode eval_neural first.")
    
    test_draws_data, _ = load_test_data(args, dir_path)

    # Load baseline merged lists once
    baseline_data_by_file = {
        bf: torch.load(os.path.join(dir_path, bf), map_location="cpu") for bf in baseline_files
    }

    def _inner_key_for_merge(outer_key: str) -> str:
        if outer_key.startswith("mcmc_hier_warmup") and outer_key.endswith("_spiral"):
            return "mcmc_hier_spiral"
        return outer_key
    
    for i in range(len(test_draws_data)):
        combined = {"context_len": args.baseline_context_len}
        combined.update(neural_res[i])
        if neural_res2 is not None:
            combined.update(neural_res2[i])
        for bf in baseline_files:
            suffix = bf[len("baseline_") : -len("_merged.pt")]
            b_data = baseline_data_by_file[bf]
            inner_key = _inner_key_for_merge(suffix)
            if isinstance(b_data, list) and i < len(b_data) and isinstance(b_data[i], dict):
                if inner_key in b_data[i]:
                    combined[suffix] = b_data[i][inner_key]
                    continue
            combined.update(b_data[i])
        final_results.append(combined)

    save_path = os.path.join(dir_path, f"final_results_ctx{args.baseline_context_len}.pt")
    torch.save(final_results, save_path)
    print(f"✓ Merged results saved to {save_path}")

# -------------------------
# Entry Point
# -------------------------

def main():
    args = parse_args()
    
    # Validate arguments
    K_MAX = max(args.k_list)
    assert K_MAX <= args.mcmc_oracle_num_samples, "K_MAX must be <= total MCMC samples"
    for K in args.k_list:
        assert K % args.mcmc_thinning == 0, f"K={K} must be divisible by thinning={args.mcmc_thinning}"
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else ("cuda" if args.device == "cuda" else "cpu"))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        pyro.set_rng_seed(args.seed)

    if args.mode == "generate_data": run_generate_data(args)
    elif args.mode == "eval_neural": run_eval_neural(args, device)
    elif args.mode == "eval_baseline": run_eval_baseline(args, device)
    elif args.mode == "merge_baseline_shards": run_merge_baseline_shards(args)
    elif args.mode == "merge": run_merge(args)

if __name__ == "__main__":
    main()
#!/usr/bin/env python
# eval_linear.py
"""
Modular linear-regression evaluation script with --mode.

Modes:
  1) generate_data:
      - generate fixed test draws (prior tasks + target pool)
      - compute Ridge oracle predictive (mu/logvar) on x_query
      - save test_data.pt

  2) eval_neural:
      - load test_data.pt
      - run neural model on GPU (or cpu)
      - optionally run permutation-sensitivity ablations:
          (a) random permutations of prior dataset order
          (b) random permutations of points within each prior dataset
      - save neural predictions (mu/logvar) to <model_name>_results.pt

  3) eval_baseline:
      - load test_data.pt
      - run hierarchical Pyro baselines on CPU (MCMC/SVI)
      - optionally shard over draws with --draw-start/--draw-end
      - save baseline_<suffix>_shard_<s>_<e>.pt

  4) merge_baseline_shards:
      - merge baseline shard files for a given suffix
      - save baseline_<suffix>_merged.pt

  5) merge:
      - merge ridge oracle + neural + baselines into final_results.pt
      - also compute KL vs Ridge oracle and plot comparison
"""

from __future__ import annotations

import argparse
import os
import math
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Project-specific imports
from src.models.multi_task_implicit_model import MultiTaskImplicitInContextLearner

from src.utils.linear_baselines import (
    gaussian_kl,
    ridge_oracle_predict,
    predict_with_bootstrap,
    predict_hierarchical_mcmc,
    predict_hierarchical_svi,
)
from src.utils.eval_helpers import _summarize_prediction_set_kl_only
# -------------------------
# Default priors
# -------------------------
PRIOR_DISTRIBUTIONS = {
    "Mean 0": (0.0, 1.0),
    "Mean 4": (4.0, 1.0),
    "Mean 8": (8.0, 1.0),
    "Mean 10": (10.0, 1.0),
}

# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Modular Linear Evaluation Script")

    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["generate_data", "eval_neural", "eval_baseline", "merge_baseline_shards", "merge"],
        help="Which step to run.",
    )

    # Paths / output structure
    p.add_argument("--output-root", type=str, default="results/linear_results_short_prefix")
    p.add_argument("--tag", type=str, default="", help="Optional extra tag for output folder naming.")

    # Data settings
    p.add_argument("--x-dim", type=int, default=8, help="If None, infer from checkpoint in eval_neural.")
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--noise-mean", type=float, default=0.0)
    p.add_argument("--num-prior-tasks", type=int, default=5)
    p.add_argument("--prior-task-len", type=int, default=20)
    p.add_argument("--baseline-context-lens", type=int, nargs="+", default=[1, 5, 10, 15, 20])
    p.add_argument("--num-query-points", type=int, default=100)
    p.add_argument("--num-test-draws", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)

    # Neural model settings
    p.add_argument("--model-name", type=str, default="multi_task_implicit_learner_large_-8to8_training_short_data")
    p.add_argument("--checkpoint-filename", type=str, default="best-model-epoch=98-val_loss=0.67.ckpt")
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-bootstraps", type=int, default=1)
    p.add_argument(
        "--permute-prior-datasets",
        action="store_true",
        help="For eval_neural: run ablation over random permutations of prior dataset order.",
    )
    p.add_argument(
        "--permute-prior-points",
        action="store_true",
        help="For eval_neural: run ablation over random permutations of points within each prior dataset.",
    )
    p.add_argument(
        "--permute-trials",
        type=int,
        default=0,
        help="For eval_neural permutation ablations: number of random permutations per example.",
    )
    p.add_argument(
        "--permute-seed",
        type=int,
        default=None,
        help="Optional base seed for permutation ablations. Defaults to --seed when omitted.",
    )
    p.add_argument(
        "--no-prior",
        action="store_true",
        help="Evaluate a model trained with no prior prefix: ignore prior_tasks and feed only the target context.",
    )
    p.add_argument(
        "--neural-results",
        nargs="+",
        default=None,
        help=(
            "For --mode merge: load one or more neural result .pt files. "
            "Each entry can be either a path or 'label=path'. "
            "If omitted, defaults to <out_dir>/<model_name>_results.pt."
        ),
    )

    # Baseline settings
    p.add_argument("--baseline-types", nargs="+", default=["mcmc_hier", "svi_hier"],
                   choices=["mcmc_hier", "svi_hier"],
                   help="Which baselines to run in eval_baseline.")
    p.add_argument("--hier-prior-mean-min", type=float, default=-8.0)
    p.add_argument("--hier-prior-mean-max", type=float, default=8.0)

    p.add_argument("--mcmc-num-samples", type=int, default=1000)
    p.add_argument("--mcmc-warmup-steps", type=int, default=1000)
    p.add_argument("--mcmc-thinning", type=int, default=10)
    p.add_argument("--mcmc-disable-progbar", action="store_true")

    p.add_argument("--svi-steps", type=int, default=1000)
    p.add_argument("--svi-posterior-samples", type=int, default=200)
    p.add_argument("--svi-lr", type=float, default=0.01)

    # Sharding for baseline
    p.add_argument("--draw-start", type=int, default=0)
    p.add_argument("--draw-end", type=int, default=None)  # exclusive

    return p.parse_args()

# -------------------------
# Helpers
# -------------------------
def _load_test_data_or_raise(out_dir: str) -> List[Dict[str, Any]]:
    data_path = os.path.join(out_dir, "test_data.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run --mode generate_data first.")
    payload = torch.load(data_path, map_location="cpu")
    return payload["test_data"]


def _resolve_checkpoint_or_raise(args) -> str:
    ckpt = args.checkpoint_path
    if ckpt is None:
        ckpt = os.path.join("checkpoints", args.model_name, args.checkpoint_filename)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_save_dir(args) -> str:
    # stable tag so multiple runs don’t collide
    # includes noise, draws, queries, seed, optional user tag
    tag = f"{args.num_test_draws}draws_{args.num_query_points}queries_noise{args.noise_std:g}_seed{args.seed}"
    if args.tag:
        tag = f"{tag}_{args.tag}"
    out_dir = os.path.join(args.output_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _parse_labeled_path(spec: str) -> Tuple[str, str]:
    """
    Parse either:
      - "label=path/to/file.pt"
      - "path/to/file.pt"  (label inferred from basename without extension)
    Returns (label, path).
    """
    if "=" in spec:
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label:
            raise ValueError(f"Empty label in neural result spec: {spec!r}")
        if not path:
            raise ValueError(f"Empty path in neural result spec: {spec!r}")
        return label, path
    base = os.path.basename(spec)
    label = os.path.splitext(base)[0]
    return label, spec

def _sample_w(mean_scalar: float, std_scalar: float, x_dim: int) -> torch.Tensor:
    return torch.randn(x_dim, 1) * std_scalar + mean_scalar

def _generate_linear_task(w: torch.Tensor, sequence_length: int, noise_std: float,
                          x_mean: float = 0.0, x_std: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [N, D], y: [N, 1]
    x = torch.empty(sequence_length, w.shape[0]).normal_(x_mean, x_std)
    y = x @ w
    if noise_std > 0:
        y = y + torch.randn_like(y) * noise_std
    return x, y

def _make_x_query_normal(num_query_points: int, x_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(num_query_points, x_dim, device=device)

def _compute_gaussian_metrics_vs_oracle(
    *,
    oracle_mu: np.ndarray,
    oracle_lv: np.ndarray,
    pred_mu: np.ndarray,
    pred_lv: np.ndarray,
) -> Tuple[float, float, float]:
    """Return (kl, abs_mean_diff, nll) averaged across query points."""
    kl = float(np.mean(gaussian_kl(oracle_mu, oracle_lv, pred_mu, pred_lv)))
    absdiff = float(np.mean(np.abs(pred_mu - oracle_mu)))
    nll = float(np.mean(gaussian_nll_from_oracle(oracle_mu, oracle_lv, pred_mu, pred_lv)))
    return kl, absdiff, nll


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
    # Deterministic seed per (example, trial, ablation kind) so runs are reproducible.
    return int(base_seed + 1_000_003 * global_draw_id + 10_007 * trial_idx + kind_offset)


def _summarize_trials(trial_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {"num_trials": int(len(trial_metrics))}
    if len(trial_metrics) == 0:
        for k in ("kl", "abs_mean_diff", "nll"):
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_std"] = float("nan")
        return out

    for k in ("kl", "abs_mean_diff", "nll"):
        vals = np.asarray([tm[k] for tm in trial_metrics], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
    return out


def _summary_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for method, by_prior in metrics.items():
        summary[method] = {}
        for prior_name, by_ctx in by_prior.items():
            summary[method][prior_name] = {}
            for ctx_len, vals in by_ctx.items():
                m, s = _mean_sem(vals)
                summary[method][prior_name][str(ctx_len)] = {"mean": m, "sem": s, "n": len(vals)}
    return summary


def _method_order_from_args(args) -> List[str]:
    if args.neural_results:
        return [f"Neural[{_parse_labeled_path(s)[0]}]" for s in args.neural_results]
    return ["Neural"]


def _method_color(method: str) -> Optional[str]:
    if method == "Ridge":
        return "tab:red"
    if method == "MCMC_hier":
        return "tab:blue"

    if method.startswith("Neural"):
        m = method.lower()
        no_prefix_tokens = (
            "no prefix",
            "no prior",
            "noprefix",
            "no_prefix",
            "no-prior",
            "no_prior",
            "withoutprefix",
            "wo_prefix",
            "woprefix",
        )
        if any(tok in m for tok in no_prefix_tokens):
            return "tab:pink"
        return "tab:green"

    return None


def _plot_metric_by_prior(
    *,
    out_dir: str,
    priors: List[str],
    ctx_lens: List[int],
    summary: Dict[str, Any],
    methods_in_plot: List[str],
    ylabel: str,
    filename: str,
) -> str:
    nrows = len(priors)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, max(3, 2.4 * nrows)), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, prior_name in zip(axes, priors):
        for method in methods_in_plot:
            ys, es = [], []
            for c in ctx_lens:
                rec = summary.get(method, {}).get(prior_name, {}).get(str(int(c)), None)
                ys.append(np.nan if rec is None else rec["mean"])
                es.append(np.nan if rec is None else rec["sem"])
            color = _method_color(method)
            if method == "MCMC_hier":
                method_name = "MCMC-HIER"
            elif method == "Neural[with prefix]":
                method_name = "ICL (with prefix)"
            elif method == "Neural[no prefix]":
                method_name = "ICL (no prefix)"
            else:
                method_name = method
            line, = ax.plot(ctx_lens, ys, marker="o", label=method_name, color=color)
            if not all(np.isnan(ys)) and not all(np.isnan(es)):
                y_arr = np.array(ys)
                e_arr = np.array(es)
                ax.fill_between(ctx_lens, y_arr - e_arr, y_arr + e_arr, alpha=0.2, color=line.get_color())
        if prior_name == "Mean 10":
            prior_name = "Mean 10 (OoMD)"
        ax.set_title(prior_name)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Context length", fontsize=14)
    axes[0].legend(loc="best")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, filename)
    fig.savefig(plot_path, dpi=300)
    return plot_path


# -------------------------
# Modules
# -------------------------
def run_generate_data(args):
    """
    Step 1: Generate and store test data + ridge oracle predictive (mu/logvar).
    Saves:
      - test_data.pt
        list of dicts, one per (prior_name, ctx_len, draw_idx)
    """
    out_dir = _get_save_dir(args)
    save_path = os.path.join(out_dir, "test_data.pt")

    if args.x_dim is None:
        raise ValueError("--x-dim is required for generate_data (or run eval_neural first to infer).")

    x_dim = int(args.x_dim)

    x_query_cpu = _make_x_query_normal(args.num_query_points, x_dim, device=torch.device("cpu"))

    test_data: List[Dict[str, Any]] = []
    global_draw_id = 0
    max_ctx = int(max(args.baseline_context_lens))

    rng_base = int(args.seed)

    for prior_name, (p_mean, p_std) in PRIOR_DISTRIBUTIONS.items():
        for ctx_len in args.baseline_context_lens:
            for draw_idx in tqdm(range(args.num_test_draws), desc=f"gen {prior_name} ctx={ctx_len}", leave=False):
                seed_i = rng_base + draw_idx + 10_000 * (hash(prior_name) % 10_000) + 1_000_000 * int(ctx_len)
                torch.manual_seed(seed_i)
                np.random.seed(seed_i % (2**32 - 1))

                # Target task pool (length = max ctx len)
                w_target = _sample_w(p_mean, p_std, x_dim)
                x_pool = torch.empty(max_ctx, x_dim).normal_(0.0, 1.0)
                y_pool = x_pool @ w_target
                if args.noise_std > 0:
                    y_pool = y_pool + torch.randn_like(y_pool) * float(args.noise_std)

                x_ctx = x_pool[:ctx_len]
                y_ctx = y_pool[:ctx_len]

                # Prior tasks
                prior_tasks: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for _ in range(int(args.num_prior_tasks)):
                    w_p = _sample_w(p_mean, p_std, x_dim)
                    x_p, y_p = _generate_linear_task(w_p, int(args.prior_task_len), float(args.noise_std))
                    prior_tasks.append((x_p, y_p))

                # Ridge oracle predictive
                ridge_mu, ridge_lv = ridge_oracle_predict(
                    x_ctx=x_ctx,
                    y_ctx=y_ctx,
                    x_query=x_query_cpu,
                    prior_mean=float(p_mean),
                    prior_std=float(p_std),
                    noise_std=float(args.noise_std),
                    noise_mean=float(args.noise_mean),
                )

                test_data.append({
                    "global_draw_id": global_draw_id,
                    "prior_name": prior_name,
                    "p_mean": float(p_mean),
                    "p_std": float(p_std),
                    "ctx_len": int(ctx_len),
                    "draw_idx": int(draw_idx),
                    "seed": int(seed_i),

                    # Store CPU tensors
                    "x_ctx": x_ctx.cpu(),
                    "y_ctx": y_ctx.cpu(),
                    "prior_tasks": [(xp.cpu(), yp.cpu()) for (xp, yp) in prior_tasks],

                    # Store oracle
                    "x_query": x_query_cpu.cpu(),
                    "ridge_mu": torch.tensor(ridge_mu, dtype=torch.float32),
                    "ridge_lv": torch.tensor(ridge_lv, dtype=torch.float32),
                })
                global_draw_id += 1

    torch.save({"meta": vars(args), "test_data": test_data}, save_path)
    print(f"✓ Saved data + ridge oracle to {save_path} (num_examples={len(test_data)})")

def run_eval_neural(args, device: torch.device):
    """
    Step 2: Neural model eval (GPU).

    Loads test_data.pt, runs predict_with_bootstrap, saves <model_name>_results.pt
    """
    out_dir = _get_save_dir(args)
    test_data_all = _load_test_data_or_raise(out_dir)
    ckpt = _resolve_checkpoint_or_raise(args)

    model = MultiTaskImplicitInContextLearner.load_from_checkpoint(ckpt, map_location="cpu").to(device).eval()

    inferred_x_dim = int(getattr(model.hparams, "x_dim", None) or 0)
    assert inferred_x_dim == int(args.x_dim)
    x_dim = inferred_x_dim

    # -------------------------
    # Rebuttal-specific filtering
    # Only prior mean = 0 and ctx_len in {20, 50}
    # -------------------------
    allowed_prior_names = {"Mean 4", "Mean 8", "Mean 10"}
    allowed_ctx_lens = {20, 50}

    test_data = [
        ex for ex in test_data_all
        if str(ex["prior_name"]) in allowed_prior_names and int(ex["ctx_len"]) in allowed_ctx_lens
    ]

    if len(test_data) == 0:
        raise ValueError(
            "After filtering for prior mean 0 and context lengths {20, 50}, no evaluation examples remain. "
            "Please check your existing test_data.pt."
        )

    print(
        f"[info] Filtered eval_neural examples for permutation sensitivity: "
        f"{len(test_data)} / {len(test_data_all)} kept "
        f"(prior_name in {sorted(allowed_prior_names)}, ctx_len in {sorted(allowed_ctx_lens)})"
    )

    do_perm_dataset_order = bool(args.permute_prior_datasets)
    do_perm_points = bool(args.permute_prior_points)
    do_any_perm = do_perm_dataset_order or do_perm_points
    if do_any_perm and int(args.permute_trials) <= 0:
        raise ValueError("Permutation ablations requested, but --permute-trials must be >= 1.")

    perm_seed_base = int(args.seed if args.permute_seed is None else args.permute_seed)

    results = {}  # global_draw_id -> {mu, lv, ...}

    perm_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for ex in tqdm(test_data, desc="Neural Eval"):
        global_draw_id = int(ex["global_draw_id"])

        # Move tensors
        x_query = ex["x_query"].to(device)           # [Q, D]
        x_ctx = ex["x_ctx"].to(device)               # [N, D]
        y_ctx = ex["y_ctx"].to(device)               # [N, 1]
        prior_tasks = [(xp.to(device), yp.to(device)) for (xp, yp) in ex["prior_tasks"]] if not args.no_prior else []

        # Standard prediction for this example (kept for consistency with existing pipeline)
        if args.no_prior:
            mu, lv = predict_with_bootstrap(
                model=model,
                prior_tasks=[],
                x_target_context_pool=x_ctx,
                y_target_context_pool=y_ctx,
                x_target_query=x_query,
            )
        else:
            mu, lv = predict_with_bootstrap(
                model=model,
                prior_tasks=prior_tasks,
                x_target_context_pool=x_ctx,
                y_target_context_pool=y_ctx,
                x_target_query=x_query,
            )

        mu, lv = mu.squeeze(-1), lv.squeeze(-1)  # [Q]

        results[global_draw_id] = {
            "mu": torch.tensor(mu, dtype=torch.float32),
            "lv": torch.tensor(lv, dtype=torch.float32),
            "prior_name": str(ex["prior_name"]),
            "ctx_len": int(ex["ctx_len"]),
        }

        # -------------------------
        # Permutation sensitivity
        # -------------------------
        if do_any_perm and (not args.no_prior) and len(prior_tasks) > 0:
            oracle_mu = ex["ridge_mu"].view(-1).numpy()
            oracle_lv = ex["ridge_lv"].view(-1).numpy()
            prior_name = str(ex["prior_name"])
            ctx_len = int(ex["ctx_len"])

            perm_info: Dict[str, Any] = {}

            # (a) Permute order of prior datasets
            if do_perm_dataset_order:
                dataset_trial_preds: List[Dict[str, Any]] = []

                for trial_idx in range(int(args.permute_trials)):
                    s = _perm_trial_seed(perm_seed_base, global_draw_id, trial_idx, kind_offset=17)
                    rng = np.random.default_rng(s)
                    prior_perm = _permute_prior_task_order(prior_tasks, rng)

                    mu_p, lv_p = predict_with_bootstrap(
                        model=model,
                        prior_tasks=prior_perm,
                        x_target_context_pool=x_ctx,
                        y_target_context_pool=y_ctx,
                        x_target_query=x_query,
                    )
                    mu_p = np.asarray(mu_p.squeeze(-1), dtype=np.float64)
                    lv_p = np.asarray(lv_p.squeeze(-1), dtype=np.float64)

                    dataset_trial_preds.append({
                        "trial_idx": int(trial_idx),
                        "seed": int(s),
                        "mu": mu_p,
                        "lv": lv_p,
                    })

                dataset_summary = _summarize_prediction_set_kl_only(
                    trial_preds=dataset_trial_preds,
                    oracle_mu=oracle_mu,
                    oracle_lv=oracle_lv,
                )

                perm_info["prior_dataset_order"] = {
                    "trials": [
                        {"trial_idx": t["trial_idx"], "seed": t["seed"]}
                        for t in dataset_trial_preds
                    ],
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
                        perm_agg["prior_dataset_order"][prior_name][ctx_len][metric_name].append(float(v))

            # (b) Permute points within each prior dataset
            if do_perm_points:
                point_trial_preds: List[Dict[str, Any]] = []

                for trial_idx in range(int(args.permute_trials)):
                    s = _perm_trial_seed(perm_seed_base, global_draw_id, trial_idx, kind_offset=71)
                    rng = np.random.default_rng(s)
                    prior_perm = _permute_points_within_prior_tasks(prior_tasks, rng)

                    mu_p, lv_p = predict_with_bootstrap(
                        model=model,
                        prior_tasks=prior_perm,
                        x_target_context_pool=x_ctx,
                        y_target_context_pool=y_ctx,
                        x_target_query=x_query,
                    )
                    mu_p = np.asarray(mu_p.squeeze(-1), dtype=np.float64)
                    lv_p = np.asarray(lv_p.squeeze(-1), dtype=np.float64)

                    point_trial_preds.append({
                        "trial_idx": int(trial_idx),
                        "seed": int(s),
                        "mu": mu_p,
                        "lv": lv_p,
                    })

                point_summary = _summarize_prediction_set_kl_only(
                    trial_preds=point_trial_preds,
                    oracle_mu=oracle_mu,
                    oracle_lv=oracle_lv,
                )

                perm_info["within_prior_dataset_points"] = {
                    "trials": [
                        {"trial_idx": t["trial_idx"], "seed": t["seed"]}
                        for t in point_trial_preds
                    ],
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
                        perm_agg["within_prior_dataset_points"][prior_name][ctx_len][metric_name].append(float(v))

            if perm_info:
                results[global_draw_id]["perm_sensitivity"] = perm_info

    if do_any_perm and (args.no_prior or int(args.num_prior_tasks) == 0):
        print("[warn] Permutation ablations requested but no prior is used (or no prior tasks configured). Skipping ablations.")
        
    def _perm_run_suffix(args) -> str:
        if args.permute_prior_datasets and args.permute_prior_points:
            mode = "perm_both"
        elif args.permute_prior_datasets:
            mode = "perm_dataset"
        elif args.permute_prior_points:
            mode = "perm_points"
        else:
            mode = "no_perm"
        return f"{mode}_{int(args.permute_trials)}trials"
    
    run_suffix = _perm_run_suffix(args)
    out_path = os.path.join(out_dir, f"{args.model_name}_{run_suffix}_results.pt")
    payload_meta = {
        "checkpoint": ckpt,
        "device": str(device),
        "permute_prior_datasets": bool(args.permute_prior_datasets),
        "permute_prior_points": bool(args.permute_prior_points),
        "permute_trials": int(args.permute_trials),
        "permute_seed": int(perm_seed_base),
        "filtered_prior_names": sorted(list(allowed_prior_names)),
        "filtered_ctx_lens": sorted(list(allowed_ctx_lens)),
    }
    torch.save({"meta": payload_meta, "preds": results}, out_path)
    print(f"✓ Saved neural preds to {out_path} (num_examples={len(results)})")

    # Aggregate permutation-sensitivity summaries across examples
    if do_any_perm and (not args.no_prior) and int(args.num_prior_tasks) > 0:
        perm_summary: Dict[str, Any] = {}
        for ablation_kind, by_prior in perm_agg.items():
            perm_summary[ablation_kind] = {}
            for prior_name, by_ctx in by_prior.items():
                perm_summary[ablation_kind][prior_name] = {}
                for ctx_len, metric_lists in by_ctx.items():
                    perm_summary[ablation_kind][prior_name][str(int(ctx_len))] = {}
                    for metric_name, vals in metric_lists.items():
                        m, s = _mean_sem(vals)
                        perm_summary[ablation_kind][prior_name][str(int(ctx_len))][metric_name] = {
                            "mean": m,
                            "sem": s,
                            "n": len(vals),
                        }
        perm_out_path = os.path.join(out_dir, f"{args.model_name}_{run_suffix}_permutation_sensitivity.pt")
        torch.save(
            {
                "meta": payload_meta,
                "summary": perm_summary,
            },
            perm_out_path,
        )
        print(f"✓ Saved permutation sensitivity summary to {perm_out_path}")

def run_eval_baseline(args):
    """
    Step 3: Baseline eval (CPU), shardable.
    Produces:
      baseline_<suffix>_shard_<draw_start>_<draw_end>.pt
    """
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    device = torch.device("cpu")

    out_dir = _get_save_dir(args)
    test_data = _load_test_data_or_raise(out_dir)

    draw_start = int(args.draw_start)
    draw_end = len(test_data) if args.draw_end is None else int(args.draw_end)
    assert 0 <= draw_start < draw_end <= len(test_data)

    baseline_types = list(args.baseline_types)
    suffix = baseline_types[0]
    assert len(baseline_types) == 1, "Only one baseline type is supported for now"

    shard_results: Dict[int, Dict[str, Any]] = {}

    for idx in tqdm(range(draw_start, draw_end), desc=f"Baseline Eval [{draw_start},{draw_end})"):
        ex = test_data[idx]
        global_draw_id = int(ex["global_draw_id"])

        x_query = ex["x_query"].to(device)
        x_ctx = ex["x_ctx"].to(device)
        y_ctx = ex["y_ctx"].to(device)
        prior_tasks = [(xp.to(device), yp.to(device)) for (xp, yp) in ex["prior_tasks"]]
        x_dim = int(x_query.shape[1])

        out_ex: Dict[str, Any] = {}

        if "mcmc_hier" in baseline_types:
            m_mu, m_lv = predict_hierarchical_mcmc(
                prior_tasks,
                x_ctx,
                y_ctx,
                x_query,
                x_dim=x_dim,
                noise_std=float(args.noise_std),
                num_samples=int(args.mcmc_num_samples),
                thinning=int(args.mcmc_thinning),
                warmup_steps=int(args.mcmc_warmup_steps),
                hier_prior_mean_min=float(args.hier_prior_mean_min),
                hier_prior_mean_max=float(args.hier_prior_mean_max),
                disable_progbar=bool(args.mcmc_disable_progbar),
            )
            out_ex["mcmc_hier"] = {
                "mu": torch.tensor(m_mu, dtype=torch.float32),
                "lv": torch.tensor(m_lv, dtype=torch.float32),
            }

        if "svi_hier" in baseline_types:
            s_mu, s_lv = predict_hierarchical_svi(
                prior_tasks,
                x_ctx,
                y_ctx,
                x_query,
                x_dim=x_dim,
                noise_std=float(args.noise_std),
                steps=int(args.svi_steps),
                posterior_samples=int(args.svi_posterior_samples),
                hier_prior_mean_min=float(args.hier_prior_mean_min),
                hier_prior_mean_max=float(args.hier_prior_mean_max),
                lr=float(args.svi_lr),
            )
            out_ex["svi_hier"] = {
                "mu": torch.tensor(s_mu, dtype=torch.float32),
                "lv": torch.tensor(s_lv, dtype=torch.float32),
            }

        shard_results[global_draw_id] = out_ex

    shard_tag = f"{draw_start}_{draw_end}"
    out_path = os.path.join(out_dir, f"baseline_{suffix}_shard_{shard_tag}.pt")
    torch.save({"draw_start": draw_start, "draw_end": draw_end, "suffix": suffix, "preds": shard_results}, out_path)
    print(f"✓ Saved baseline shard to {out_path} (num_examples={len(shard_results)})")

def run_merge_baseline_shards(args):
    """
    Step 3.5: Merge baseline shards for the requested one baseline_types suffix.
    Produces:
      baseline_<suffix>_merged.pt
    """
    out_dir = _get_save_dir(args)
    
    assert len(args.baseline_types) == 1, "Only one baseline type is supported for now"
    suffix = args.baseline_types[0]

    shard_files = sorted([
        f for f in os.listdir(out_dir)
        if f.startswith(f"baseline_{suffix}_shard_") and f.endswith(".pt")
    ])
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {out_dir} for suffix={suffix}")

    merged: Dict[int, Dict[str, Any]] = {}
    for sf in shard_files:
        shard = torch.load(os.path.join(out_dir, sf), map_location="cpu")
        preds = shard["preds"]
        for global_draw_id, rec in preds.items():
            merged.setdefault(global_draw_id, {}).update(rec)

    out_path = os.path.join(out_dir, f"baseline_{suffix}_merged.pt")
    torch.save({"suffix": suffix, "preds": merged, "shards": shard_files}, out_path)
    print(f"✓ Merged {len(shard_files)} shards -> {out_path} (num_examples={len(merged)})")

def _mean_sem(xs: List[float]) -> Tuple[float, float]:
    xs = np.asarray(xs, dtype=np.float64)
    if len(xs) == 0:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return float(xs.mean()), float("nan")
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs)))

def gaussian_nll_from_oracle(
    oracle_mu: np.ndarray,
    oracle_lv: np.ndarray,
    pred_mu: np.ndarray,
    pred_lv: np.ndarray,
    *,
    var_floor: float = 1e-8,
) -> np.ndarray:
    """
    Cross-entropy / expected NLL of y ~ N(oracle_mu, oracle_var) under pred distribution:
      E_y[-log N(y; pred_mu, pred_var)].

    All inputs are shape [Q] (or broadcastable). Returns shape [Q].
    """
    oracle_mu = np.asarray(oracle_mu, dtype=np.float64)
    oracle_lv = np.asarray(oracle_lv, dtype=np.float64)
    pred_mu   = np.asarray(pred_mu, dtype=np.float64)
    pred_lv   = np.asarray(pred_lv, dtype=np.float64)

    oracle_var = np.maximum(np.exp(oracle_lv), var_floor)
    pred_var   = np.maximum(np.exp(pred_lv), var_floor)

    return 0.5 * (
        np.log(2.0 * np.pi * pred_var) +
        (oracle_var + (oracle_mu - pred_mu) ** 2) / pred_var
    )


def run_merge(args):
    """
    Step 4: Merge ridge oracle + neural + baseline(s), compute KL vs ridge, plot.
    Produces:
      - final_results.pt (per-example record)
      - summary.json-like dict in final_summary.pt
      - plot_kl.png
      - plot_abs_mean_diff.png
    """
    out_dir = _get_save_dir(args)

    test_data = _load_test_data_or_raise(out_dir)

    # Load one or more neural preds.
    # Each neural result file is expected to contain: {"preds": {global_draw_id: {"mu":..., "lv":...}}}
    neural_models: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
    if args.neural_results:
        for spec in args.neural_results:
            label, path = _parse_labeled_path(spec)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing neural results file: {path}")
            payload_i = torch.load(path, map_location="cpu")
            neural_models[label] = payload_i["preds"]
    else:
        neural_path = os.path.join(out_dir, f"{args.model_name}_results.pt")
        if not os.path.exists(neural_path):
            raise FileNotFoundError(f"Missing {neural_path}. Run --mode eval_neural first.")
        neural_payload = torch.load(neural_path, map_location="cpu")
        neural_models["Neural"] = neural_payload["preds"]

    # Load baseline merged preds (optional)
    baseline_types = list(args.baseline_types)
    baseline_preds = {}
    for suffix in baseline_types:
        baseline_merged_path = os.path.join(out_dir, f"baseline_{suffix}_merged.pt")
        if os.path.exists(baseline_merged_path):
            baseline_payload = torch.load(baseline_merged_path, map_location="cpu")
            baseline_preds[suffix] = baseline_payload["preds"]
        else:
            print(f"[warn] No merged baseline file found at {baseline_merged_path}. Will merge only neural vs ridge.")
    # Per-example merge + KL
    final_results: List[Dict[str, Any]] = []
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    metrics_abs_mean_diff = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    metrics_nll = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for ex in tqdm(test_data, desc="Merging"):
        global_draw_id = int(ex["global_draw_id"])
        prior_name = ex["prior_name"]
        ctx_len = int(ex["ctx_len"])

        ridge_mu = ex["ridge_mu"].view(-1).numpy()
        ridge_lv = ex["ridge_lv"].view(-1).numpy()
        ridge_nll = float(np.mean(gaussian_nll_from_oracle(ridge_mu, ridge_lv, ridge_mu, ridge_lv)))

        rec: Dict[str, Any] = {
            "global_draw_id": global_draw_id,
            "prior_name": prior_name,
            "ctx_len": ctx_len,
            "ridge": {"mu": ex["ridge_mu"], "lv": ex["ridge_lv"]},
        }
        rec["ridge"]["nll"] = ridge_nll
        metrics_nll["Ridge"][prior_name][ctx_len].append(ridge_nll)

        # Neural models (one or many)
        rec["neural_models"] = {}
        for label, preds_by_id in neural_models.items():
            if global_draw_id not in preds_by_id:
                print(f"[warn] global_draw_id={global_draw_id} missing from neural preds ({label}); skipping.")
                continue
            n_mu = preds_by_id[global_draw_id]["mu"].view(-1).numpy()
            n_lv = preds_by_id[global_draw_id]["lv"].view(-1).numpy()
            kl_n, absdiff_n, nll_n = _compute_gaussian_metrics_vs_oracle(
                oracle_mu=ridge_mu, oracle_lv=ridge_lv, pred_mu=n_mu, pred_lv=n_lv
            )

            # Back-compat: if a single default model is used, keep method name as "Neural".
            method_name = "Neural" if (not args.neural_results and label == "Neural") else f"Neural[{label}]"
            entry = {
                "mu": preds_by_id[global_draw_id]["mu"],
                "lv": preds_by_id[global_draw_id]["lv"],
                "kl": kl_n,
                "nll": nll_n,
                "abs_mean_diff": absdiff_n,
            }
            rec["neural_models"][label] = entry
            if method_name == "Neural":
                rec["neural"] = entry
            metrics[method_name][prior_name][ctx_len].append(kl_n)
            metrics_abs_mean_diff[method_name][prior_name][ctx_len].append(absdiff_n)
            metrics_nll[method_name][prior_name][ctx_len].append(nll_n)

        # Baselines
        if baseline_preds:
            for _suffix, preds_by_id in baseline_preds.items():
                if global_draw_id not in preds_by_id:
                    continue
                b = preds_by_id[global_draw_id]
                if "mcmc_hier" in b:
                    m_mu = b["mcmc_hier"]["mu"].view(-1).numpy()
                    m_lv = b["mcmc_hier"]["lv"].view(-1).numpy()
                    kl_m = float(np.mean(gaussian_kl(ridge_mu, ridge_lv, m_mu, m_lv)))
                    absdiff_m = float(np.mean(np.abs(m_mu - ridge_mu)))
                    nll_m = float(np.mean(gaussian_nll_from_oracle(ridge_mu, ridge_lv, m_mu, m_lv)))
                    rec["mcmc_hier"] = {"mu": b["mcmc_hier"]["mu"], "lv": b["mcmc_hier"]["lv"], "kl": kl_m, "abs_mean_diff": absdiff_m, "nll": nll_m}
                    metrics["MCMC_hier"][prior_name][ctx_len].append(kl_m)
                    metrics_abs_mean_diff["MCMC_hier"][prior_name][ctx_len].append(absdiff_m)
                    metrics_nll["MCMC_hier"][prior_name][ctx_len].append(nll_m)
                if "svi_hier" in b:
                    s_mu = b["svi_hier"]["mu"].view(-1).numpy()
                    s_lv = b["svi_hier"]["lv"].view(-1).numpy()
                    kl_s, absdiff_s, nll_s = _compute_gaussian_metrics_vs_oracle(
                        oracle_mu=ridge_mu, oracle_lv=ridge_lv, pred_mu=s_mu, pred_lv=s_lv
                    )
                    rec["svi_hier"] = {"mu": b["svi_hier"]["mu"], "lv": b["svi_hier"]["lv"], "kl": kl_s, "abs_mean_diff": absdiff_s, "nll": nll_s}
                    metrics["SVI_hier"][prior_name][ctx_len].append(kl_s)
                    metrics_abs_mean_diff["SVI_hier"][prior_name][ctx_len].append(absdiff_s)
                    metrics_nll["SVI_hier"][prior_name][ctx_len].append(nll_s)
        final_results.append(rec)

    # Save final per-example merged file
    final_path = os.path.join(out_dir, "final_results.pt")
    torch.save(final_results, final_path)

    # Aggregate summary
    summary = _summary_from_metrics(metrics)

    summary_path = os.path.join(out_dir, "final_summary.pt")
    torch.save(summary, summary_path)
    
    # Aggregate NLL summary
    nll_summary = _summary_from_metrics(metrics_nll)

    nll_summary_path = os.path.join(out_dir, "final_nll_summary.pt")
    torch.save(nll_summary, nll_summary_path)

    # Aggregate abs-mean-diff summary (parallel to KL summary)
    abs_mean_diff_summary = {}
    for method, by_prior in metrics_abs_mean_diff.items():
        abs_mean_diff_summary[method] = {}
        for prior_name, by_ctx in by_prior.items():
            abs_mean_diff_summary[method][prior_name] = {}
            for ctx_len, vals in by_ctx.items():
                m, s = _mean_sem(vals)
                abs_mean_diff_summary[method][prior_name][str(ctx_len)] = {"mean": m, "sem": s, "n": len(vals)}
    abs_mean_diff_summary_path = os.path.join(out_dir, "final_abs_mean_diff_summary.pt")
    torch.save(abs_mean_diff_summary, abs_mean_diff_summary_path)

    # Plot: one subplot per prior
    priors = list(PRIOR_DISTRIBUTIONS.keys())
    ctx_lens = list(args.baseline_context_lens)

    # Preserve user-specified order for neural curves if provided.
    neural_method_order = _method_order_from_args(args)

    methods_in_plot = neural_method_order + ["MCMC_hier"]+["SVI_hier"]
    methods_in_plot = [m for m in methods_in_plot if m in summary]

    plot_path = _plot_metric_by_prior(
        out_dir=out_dir,
        priors=priors,
        ctx_lens=ctx_lens,
        summary=summary,
        methods_in_plot=methods_in_plot,
        ylabel="KL",
        filename="plot_kl.png",
    )

    # Plot mean absolute difference in means: E[|mu_method - mu_ridge|]
    methods_in_plot_abs = neural_method_order + ["MCMC_hier"]+["SVI_hier"]
    methods_in_plot_abs = [m for m in methods_in_plot_abs if m in abs_mean_diff_summary]
    plot_abs_path = _plot_metric_by_prior(
        out_dir=out_dir,
        priors=priors,
        ctx_lens=ctx_lens,
        summary=abs_mean_diff_summary,
        methods_in_plot=methods_in_plot_abs,
        ylabel="Mean |μ - μ_oracle|",
        filename="plot_abs_mean_diff.png",
    )

    print(f"✓ Saved final merged results: {final_path}")
    print(f"✓ Saved summary: {summary_path}")
    print(f"✓ Saved abs-mean-diff summary: {abs_mean_diff_summary_path}")
    print(f"✓ Saved plot: {plot_path}")
    print(f"✓ Saved plot: {plot_abs_path}")
    
    # Plot Gaussian NLL: Ridge, Neural(with/without prefix), MCMC_hier
    # Determine desired method order
    neural_method_order = _method_order_from_args(args)

    methods_in_plot_nll = ["Ridge"] + neural_method_order + ["MCMC_hier"]+["SVI_hier"]
    methods_in_plot_nll = [m for m in methods_in_plot_nll if m in nll_summary]
    plot_nll_path = _plot_metric_by_prior(
        out_dir=out_dir,
        priors=priors,
        ctx_lens=ctx_lens,
        summary=nll_summary,
        methods_in_plot=methods_in_plot_nll,
        ylabel="Gaussian NLL",
        filename="plot_nll.png",
    )

    print(f"✓ Saved NLL summary: {nll_summary_path}")
    print(f"✓ Saved plot: {plot_nll_path}")


# -------------------------
# Entry Point
# -------------------------
def main():
    args = parse_args()

    # Seed
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed) % (2**32 - 1))

    device = _device_from_arg(args.device)

    if args.mode == "generate_data":
        run_generate_data(args)
    elif args.mode == "eval_neural":
        run_eval_neural(args, device)
    elif args.mode == "eval_baseline":
        run_eval_baseline(args)
    elif args.mode == "merge_baseline_shards":
        run_merge_baseline_shards(args)
    elif args.mode == "merge":
        run_merge(args)
    else:
        raise ValueError(args.mode)

if __name__ == "__main__":
    main()

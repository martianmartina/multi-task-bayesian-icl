#!/usr/bin/env python
"""
Evaluate a Student-t tailedness OOD grid for multi-task logistic ICL.

  - mode=make_draws: pre-generate shared draws and save to draws_path
  - mode=neural: evaluate neural checkpoints only; save neural results
  - mode=baseline: evaluate ONE baseline method only; save baseline shard
  - mode=merge: merge neural + baseline shards into a final results file

Also supports df sharding (for job arrays):
  --df_shard_id k --df_num_shards K  => only compute df indices i where i % K == k
"""

import argparse
import os
import glob
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.multi_task_implicit_model import MultiTaskImplicitInContextLearner
from utils.eval_helpers import (
    sample_normal_vec,
    sample_student_t_vec,
    generate_logistic_task,
    bernoulli_kl,
    bernoulli_tv,
    predict_with_bootstrap,
    run_oracle_mcmc_predictive,
    run_oracle_hierarchical_mcmc_predictive,
    run_oracle_hierarchical_df_mcmc_predictive,
    run_oracle_hierarchical_df_mixture_mu_mcmc_predictive,
    run_svi_curve,
    run_hier_svi_curve,
    run_hier_df_svi_curve,
    run_hier_df_mixture_mu_svi_curve,
)


# ----------------------------- small utils -----------------------------

def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_x_query(num_query_points: int, x_dim: int) -> torch.Tensor:
    return torch.linspace(-1, 1, num_query_points).unsqueeze(1).expand(-1, x_dim)

def _make_x_query_normal(num_query_points: int, x_dim: int) -> torch.Tensor:
    return torch.randn(num_query_points, x_dim)

def _df_to_prior_family(df: float) -> Tuple[str, Optional[float]]:
    # df=inf (tailedness=0) -> Normal
    if df is None or (isinstance(df, float) and (np.isinf(df) or df >= 1e6)):
        return "normal", None
    return "student_t", float(df)


def _sample_w(df: float, w_mean: float, w_std: float, x_dim: int) -> torch.Tensor:
    prior_family, st_df = _df_to_prior_family(df)
    if prior_family == "normal":
        return sample_normal_vec(mean_scalar=w_mean, std_scalar=w_std, dim=x_dim)
    return sample_student_t_vec(df=st_df, mean_scalar=w_mean, scale_scalar=w_std, dim=x_dim)


def _normalize_methods(methods: List[str]) -> List[str]:
    return [m.strip().lower() for m in methods]


def _config_fingerprint(cfg: Dict[str, Any]) -> str:
    payload = yaml.safe_dump(cfg, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _df_indices_for_shard(n: int, shard_id: int, num_shards: int) -> List[int]:
    if num_shards <= 1:
        return list(range(n))
    return [i for i in range(n) if (i % num_shards) == shard_id]


# ----------------------------- draw generation -----------------------------

def _generate_draw(
    *,
    df: float,
    x_dim: int,
    num_prior_tasks: int,
    sequence_length: int,
    context_len: int,
    noise_std: float,
    x_mean: float,
    x_std: float,
    w_mean: float,
    w_std: float,
    num_query_points: int,
) -> Dict[str, Any]:
    # Target w
    w_target = _sample_w(df=df, w_mean=w_mean, w_std=w_std, x_dim=x_dim)
    x_pool, y_pool = generate_logistic_task(
        w_target,
        sequence_length=max(sequence_length, context_len),
        noise_std=noise_std,
        x_mean=x_mean,
        x_std=x_std,
    )

    # Prior tasks
    prior_tasks_data = []
    for _ in range(num_prior_tasks):
        w_p = _sample_w(df=df, w_mean=w_mean, w_std=w_std, x_dim=x_dim)
        x_p, y_p = generate_logistic_task(
            w_p,
            sequence_length=sequence_length,
            noise_std=noise_std,
            x_mean=x_mean,
            x_std=x_std,
        )
        prior_tasks_data.append((x_p, y_p))

    x_query = _make_x_query(num_query_points=num_query_points, x_dim=x_dim)
    p_gt = torch.sigmoid(x_query @ w_target)  # [N,1]

    w_target = w_target.cpu()
    x_pool, y_pool = x_pool.cpu(), y_pool.cpu()
    prior_tasks_data = [(xp.cpu(), yp.cpu()) for xp, yp in prior_tasks_data]
    x_query, p_gt = x_query.cpu(), p_gt.cpu()

    return {
        "df": float(df),
        "w_target": w_target,
        "x_pool": x_pool,
        "y_pool": y_pool,
        "prior_tasks_data": prior_tasks_data,
        "x_query": x_query,
        "p_gt": p_gt,
    }


def make_and_save_draws(
    *,
    df_grid: List[float],
    data: Dict[str, Any],
    draws_path: str,
    seed: int,
) -> Dict[float, List[Dict[str, Any]]]:
    """Generate draws_by_df and save to draws_path.""" 
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_test_draws = int(data["num_test_draws"])
    x_dim = int(data["x_dim"])
    num_prior_tasks = int(data["num_prior_tasks"])
    sequence_length = int(data["sequence_length"])
    context_len = int(data["context_len"])
    num_query_points = int(data["num_query_points"])
    noise_std = float(data.get("noise_std", 0.0))
    x_mean = float(data.get("x_mean", 0.0))
    x_std = float(data.get("x_std", 1.0))
    w_mean = float(data.get("w_mean", 0.0))
    w_std = float(data.get("w_std", 1.0))

    draws_by_df: Dict[float, List[Dict[str, Any]]] = {}
    for df in df_grid:
        draws = [
            _generate_draw(
                df=df,
                x_dim=x_dim,
                num_prior_tasks=num_prior_tasks,
                sequence_length=sequence_length,
                context_len=context_len,
                noise_std=noise_std,
                x_mean=x_mean,
                x_std=x_std,
                w_mean=w_mean,
                w_std=w_std,
                num_query_points=num_query_points,
            )
            for _ in range(num_test_draws)
        ]
        draws_by_df[float(df)] = draws

    os.makedirs(os.path.dirname(draws_path) or ".", exist_ok=True)
    torch.save(
        {
            "df_grid": [float(x) for x in df_grid],
            "draws_by_df": draws_by_df,
            "data": data,
            "seed": int(seed),
        },
        draws_path,
    )
    print(f"✓ Saved shared draws to: {draws_path}")
    return draws_by_df


def load_draws(draws_path: str) -> Tuple[List[float], Dict[float, List[Dict[str, Any]]], Dict[str, Any]]:
    """Load df_grid + draws_by_df + data from draws_path."""  
    payload = torch.load(draws_path, map_location="cpu")
    df_grid = [float(x) for x in payload["df_grid"]]
    draws_by_df = payload["draws_by_df"]
    data = payload.get("data", {})
    return df_grid, draws_by_df, data


# ----------------------------- evaluation: neural -----------------------------

def _eval_neural_on_draw(
    *,
    model: MultiTaskImplicitInContextLearner,
    draw: Dict[str, Any],
    context_len: int,
    num_bootstraps: int,
) -> Tuple[float, float]:
    x_pool = draw["x_pool"]
    y_pool = draw["y_pool"]
    prior_tasks_data = draw["prior_tasks_data"]
    x_query = draw["x_query"]
    p_gt = draw["p_gt"].to(next(model.parameters()).device)

    x_ctx = x_pool[:context_len]
    y_ctx = y_pool[:context_len]

    logits_boot = predict_with_bootstrap(
        model,
        prior_tasks=prior_tasks_data,
        x_target_context_pool=x_ctx,
        y_target_context_pool=y_ctx,
        x_target_query=x_query,
        use_task_ids=(getattr(model.hparams, "identity_dim", 0) > 0),
        num_bootstraps=num_bootstraps,
    )  # [B,N,1]
    q_probs = torch.sigmoid(logits_boot).mean(dim=0)  # [N,1]

    kl = float(bernoulli_kl(p_gt, q_probs).item())
    tv = float(bernoulli_tv(p_gt, q_probs).item())
    return kl, tv


def eval_neural(
    *,
    train_models: List[Dict[str, Any]],
    df_grid: List[float],
    draws_by_df: Dict[float, List[Dict[str, Any]]],
    context_len: int,
    num_bootstraps: int,
    device: torch.device,
    df_indices: List[int],  
) -> Dict[str, Any]:
    """Compute neural kl/tv matrices; fill non-shard df cols with NaN."""  
    n_rows = len(train_models)
    n_cols = len(df_grid)
    kl_mat = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    tv_mat = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    for i, row in enumerate(train_models):
        ckpt = row["checkpoint_path"]
        print(f"Loading model [{i+1}/{len(train_models)}]: {ckpt}")
        model = MultiTaskImplicitInContextLearner.load_from_checkpoint(ckpt, map_location="cpu")
        model.eval().to(device)

        for j in df_indices:
            df = df_grid[j]
            kls, tvs = [], []
            for draw in tqdm(draws_by_df[df], desc=f"neural row {i} df {df}", leave=False):
                kl, tv = _eval_neural_on_draw(
                    model=model, draw=draw, context_len=context_len, num_bootstraps=num_bootstraps
                )
                kls.append(kl)
                tvs.append(tv)
            kl_mat[i, j] = float(np.mean(kls))
            tv_mat[i, j] = float(np.mean(tvs))

    return {
        "df_grid": df_grid,
        "train_models": train_models,
        "kl_mat": kl_mat,
        "tv_mat": tv_mat,
        "computed_df_indices": df_indices,
    }


# ----------------------------- evaluation: baselines -----------------------------

def _eval_baseline_method_on_draw(
    *,
    method: str,
    draw: Dict[str, Any],
    context_len: int,
    w_mean: float,
    w_std: float,
    hier_prior_mean_min: float,
    hier_prior_mean_max: float,
    mcmc_num_thinned: int,
    mcmc_warmup: int,
    mcmc_thinning: int,
    mcmc_num_chains: int,
    svi_steps: int,
    svi_posterior_samples: int,
    mixture_df_list: Optional[List[float]] = None,
    tau_eps: float = 1e-3,
) -> Tuple[float, float]:
    """Evaluate a SINGLE baseline method on a draw."""  
    method = method.strip().lower()

    x_pool = draw["x_pool"]
    y_pool = draw["y_pool"]
    prior_tasks_data = draw["prior_tasks_data"]
    x_query = draw["x_query"]
    p_gt = draw["p_gt"]

    x_ctx = x_pool[:context_len]
    y_ctx = y_pool[:context_len]

    prior_family, st_df = _df_to_prior_family(draw["df"])

    def _metrics(q_probs: torch.Tensor) -> Tuple[float, float]:
        return float(bernoulli_kl(p_gt, q_probs).item()), float(bernoulli_tv(p_gt, q_probs).item())

    # -------------------- ORACLE (knows true df) --------------------
    if method == "mcmc_oracle":
        _, q_probs = run_oracle_mcmc_predictive(
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_query=x_query,
            w_mean_oracle=w_mean,
            w_std_oracle=w_std,
            num_thinned=mcmc_num_thinned,
            warmup=mcmc_warmup,
            thinning=mcmc_thinning,
            num_chains=mcmc_num_chains,
            prior_family=prior_family,
            student_t_df=st_df,
            student_t_df_list=None,
        )
        return _metrics(q_probs)

    if method == "svi_oracle":
        q_list = run_svi_curve(
            x_ctx,
            y_ctx,
            x_query,
            prior_mean=w_mean,
            prior_scale=w_std,
            steps_set={int(svi_steps)},
            svi_posterior_samples=int(svi_posterior_samples),
            prior_family=prior_family,
            student_t_df=st_df,
            student_t_df_list=None,
        )
        return _metrics(q_list[-1])

    # -------------------- MIXTURE (meta-prior over df; does NOT know test df) --------------------
    if method == "mcmc_mixture":
        if prior_family != "student_t":
            # fallback
            _, q_probs = run_oracle_mcmc_predictive(
                x_ctx=x_ctx,
                y_ctx=y_ctx,
                x_query=x_query,
                w_mean_oracle=w_mean,
                w_std_oracle=w_std,
                num_thinned=mcmc_num_thinned,
                warmup=mcmc_warmup,
                thinning=mcmc_thinning,
                num_chains=mcmc_num_chains,
                prior_family="normal",
                student_t_df=None,
                student_t_df_list=None,
            )
            return _metrics(q_probs)

        if not mixture_df_list:
            raise ValueError("mcmc_mixture requested but mixture_df_list is empty/missing.")

        _, q_probs = run_oracle_mcmc_predictive(
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_query=x_query,
            w_mean_oracle=w_mean,
            w_std_oracle=w_std,
            num_thinned=mcmc_num_thinned,
            warmup=mcmc_warmup,
            thinning=mcmc_thinning,
            num_chains=mcmc_num_chains,
            prior_family="student_t",
            student_t_df=None,                 # unknown df
            student_t_df_list=mixture_df_list, # mixture meta-prior
        )
        return _metrics(q_probs)

    if method == "svi_mixture":
        if prior_family != "student_t":
            q_list = run_svi_curve(
                x_ctx,
                y_ctx,
                x_query,
                prior_mean=w_mean,
                prior_scale=w_std,
                steps_set={int(svi_steps)},
                svi_posterior_samples=int(svi_posterior_samples),
                prior_family="normal",
                student_t_df=None,
                student_t_df_list=None,
            )
            return _metrics(q_list[-1])

        if not mixture_df_list:
            raise ValueError("svi_mixture requested but mixture_df_list is empty/missing.")

        q_list = run_svi_curve(
            x_ctx,
            y_ctx,
            x_query,
            prior_mean=w_mean,
            prior_scale=w_std,
            steps_set={int(svi_steps)},
            svi_posterior_samples=int(svi_posterior_samples),
            prior_family="student_t",
            student_t_df=None,                 # unknown df
            student_t_df_list=mixture_df_list, # mixture meta-prior
        )
        return _metrics(q_list[-1])

    # -------------------- HIER-T (infers df via tau; loc=0, scale=1 fixed) --------------------
    if method == "mcmc_hier_t":
        _, q_probs_mcmc_hier_t, _samples = run_oracle_hierarchical_df_mcmc_predictive(
            prior_tasks=prior_tasks_data,
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_query=x_query,
            tau_eps=float(tau_eps),
            num_thinned=mcmc_num_thinned,
            warmup=mcmc_warmup,
            thinning=mcmc_thinning,
            num_chains=mcmc_num_chains,
        )
        return _metrics(q_probs_mcmc_hier_t)

    if method == "svi_hier_t":
        q_list = run_hier_df_svi_curve(
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_query=x_query,
            prior_tasks_data=prior_tasks_data,
            steps_set={int(svi_steps)},
            tau_eps=float(tau_eps),
            svi_posterior_samples=int(svi_posterior_samples),
        )
        return _metrics(q_list[-1])

    # -------------------- HIER-DF-MIX-MU (infers mu in [-8,8] and df from df-list; shared across tasks) --------------------
    if method == "mcmc_hier_df_mix_mu":
        if prior_family != "student_t":
            raise ValueError("mcmc_hier_df_mix_mu is intended for prior_family='student_t'.")
        if not mixture_df_list:
            raise ValueError("mcmc_hier_df_mix_mu requested but mixture_df_list is empty/missing.")
        _, q_probs_h, _samples = run_oracle_hierarchical_df_mixture_mu_mcmc_predictive(
            prior_tasks=prior_tasks_data,
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_query=x_query,
            student_t_df_list=[float(x) for x in mixture_df_list],
            prior_mean_min=-8.0,
            prior_mean_max=8.0,
            num_thinned=mcmc_num_thinned,
            warmup=mcmc_warmup,
            thinning=mcmc_thinning,
            num_chains=mcmc_num_chains,
        )
        return _metrics(q_probs_h)

    if method == "svi_hier_df_mix_mu":
        if prior_family != "student_t":
            raise ValueError("svi_hier_df_mix_mu is intended for prior_family='student_t'.")
        if not mixture_df_list:
            raise ValueError("svi_hier_df_mix_mu requested but mixture_df_list is empty/missing.")
        q_list = run_hier_df_mixture_mu_svi_curve(
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_query=x_query,
            prior_tasks_data=prior_tasks_data,
            student_t_df_list=[float(x) for x in mixture_df_list],
            steps_set={int(svi_steps)},
            prior_mean_min=-8.0,
            prior_mean_max=8.0,
            svi_posterior_samples=int(svi_posterior_samples),
        )
        return _metrics(q_list[-1])

    raise KeyError(f"Unknown baseline method: {method}")


def eval_baseline_method(
    *,
    method: str,
    df_grid: List[float],
    draws_by_df: Dict[float, List[Dict[str, Any]]],
    context_len: int,
    w_mean: float,
    w_std: float,
    hier_prior_mean_min: float,
    hier_prior_mean_max: float,
    mcmc_num_thinned: int,
    mcmc_warmup: int,
    mcmc_thinning: int,
    mcmc_num_chains: int,
    svi_steps: int,
    svi_posterior_samples: int,
    mixture_df_list: Optional[List[float]],
    tau_eps: float,
    df_indices: List[int],  
) -> Dict[str, Any]:
    """Compute KL/TV for ONE method across df_grid; fill non-shard df cols with NaN."""  
    n_cols = len(df_grid)
    KL_gt = [float("nan")] * n_cols
    TV_gt = [float("nan")] * n_cols

    for j in df_indices:
        df = df_grid[j]
        acc: List[Tuple[float, float]] = []
        for draw in tqdm(draws_by_df[df], desc=f"{method} df {df}", leave=False):
            acc.append(
                _eval_baseline_method_on_draw(
                    method=method,
                    draw=draw,
                    context_len=context_len,
                    w_mean=w_mean,
                    w_std=w_std,
                    hier_prior_mean_min=hier_prior_mean_min,
                    hier_prior_mean_max=hier_prior_mean_max,
                    mcmc_num_thinned=mcmc_num_thinned,
                    mcmc_warmup=mcmc_warmup,
                    mcmc_thinning=mcmc_thinning,
                    mcmc_num_chains=mcmc_num_chains,
                    svi_steps=svi_steps,
                    svi_posterior_samples=svi_posterior_samples,
                    mixture_df_list=mixture_df_list,
                    tau_eps=tau_eps,
                )
            )
        KL_gt[j] = float(np.mean([x[0] for x in acc]))
        TV_gt[j] = float(np.mean([x[1] for x in acc]))

    return {
        "method": method,
        "df_grid": df_grid,
        "KL_gt": KL_gt,
        "TV_gt": TV_gt,
        "computed_df_indices": df_indices,
    }


# ----------------------------- merging -----------------------------

def merge_results(
    *,
    cfg_fingerprint: str,
    df_grid: List[float],
    train_models: List[Dict[str, Any]],
    neural_paths: List[str],
    baseline_paths: List[str],
    out_path: str,
) -> None:
    """Merge sharded neural + baseline outputs."""  
    # --- merge neural shards ---
    kl_mat = None
    tv_mat = None

    for p in neural_paths:
        shard = torch.load(p, map_location="cpu")
        if shard.get("cfg_fingerprint") != cfg_fingerprint:
            raise ValueError(f"Config mismatch for neural shard {p}: {shard.get('cfg_fingerprint')} != {cfg_fingerprint}")

        shard_kl = shard["kl_mat"]
        shard_tv = shard["tv_mat"]
        if kl_mat is None:
            kl_mat = np.array(shard_kl, dtype=np.float64)
            tv_mat = np.array(shard_tv, dtype=np.float64)
        else:
            # fill NaNs only
            mask = np.isnan(kl_mat) & ~np.isnan(shard_kl)
            kl_mat[mask] = shard_kl[mask]
            mask = np.isnan(tv_mat) & ~np.isnan(shard_tv)
            tv_mat[mask] = shard_tv[mask]

    if kl_mat is None or tv_mat is None:
        raise ValueError("No neural shards provided or could not load neural outputs.")

    # --- merge baseline shards ---
    baseline_rows: Dict[str, Dict[str, List[float]]] = {}
    for p in baseline_paths:
        shard = torch.load(p, map_location="cpu")
        if shard.get("cfg_fingerprint") != cfg_fingerprint:
            raise ValueError(f"Config mismatch for baseline shard {p}: {shard.get('cfg_fingerprint')} != {cfg_fingerprint}")
        method = shard["method"]
        KL_gt = np.array(shard["KL_gt"], dtype=np.float64)
        TV_gt = np.array(shard["TV_gt"], dtype=np.float64)

        if method not in baseline_rows:
            baseline_rows[method] = {
                "KL_gt": [float("nan")] * len(df_grid),
                "TV_gt": [float("nan")] * len(df_grid),
            }
        cur_kl = np.array(baseline_rows[method]["KL_gt"], dtype=np.float64)
        cur_tv = np.array(baseline_rows[method]["TV_gt"], dtype=np.float64)

        # fill NaNs only (allow multiple shards)
        mask = np.isnan(cur_kl) & ~np.isnan(KL_gt)
        cur_kl[mask] = KL_gt[mask]
        mask = np.isnan(cur_tv) & ~np.isnan(TV_gt)
        cur_tv[mask] = TV_gt[mask]

        baseline_rows[method]["KL_gt"] = cur_kl.tolist()
        baseline_rows[method]["TV_gt"] = cur_tv.tolist()

    results = {
        "cfg_fingerprint": cfg_fingerprint,
        "df_grid": df_grid,
        "train_models": train_models,
        "kl_mat": kl_mat,
        "tv_mat": tv_mat,
        "baseline_rows": baseline_rows,
        "neural_paths": neural_paths,
        "baseline_paths": baseline_paths,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(results, out_path)
    print(f"✓ Merged results saved to: {out_path}")


# ----------------------------- main -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to YAML config for the heatmap evaluation.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "make_draws", "neural", "baseline", "merge"],
        help="Which stage to run.",
    )
    parser.add_argument("--draws_path", type=str, default="", help="Path to shared draws .pt file.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (used in make_draws).")

    parser.add_argument("--baseline_method", type=str, default="", help="Baseline method for mode=baseline.")

    parser.add_argument("--df_shard_id", type=int, default=0)
    parser.add_argument("--df_num_shards", type=int, default=1)

    parser.add_argument("--out", type=str, default="", help="Output file for this mode (neural/baseline/merge).")
    parser.add_argument("--neural_glob", type=str, default="", help="For merge: glob pattern for neural shards.")
    parser.add_argument("--baseline_glob", type=str, default="", help="For merge: glob pattern for baseline shards.")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_fp = _config_fingerprint(cfg)  

    device = _device_from_arg(args.device)
    print(f"Using device: {device}")
    print(f"cfg_fingerprint: {cfg_fp}")  

    heat = cfg["heatmap"]
    data = cfg["data"]
    ev = cfg.get("eval", {})
    baselines_cfg = cfg.get("baselines", {"enabled": False})

    df_grid: List[float] = [float(x) for x in heat["df_grid"]]
    train_models: List[Dict[str, Any]] = list(heat["train_models"])

    df_indices = _df_indices_for_shard(len(df_grid), args.df_shard_id, args.df_num_shards)
    print(f"df shard: id={args.df_shard_id} / {args.df_num_shards} => {len(df_indices)}/{len(df_grid)} columns")

    # -------------------- mode: make_draws --------------------
    if args.mode == "make_draws":
        if not args.draws_path:
            raise ValueError("--draws_path is required for mode=make_draws")
        make_and_save_draws(df_grid=df_grid, data=data, draws_path=args.draws_path, seed=args.seed)
        return

    # For any non-make_draws mode, we need draws loaded (or generated if mode=all and no draws_path given)
    draws_by_df: Dict[float, List[Dict[str, Any]]]
    if args.draws_path:
        df_grid_loaded, draws_by_df, _data_loaded = load_draws(args.draws_path)
        # basic safety: ensure grid matches
        if [float(x) for x in df_grid_loaded] != [float(x) for x in df_grid]:
            raise ValueError(f"df_grid mismatch between config and draws_path. config={df_grid}, draws={df_grid_loaded}")
    else:
        if args.mode != "all":
            raise ValueError("--draws_path is required unless mode=all")
        # generate in-memory draws (NOT shared across jobs)
        print("WARNING: no --draws_path provided; generating draws in-memory (not shareable across jobs).")
        draws_by_df = make_and_save_draws(df_grid=df_grid, data=data, draws_path="/tmp/_tmp_draws.pt", seed=args.seed)

    # -------------------- mode: neural --------------------
    if args.mode == "neural":
        if not args.out:
            raise ValueError("--out is required for mode=neural")
        context_len = int(data["context_len"])
        num_bootstraps = int(ev.get("num_bootstraps", 1))

        out_neural = eval_neural(
            train_models=train_models,
            df_grid=df_grid,
            draws_by_df=draws_by_df,
            context_len=context_len,
            num_bootstraps=num_bootstraps,
            device=device,
            df_indices=df_indices,
        )
        out_neural["cfg_fingerprint"] = cfg_fp  
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(out_neural, args.out)
        print(f"✓ Saved neural results to: {args.out}")
        return

    # -------------------- mode: baseline --------------------
    if args.mode == "baseline":
        if not args.out:
            raise ValueError("--out is required for mode=baseline")
        if not args.baseline_method:
            raise ValueError("--baseline_method is required for mode=baseline")

        mcmc = baselines_cfg.get("mcmc", {})
        svi = baselines_cfg.get("svi", {})
        hier = baselines_cfg.get("hier", {})
        mix = baselines_cfg.get("mixture", {})
        hier_t = baselines_cfg.get("hier_t", {})

        hier_prior_mean_min = float(hier.get("prior_mean_min", -8.0))
        hier_prior_mean_max = float(hier.get("prior_mean_max", 8.0))

        mcmc_num_thinned = int(mcmc.get("num_thinned", 200))
        mcmc_warmup = int(mcmc.get("warmup", 1000))
        mcmc_thinning = int(mcmc.get("thinning", 10))
        mcmc_num_chains = int(mcmc.get("num_chains", 1))

        svi_steps = int(svi.get("steps", 1000))
        svi_posterior_samples = int(svi.get("posterior_samples", 200))

        mixture_df_list = list(mix.get("df_list", []))
        tau_eps = float(hier_t.get("tau_eps", 1e-3))

        w_mean = float(data.get("w_mean", 0.0))
        w_std = float(data.get("w_std", 1.0))
        context_len = int(data["context_len"])

        out_base = eval_baseline_method(
            method=args.baseline_method,
            df_grid=df_grid,
            draws_by_df=draws_by_df,
            context_len=context_len,
            w_mean=w_mean,
            w_std=w_std,
            hier_prior_mean_min=hier_prior_mean_min,
            hier_prior_mean_max=hier_prior_mean_max,
            mcmc_num_thinned=mcmc_num_thinned,
            mcmc_warmup=mcmc_warmup,
            mcmc_thinning=mcmc_thinning,
            mcmc_num_chains=mcmc_num_chains,
            svi_steps=svi_steps,
            svi_posterior_samples=svi_posterior_samples,
            mixture_df_list=mixture_df_list,
            tau_eps=tau_eps,
            df_indices=df_indices,
        )
        out_base["cfg_fingerprint"] = cfg_fp  
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(out_base, args.out)
        print(f"✓ Saved baseline({args.baseline_method}) results to: {args.out}")
        return

    # -------------------- mode: merge --------------------
    if args.mode == "merge":
        if not args.out:
            raise ValueError("--out is required for mode=merge")

        if not args.neural_glob:
            raise ValueError("--neural_glob is required for mode=merge (e.g. '.../neural_*.pt')")
        if not args.baseline_glob:
            raise ValueError("--baseline_glob is required for mode=merge (e.g. '.../baseline_*.pt')")

        neural_paths = sorted(glob.glob(args.neural_glob))
        baseline_paths = sorted(glob.glob(args.baseline_glob))
        if not neural_paths:
            raise ValueError(f"No neural shards matched neural_glob={args.neural_glob}")
        if not baseline_paths:
            raise ValueError(f"No baseline shards matched baseline_glob={args.baseline_glob}")

        merge_results(
            cfg_fingerprint=cfg_fp,
            df_grid=df_grid,
            train_models=train_models,
            neural_paths=neural_paths,
            baseline_paths=baseline_paths,
            out_path=args.out,
        )
        return

    # -------------------- mode: all --------------------
    if args.mode == "all":
        output_path = heat["output_path"]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        context_len = int(data["context_len"])
        num_bootstraps = int(ev.get("num_bootstraps", 1))
        w_mean = float(data.get("w_mean", 0.0))
        w_std = float(data.get("w_std", 1.0))

        neural = eval_neural(
            train_models=train_models,
            df_grid=df_grid,
            draws_by_df=draws_by_df,
            context_len=context_len,
            num_bootstraps=num_bootstraps,
            device=device,
            df_indices=list(range(len(df_grid))),
        )

        baseline_rows: Dict[str, Dict[str, List[float]]] = {}
        if bool(baselines_cfg.get("enabled", False)):
            methods = _normalize_methods(list(baselines_cfg.get(
                "methods",
                [
                    "mcmc_oracle",
                    "mcmc_mixture",
                    "mcmc_hier_t",
                    "mcmc_hier_df_mix_mu",
                    "svi_oracle",
                    "svi_mixture",
                    "svi_hier_t",
                    "svi_hier_df_mix_mu",
                ],
            )))

            mcmc = baselines_cfg.get("mcmc", {})
            svi = baselines_cfg.get("svi", {})
            hier = baselines_cfg.get("hier", {})
            mix = baselines_cfg.get("mixture", {})
            hier_t = baselines_cfg.get("hier_t", {})

            hier_prior_mean_min = float(hier.get("prior_mean_min", -8.0))
            hier_prior_mean_max = float(hier.get("prior_mean_max", 8.0))

            mcmc_num_thinned = int(mcmc.get("num_thinned", 200))
            mcmc_warmup = int(mcmc.get("warmup", 1000))
            mcmc_thinning = int(mcmc.get("thinning", 10))
            mcmc_num_chains = int(mcmc.get("num_chains", 1))

            svi_steps = int(svi.get("steps", 1000))
            svi_posterior_samples = int(svi.get("posterior_samples", 200))

            mixture_df_list = list(mix.get("df_list", []))
            tau_eps = float(hier_t.get("tau_eps", 1e-3))

            for m in methods:
                out_m = eval_baseline_method(
                    method=m,
                    df_grid=df_grid,
                    draws_by_df=draws_by_df,
                    context_len=context_len,
                    w_mean=w_mean,
                    w_std=w_std,
                    hier_prior_mean_min=hier_prior_mean_min,
                    hier_prior_mean_max=hier_prior_mean_max,
                    mcmc_num_thinned=mcmc_num_thinned,
                    mcmc_warmup=mcmc_warmup,
                    mcmc_thinning=mcmc_thinning,
                    mcmc_num_chains=mcmc_num_chains,
                    svi_steps=svi_steps,
                    svi_posterior_samples=svi_posterior_samples,
                    mixture_df_list=mixture_df_list,
                    tau_eps=tau_eps,
                    df_indices=list(range(len(df_grid))),
                )
                baseline_rows[m] = {"KL_gt": out_m["KL_gt"], "TV_gt": out_m["TV_gt"]}

        results = {
            "cfg_fingerprint": cfg_fp,
            "df_grid": df_grid,
            "train_models": train_models,
            "data": data,
            "eval": ev,
            "kl_mat": neural["kl_mat"],
            "tv_mat": neural["tv_mat"],
            "baseline_rows": baseline_rows,
        }
        torch.save(results, output_path)
        print(f"✓ Saved all-in-one results to: {output_path}")
        return

    raise ValueError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    main()

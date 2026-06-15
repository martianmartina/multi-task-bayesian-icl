#!/usr/bin/env python

import os
import hashlib 
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.models.multi_task_implicit_model import MultiTaskImplicitInContextLearner
from src.utils.eval_helpers import (
    sample_normal_vec,
    sample_student_t_vec,
    generate_logistic_task,
    bernoulli_kl,
    bernoulli_tv,
    predict_with_bootstrap,
    run_oracle_mcmc_predictive,
    run_oracle_hierarchical_df_mcmc_predictive,
    run_oracle_hierarchical_df_mixture_mu_mcmc_predictive,
    run_svi_curve,
    run_hier_df_svi_curve,
    run_hier_df_mixture_mu_svi_curve,
)

# ----------------------------- small utils -----------------------------

def device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def make_x_query(num_query_points: int, x_dim: int) -> torch.Tensor:
    # previously:
    # return torch.linspace(-1, 1, num_query_points).unsqueeze(1).expand(-1, x_dim)
    # i.i.d. N(0,1) across query points and dimensions
    return torch.randn(num_query_points, x_dim)

def df_to_prior_family(df: float) -> Tuple[str, Optional[float]]:
    if df is None or (isinstance(df, float) and (np.isinf(df) or df >= 1e6)):
        return "normal", None
    return "student_t", float(df)

def sample_w(df: float, w_mean: float, w_std: float, x_dim: int) -> torch.Tensor:
    prior_family, st_df = df_to_prior_family(df)
    if prior_family == "normal":
        return sample_normal_vec(mean_scalar=w_mean, std_scalar=w_std, dim=x_dim)
    return sample_student_t_vec(df=st_df, mean_scalar=w_mean, scale_scalar=w_std, dim=x_dim)

def normalize_methods(methods: List[str]) -> List[str]:
    return [m.strip().lower() for m in methods]

def config_fingerprint(cfg: Dict[str, Any]) -> str:
    payload = yaml.safe_dump(cfg, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

def df_indices_for_shard(n: int, shard_id: int, num_shards: int) -> List[int]:
    if num_shards <= 1:
        return list(range(n))
    return [i for i in range(n) if (i % num_shards) == shard_id]

def draw_indices_for_shard(n: int, shard_id: int, num_shards: int) -> List[int]:
    """
    Indices of draws to compute within a df column.
    Uses the same modulo sharding convention as df_indices_for_shard.
    """
    if num_shards <= 1:
        return list(range(n))
    return [i for i in range(n) if (i % num_shards) == shard_id]

def df_grid_prefix(df_grid: List[float], mixture_row_id: int) -> List[float]:
    """Row r corresponds to prefix df_grid[:r+1].""" 
    if mixture_row_id < 0 or mixture_row_id >= len(df_grid):
        raise ValueError(f"mixture_row_id out of range: {mixture_row_id} (len(df_grid)={len(df_grid)})")
    return [float(x) for x in df_grid[: mixture_row_id + 1]]

# ----------------------------- draw generation -----------------------------

def generate_draw(
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
    w_target = sample_w(df=df, w_mean=w_mean, w_std=w_std, x_dim=x_dim)
    x_pool, y_pool = generate_logistic_task(
        w_target,
        sequence_length=max(sequence_length, context_len),
        noise_std=noise_std,
        x_mean=x_mean,
        x_std=x_std,
    )

    prior_tasks_data = []
    for _ in range(num_prior_tasks):
        w_p = sample_w(df=df, w_mean=w_mean, w_std=w_std, x_dim=x_dim)
        
        x_p, y_p = generate_logistic_task(
            w_p,
            sequence_length=sequence_length,
            noise_std=noise_std,
            x_mean=x_mean,
            x_std=x_std,
        )
        prior_tasks_data.append((x_p, y_p))

    x_query = make_x_query(num_query_points=num_query_points, x_dim=x_dim)
    p_gt = torch.sigmoid(x_query @ w_target)  # [N,1]

    return {
        "df": float(df),
        "w_target": w_target.cpu(),
        "x_pool": x_pool.cpu(),
        "y_pool": y_pool.cpu(),
        "prior_tasks_data": [(xp.cpu(), yp.cpu()) for xp, yp in prior_tasks_data],
        "x_query": x_query.cpu(),
        "p_gt": p_gt.cpu(),
    }

def make_and_save_draws(
    *,
    df_grid: List[float],
    data: Dict[str, Any],
    draws_path: str,
    seed: int,
) -> Dict[float, List[Dict[str, Any]]]:
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
            generate_draw(
                df=float(df),
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
    payload = torch.load(draws_path, map_location="cpu")
    df_grid = [float(x) for x in payload["df_grid"]]
    draws_by_df = payload["draws_by_df"]
    data = payload.get("data", {})
    return df_grid, draws_by_df, data

# ----------------------------- evaluation: neural -----------------------------

def eval_neural_on_draw(
    *,
    model: MultiTaskImplicitInContextLearner,
    draw: Dict[str, Any],
    context_len: int,
    num_bootstraps: int,
) -> torch.Tensor:
    x_pool = draw["x_pool"]
    y_pool = draw["y_pool"]
    prior_tasks_data = draw["prior_tasks_data"]
    x_query = draw["x_query"]

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

    return q_probs.detach().cpu()

def eval_neural(
    *,
    train_models: List[Dict[str, Any]],
    df_grid: List[float],
    draws_by_df: Dict[float, List[Dict[str, Any]]],
    context_len: int,
    num_bootstraps: int,
    device: torch.device,
    df_indices: List[int],
    draw_shard_id: int = 0,
    draw_num_shards: int = 1,
) -> Dict[str, Any]:
    n_rows = len(train_models)
    n_cols = len(df_grid)
    q_probs_mat: List[List[Optional[torch.Tensor]]] = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    for i, row in enumerate(train_models):
        ckpt = row["checkpoint_path"]
        print(f"Loading model [{i+1}/{len(train_models)}]: {ckpt}")
        model = MultiTaskImplicitInContextLearner.load_from_checkpoint(ckpt, map_location="cpu")
        model.eval().to(device)

        for j in df_indices:
            df = float(df_grid[j])
            draws = draws_by_df[df]
            num_draws_total = len(draws)
            draw_indices = draw_indices_for_shard(num_draws_total, int(draw_shard_id), int(draw_num_shards))
            draw_q_probs: List[torch.Tensor] = []
            for k in tqdm(draw_indices, desc=f"neural row {i} df {df}", leave=False):
                draw = draws[k]
                q_probs = eval_neural_on_draw(
                    model=model,
                    draw=draw,
                    context_len=context_len,
                    num_bootstraps=num_bootstraps,
                )
                draw_q_probs.append(q_probs)

            if draw_q_probs:
                q_probs_mat[i][j] = torch.stack(draw_q_probs, dim=0)  # [num_draws, N, 1]

    try:
        _any_df = next(iter(draws_by_df.keys()))
        num_draws_total_global = int(len(draws_by_df[_any_df]))
    except StopIteration:
        num_draws_total_global = 0

    return {
        "df_grid": [float(x) for x in df_grid],
        "train_models": train_models,
        "q_probs_mat": q_probs_mat,
        "computed_df_indices": df_indices,
        "draw_shard_id": int(draw_shard_id),
        "draw_num_shards": int(draw_num_shards),
        "num_draws_total": int(num_draws_total_global),
    }

# ----------------------------- evaluation: baselines -----------------------------

def eval_baseline_method_on_draw(
    *,
    method: str,
    draw: Dict[str, Any],
    context_len: int,
    w_mean: float,
    w_std: float,
    mcmc_num_thinned: int,
    mcmc_num_thinned_oracle: int,
    mcmc_warmup: int,
    mcmc_thinning: int,
    mcmc_num_chains: int,
    svi_steps: int,
    svi_posterior_samples: int,
    mixture_df_list: Optional[List[float]] = None,
    tau_eps: float = 1e-3,
) -> torch.Tensor:
    method = method.strip().lower()

    x_pool = draw["x_pool"]
    y_pool = draw["y_pool"]
    prior_tasks_data = draw["prior_tasks_data"]
    x_query = draw["x_query"]

    x_ctx = x_pool[:context_len]
    y_ctx = y_pool[:context_len]

    prior_family, st_df = df_to_prior_family(draw["df"])

    # ORACLE (knows true df)
    if method == "mcmc_oracle":
        _, q_probs = run_oracle_mcmc_predictive(
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_query=x_query,
            w_mean_oracle=w_mean,
            w_std_oracle=w_std,
            num_thinned=mcmc_num_thinned_oracle,
            warmup=mcmc_warmup,
            thinning=mcmc_thinning,
            num_chains=mcmc_num_chains,
            prior_family=prior_family,
            student_t_df=st_df,
            student_t_df_list=None,
        )
        return q_probs.detach().cpu()

    if method == "svi_oracle":
        q_list = run_svi_curve(
            x_ctx,
            y_ctx,
            x_query,
            prior_mean=w_mean,
            prior_scale=w_std,
            steps_set={mcmc_num_thinned_oracle*mcmc_thinning},
            svi_posterior_samples=int(svi_posterior_samples),
            prior_family=prior_family,
            student_t_df=st_df,
            student_t_df_list=None,
        )
        return q_list[-1].detach().cpu()

    # MIXTURE (meta-prior over df; unknown test df)
    if method == "mcmc_mixture":
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
            student_t_df=None,
            student_t_df_list=[float(x) for x in mixture_df_list],
        )
        return q_probs.detach().cpu()

    if method == "svi_mixture":
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
            student_t_df=None,
            student_t_df_list=[float(x) for x in mixture_df_list],
        )
        return q_list[-1].detach().cpu()

    # HIER-T (infers df via tau AND infers mu; scale=1 fixed)
    if method == "mcmc_hier_t":
        _, q_probs_h, _samples = run_oracle_hierarchical_df_mcmc_predictive(
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
        return q_probs_h.detach().cpu()

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
        return q_list[-1].detach().cpu()

    # HIER-DF-MIX-MU (infers mu in [-8,8] and df from row-wise df-list; shared across tasks)
    if method == "mcmc_hier_df_mix_mu":
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
        return q_probs_h.detach().cpu()

    if method == "svi_hier_df_mix_mu":
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
        return q_list[-1].detach().cpu()

    raise KeyError(f"Unknown baseline method: {method}")

def eval_baseline_method(
    *,
    method: str,
    df_grid: List[float],
    draws_by_df: Dict[float, List[Dict[str, Any]]],
    context_len: int,
    w_mean: float,
    w_std: float,
    mcmc_num_thinned: int,
    mcmc_num_thinned_oracle: int,
    mcmc_warmup: int,
    mcmc_thinning: int,
    mcmc_num_chains: int,
    svi_steps: int,
    svi_posterior_samples: int,
    mixture_df_list: Optional[List[float]],
    tau_eps: float,
    df_indices: List[int],
    mixture_row_id: Optional[int] = None, 
    draw_shard_id: int = 0,
    draw_num_shards: int = 1,
) -> Dict[str, Any]:
    method_l = method.strip().lower()
    eff_mixture_df_list = mixture_df_list
    if method_l in ("mcmc_hier_df_mix_mu", "svi_hier_df_mix_mu"):
        if mixture_row_id is None:
            raise ValueError(f"{method_l} requires --mixture_row_id to build the row-wise df mixture.")
        eff_mixture_df_list = df_grid_prefix(df_grid, mixture_row_id)

    n_cols = len(df_grid)
    q_probs_vec: List[Optional[torch.Tensor]] = [None for _ in range(n_cols)]

    for j in df_indices:
        df = float(df_grid[j])
        draws = draws_by_df[df]
        num_draws_total = len(draws)
        draw_indices = draw_indices_for_shard(num_draws_total, int(draw_shard_id), int(draw_num_shards))
        draw_q_probs: List[torch.Tensor] = []
        for k in tqdm(draw_indices, desc=f"{method_l} df {df}", leave=False):
            draw = draws[k]
            draw_q_probs.append(
                eval_baseline_method_on_draw(
                    method=method_l,
                    draw=draw,
                    context_len=context_len,
                    w_mean=w_mean,
                    w_std=w_std,
                    mcmc_num_thinned=mcmc_num_thinned,
                    mcmc_num_thinned_oracle=mcmc_num_thinned_oracle,
                    mcmc_warmup=mcmc_warmup,
                    mcmc_thinning=mcmc_thinning,
                    mcmc_num_chains=mcmc_num_chains,
                    svi_steps=svi_steps,
                    svi_posterior_samples=svi_posterior_samples,
                    mixture_df_list=eff_mixture_df_list,
                    tau_eps=tau_eps,
                )
            )
        if draw_q_probs:
            q_probs_vec[j] = torch.stack(draw_q_probs, dim=0)  # [num_draws, N, 1]

    try:
        _any_df = next(iter(draws_by_df.keys()))
        num_draws_total_global = int(len(draws_by_df[_any_df]))
    except StopIteration:
        num_draws_total_global = 0

    out = {
        "method": method_l,
        "df_grid": [float(x) for x in df_grid],
        "q_probs_vec": q_probs_vec,
        "computed_df_indices": df_indices,
        "draw_shard_id": int(draw_shard_id),
        "draw_num_shards": int(draw_num_shards),
        "num_draws_total": int(num_draws_total_global),
    }

    if method_l in ("mcmc_mixture", "svi_mixture", "mcmc_hier_df_mix_mu", "svi_hier_df_mix_mu"):
        if mixture_row_id is None:
            raise ValueError(f"{method_l} requires mixture_row_id to be set for row-indexed mixture baselines.")
        out["mixture_row_id"] = int(mixture_row_id)
        out["mixture_df_list"] = [float(x) for x in eff_mixture_df_list]

    return out

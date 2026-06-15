#!/usr/bin/env python
"""
Runner for heatmap evaluation stages.

Modes:
  - make_draws
  - neural
  - baseline 
"""

import argparse
import os
from typing import Any, Dict, List

import torch
import yaml

from eval_heatmap_core import (
    device_from_arg,
    config_fingerprint,
    df_indices_for_shard,
    make_and_save_draws,
    load_draws,
    eval_neural,
    eval_baseline_method,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--mode", type=str, required=True, choices=["make_draws", "neural", "baseline"])
    parser.add_argument("--draws_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out", type=str, required=True)

    # sharding over df columns
    parser.add_argument("--df_shard_id", type=int, default=0)
    parser.add_argument("--df_num_shards", type=int, default=1)

    # sharding over draws (within each df column)
    parser.add_argument("--draw_shard_id", type=int, default=0)
    parser.add_argument("--draw_num_shards", type=int, default=1)

    # baseline selection
    parser.add_argument("--baseline_method", type=str, default="")
    parser.add_argument("--mixture_row_id", type=int, default=-1)
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_fp = config_fingerprint(cfg)
    device = device_from_arg(args.device)

    heat = cfg["heatmap"]
    data = cfg["data"]
    ev = cfg.get("eval", {})
    baselines_cfg = cfg.get("baselines", {})

    df_grid: List[float] = [float(x) for x in heat["df_grid"]]
    train_models: List[Dict[str, Any]] = list(heat["train_models"])

    df_indices = df_indices_for_shard(len(df_grid), args.df_shard_id, args.df_num_shards)
    print(f"Using device: {device}")
    print(f"cfg_fingerprint: {cfg_fp}")
    print(f"df shard: id={args.df_shard_id} / {args.df_num_shards} => {len(df_indices)}/{len(df_grid)} columns")
    print(f"draw shard: id={args.draw_shard_id} / {args.draw_num_shards}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # -------------------- make_draws --------------------
    if args.mode == "make_draws":
        make_and_save_draws(df_grid=df_grid, data=data, draws_path=args.draws_path, seed=args.seed)
        return

    # -------------------- load draws --------------------
    df_grid_loaded, draws_by_df, _data_loaded = load_draws(args.draws_path)
    if [float(x) for x in df_grid_loaded] != [float(x) for x in df_grid]:
        raise ValueError(f"df_grid mismatch between config and draws_path. config={df_grid}, draws={df_grid_loaded}")

    # -------------------- neural --------------------
    if args.mode == "neural":
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
            draw_shard_id=int(args.draw_shard_id),
            draw_num_shards=int(args.draw_num_shards),
        )
        out_neural["cfg_fingerprint"] = cfg_fp  
        torch.save(out_neural, args.out)
        print(f"✓ Saved neural shard to: {args.out}")
        return

    # -------------------- baseline (one method) --------------------
    if args.mode == "baseline":
        if not args.baseline_method:
            raise ValueError("--baseline_method required for mode=baseline")

        method = args.baseline_method.strip().lower()

        # baseline hyperparams from config
        mcmc = baselines_cfg.get("mcmc", {})
        svi = baselines_cfg.get("svi", {})
        mix = baselines_cfg.get("mixture", {})
        hier_t = baselines_cfg.get("hier_t", {})

        mcmc_num_thinned = int(mcmc.get("num_thinned", 100))
        mcmc_num_thinned_oracle = int(mcmc.get("num_thinned_oracle", 1000))
        mcmc_warmup = int(mcmc.get("warmup", 1000))
        mcmc_thinning = int(mcmc.get("thinning", 10))
        mcmc_num_chains = int(mcmc.get("num_chains", 1))

        svi_steps = int(svi.get("steps", 1000))
        svi_posterior_samples = int(svi.get("posterior_samples", 200))

        # >>> NOTE: mixture_df_list is now only a fallback; for mixture methods we use --mixture_row_id
        mixture_df_list = list(mix.get("df_list", []))
        tau_eps = float(hier_t.get("tau_eps", 1e-3))

        w_mean = float(data.get("w_mean", 0.0))
        w_std = float(data.get("w_std", 1.0))
        context_len = int(data["context_len"])

        mixture_row_id = None
        if method in ("mcmc_mixture", "svi_mixture", "mcmc_hier_df_mix_mu", "svi_hier_df_mix_mu"):
            if args.mixture_row_id < 0:
                raise ValueError(f"{method} requires --mixture_row_id >= 0")
            mixture_row_id = int(args.mixture_row_id)

        out_base = eval_baseline_method(
            method=method,
            df_grid=df_grid,
            draws_by_df=draws_by_df,
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
            mixture_df_list=mixture_df_list,
            tau_eps=tau_eps,
            df_indices=df_indices,
            mixture_row_id=mixture_row_id,  
            draw_shard_id=int(args.draw_shard_id),
            draw_num_shards=int(args.draw_num_shards),
        )
        out_base["cfg_fingerprint"] = cfg_fp  
        torch.save(out_base, args.out)
        print(f"✓ Saved baseline shard to: {args.out}")
        return

    raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()

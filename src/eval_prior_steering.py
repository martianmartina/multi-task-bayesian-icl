#!/usr/bin/env python
import argparse
import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.multi_task_implicit_model import MultiTaskImplicitInContextLearner
from utils.eval_helpers import (
    sample_normal_vec,
    generate_logistic_task,
    predict_with_bootstrap,
    run_oracle_mcmc_predictive,
    run_oracle_hierarchical_mcmc_predictive,
)
from utils.my_pyro_models import _hier_model_pooled_eval

# ----------------------------- small utils -----------------------------

def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _sample_w_normal(mean: float, std: float, x_dim: int) -> torch.Tensor:
    # returns [D, 1]
    return sample_normal_vec(mean_scalar=float(mean), std_scalar=float(std), dim=int(x_dim))


def _sample_x_query_normal(
    *,
    num_x_samples: int,
    x_dim: int,
    x_mean: float = 0.0,
    x_std: float = 1.0,
    device: torch.device,
) -> torch.Tensor:
    # [N, D]
    x = torch.randn(int(num_x_samples), int(x_dim), device=device)
    x = x * float(x_std) + float(x_mean)
    return x


# ----------------------------- data generation -----------------------------

def _generate_prior_tasks(
    *,
    num_prior_tasks: int,
    sequence_length: int,
    x_dim: int,
    prior_mean: float,
    prior_std: float,
    noise_std: float,
    x_mean: float,
    x_std: float,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate prior tasks: list of (x_p, y_p) on CPU."""
    prior_tasks_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(int(num_prior_tasks)):
        w_p = _sample_w_normal(prior_mean, prior_std, x_dim)  # [D,1]
        x_p, y_p = generate_logistic_task(
            w_p,
            sequence_length=int(sequence_length),
            noise_std=float(noise_std),
            x_mean=float(x_mean),
            x_std=float(x_std),
        )
        prior_tasks_data.append((x_p.cpu(), y_p.cpu()))
    return prior_tasks_data


def _generate_fixed_target(
    *,
    target_mean: float,
    target_std: float,
    x_dim: int,
    sequence_length: int,
    context_len: int,
    noise_std: float,
    x_mean: float,
    x_std: float,
) -> Dict[str, Any]:
    """
    Generate ONE fixed target task + pool data.
    We'll slice the first context_len points as the fixed D_t.
    """
    w_target = _sample_w_normal(target_mean, target_std, x_dim)  # [D,1]
    x_pool, y_pool = generate_logistic_task(
        w_target,
        sequence_length=int(max(sequence_length, context_len)),
        noise_std=float(noise_std),
        x_mean=float(x_mean),
        x_std=float(x_std),
    )
    return {
        "w_target": w_target.cpu(),
        "x_pool": x_pool.cpu(),
        "y_pool": y_pool.cpu(),
    }

# ----------------------------- evaluation -----------------------------

@torch.no_grad()
def _predict_logits_for_xquery(
    *,
    model: MultiTaskImplicitInContextLearner,
    prior_tasks_data: List[Tuple[torch.Tensor, torch.Tensor]],
    x_ctx: torch.Tensor,   # CPU
    y_ctx: torch.Tensor,   # CPU
    x_query: torch.Tensor, # on device
    num_bootstraps: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns:
      logits: [N] on CPU (bootstrapped mean logits per x_query row).
    Notes:
      - Processes x_query in chunks to avoid GPU OOM.
      - Uses predict_with_bootstrap which returns [B, N, 1] logits.
    """
    N = x_query.size(0)
    out_logits: List[torch.Tensor] = []
    use_task_ids = (getattr(model.hparams, "identity_dim", 0) > 0)

    x_ctx_dev = x_ctx.to(device)
    y_ctx_dev = y_ctx.to(device)

    for s in range(0, N, int(chunk_size)):
        e = min(N, s + int(chunk_size))
        xq = x_query[s:e]  # [chunk, D] on device

        logits_boot = predict_with_bootstrap(
            model,
            prior_tasks=prior_tasks_data,              
            x_target_context_pool=x_ctx_dev,           # device
            y_target_context_pool=y_ctx_dev,           # device
            x_target_query=xq,                         # device
            use_task_ids=use_task_ids,
            num_bootstraps=int(num_bootstraps),
        )  # [B, chunk, 1] (or [B, chunk, 1, 1] depending on helper)

        logits_mean = logits_boot.mean(dim=0)
        # Be robust to extra singleton dims.        
        logits_mean = logits_mean.reshape(-1)   # always shape [chunk], even if chunk=1
        out_logits.append(logits_mean.detach().cpu())

    return torch.cat(out_logits, dim=0)  # [N] CPU


def _plot_overlaid_histograms(
    *,
    logits_per_name: Dict[str, np.ndarray],
    title: str,
    out_path: str,
    bins: int = 80,
    density: bool = True,
) -> None:
    plt.figure(figsize=(9, 5))

    # Choose a common range so bins align across priors.
    all_vals = np.concatenate([v for v in logits_per_name.values() if v.size > 0], axis=0)
    if all_vals.size == 0:
        raise RuntimeError("No logits collected; nothing to plot.")
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    if np.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0

    for name, arr in logits_per_name.items():
        if '-' in name:
            continue
        if arr.size == 0:
            continue
        plt.hist(
            arr,
            bins=int(bins),
            range=(lo, hi),
            density=density,
            histtype='step',
            linewidth=2,
            label="{} (logit $\sigma$={:.2f})".format(
                name.replace('mu', r'$\mu$').replace('std', r'$\sigma$'), arr.std()
            ),
        )


    plt.axvline(0.0, linewidth=1, linestyle="dashed")
    plt.xlabel("logit", fontsize=16)
    plt.ylabel("density" if density else "count", fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"✓ Saved plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config used for evaluation+plotting. Not required if --load_logits_ckpt is set.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--load_logits_ckpt",
        type=str,
        default=None,
        help="Path to a saved logits checkpoint (*.pt) produced by this script; if set, skip evaluation and only plot.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional override for output directory (plot-only mode or eval mode).",
    )

    # histogram settings
    parser.add_argument("--num_x_samples", type=int, default=100000, help="x_query samples per target draw")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--density", action="store_true", help="plot density instead of counts")
    parser.add_argument("--chunk_size", type=int, default=2048, help="x_query chunk size for model calls")

    args = parser.parse_args()

    # ------------------ plot-only mode: load logits checkpoint ------------------
    if args.load_logits_ckpt is not None:
        ckpt_path = args.load_logits_ckpt
        payload = torch.load(ckpt_path, map_location="cpu")

        logits_per_name_t = payload.get("logits_per_name", {})
        if not isinstance(logits_per_name_t, dict) or len(logits_per_name_t) == 0:
            raise RuntimeError(f"Checkpoint {ckpt_path} missing non-empty 'logits_per_name'.")

        logits_per_name: Dict[str, np.ndarray] = {}
        for k, v in logits_per_name_t.items():
            if torch.is_tensor(v):
                logits_per_name[str(k)] = v.detach().cpu().numpy()
            else:
                logits_per_name[str(k)] = np.asarray(v)

        cfg = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
        ctx_len = int(cfg.get("data", {}).get("context_len", -1))
        num_x = int(payload.get("num_x_samples_per_draw", -1))

        out_dir = args.out_dir or cfg.get("out_dir") or os.path.dirname(os.path.abspath(ckpt_path)) or "."
        os.makedirs(out_dir, exist_ok=True)

        stem = os.path.basename(ckpt_path)
        if stem.endswith(".pt"):
            stem = stem[:-3]
        out_png = os.path.join(out_dir, f"{stem}.png")

        title = f"Logit histograms (loaded), num_x_query={num_x if num_x > 0 else 'unknown'}, ctx_len={ctx_len if ctx_len >= 0 else 'unknown'}"
        _plot_overlaid_histograms(
            logits_per_name=logits_per_name,
            title=title,
            out_path=out_png,
            bins=args.bins,
            density=args.density,
        )
        print("✓ Done (plot-only).")
        return

    if args.config is None:
        raise ValueError("Either --config must be provided (to run evaluation) or --load_logits_ckpt (to plot-only).")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = _device_from_arg(args.device)
    print(f"Using device: {device}")

    ckpt_path = cfg["checkpoint_path"]
    out_dir = args.out_dir or cfg.get("out_dir", "results/prior_steering")
    os.makedirs(out_dir, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    _set_seed(seed)

    data = cfg["data"]
    x_dim = int(data["x_dim"])
    num_prior_tasks = int(data["num_prior_tasks"])
    sequence_length = int(data["sequence_length"])
    context_len = int(data["context_len"])

    noise_std = float(data.get("noise_std", 0.0))
    x_mean = float(data.get("x_mean", 0.0))
    x_std = float(data.get("x_std", 1.0))

    num_bootstraps = int(cfg.get("num_bootstraps", 1))

    # target draw count
    test_draws = int(cfg.get("test_draws", 1)) # test draw is always 1 in this script
    print(f"test_draws={test_draws}, num_x_samples={args.num_x_samples}, num_bootstraps={num_bootstraps}")

    # target prior for generating w_target
    target_prior = cfg.get("target_prior", {"mean": 0.0, "std": 1.0})
    tgt_mean = float(target_prior.get("mean", 0.0))
    tgt_std = float(target_prior.get("std", 1.0))

    # Load model
    print(f"Loading checkpoint: {ckpt_path}")
    model = MultiTaskImplicitInContextLearner.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval().to(device)

    # Base + steered priors
    base_prior = cfg.get("base_prior", {"name": "base(mu=0,std=1)", "mean": 0.0, "std": 1.0})
    base_name = str(base_prior.get("name", "base"))
    base_mean = float(base_prior.get("mean", 0.0))
    base_std = float(base_prior.get("std", 1.0))

    steered_priors = list(cfg.get("steered_priors", []))
    include_no_prior = bool(cfg.get("include_no_prior", False))

    # Pre-generate prior prefixes ONCE
    print("Generating prior-task prefixes...")
    prior_prefixes: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    if include_no_prior:
        prior_prefixes["no_prior"] = []


    # generate one fixed prior prefix for base prior
    prior_prefixes[base_name] = _generate_prior_tasks(
        num_prior_tasks=num_prior_tasks,
        sequence_length=sequence_length,
        x_dim=x_dim,
        prior_mean=base_mean,
        prior_std=base_std,
        noise_std=noise_std,
        x_mean=x_mean,
        x_std=x_std,
    )

    # generate one fixed prior prefix for each steered prior
    for sp in steered_priors:
        name = str(sp.get("name", f"mu={sp.get('mean', 0.0)},std={sp.get('std', 1.0)}"))
        mu = float(sp.get("mean", 0.0))
        sd = float(sp.get("std", 1.0))
        prior_prefixes[name] = _generate_prior_tasks(
            num_prior_tasks=num_prior_tasks,
            sequence_length=sequence_length,
            x_dim=x_dim,
            prior_mean=mu,
            prior_std=sd,
            noise_std=noise_std,
            x_mean=x_mean,
            x_std=x_std,
        )

    names = list(prior_prefixes.keys())
    print(f"Priors: {names}")

    logits_accum: Dict[str, List[np.ndarray]] = {n: [] for n in names}
    
    # generate one fixed x_query for every draw
    x_query = _sample_x_query_normal(
            num_x_samples=args.num_x_samples,
            x_dim=x_dim,
            x_mean=0.0,
            x_std=1.0,
            device=device,
        )
    
    # generate one fixed target task for every draw
    fixed = _generate_fixed_target(
            target_mean=tgt_mean,
            target_std=tgt_std,
            x_dim=x_dim,
            sequence_length=sequence_length,
            context_len=context_len,
            noise_std=noise_std,
            x_mean=x_mean,
            x_std=x_std,
        )

    x_pool = fixed["x_pool"]
    y_pool = fixed["y_pool"]
    x_ctx = x_pool[:context_len]  # CPU
    y_ctx = y_pool[:context_len]  # CPU


    for name in names:
        logits = _predict_logits_for_xquery(
            model=model,
            prior_tasks_data=prior_prefixes[name],
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            x_query=x_query,
            num_bootstraps=num_bootstraps,
            chunk_size=args.chunk_size,
            device=device,
        )  # [N] CPU
        logits_accum[name].append(logits.numpy())

    # Concatenate across draws: each prior -> [test_draws * num_x_samples]
    logits_per_name: Dict[str, np.ndarray] = {
        name: (np.concatenate(parts, axis=0) if len(parts) > 0 else np.array([], dtype=np.float32))
        for name, parts in logits_accum.items()
    }

    # Save raw logits
    save_path = os.path.join(out_dir, f"logits_hist_{args.num_x_samples}xq_{context_len}ctx.pt")
    torch.save(
        {
            "config": cfg,
            "seed": seed,
            "checkpoint_path": ckpt_path,
            "x_query_dist": {"mean": 0.0, "std": 1.0, "dim": x_dim},
            "test_draws": test_draws,
            "num_x_samples_per_draw": int(args.num_x_samples),
            "num_bootstraps": int(num_bootstraps),
            "logits_per_name": {k: torch.from_numpy(v) for k, v in logits_per_name.items()},
        },
        save_path,
    )
    print(f"✓ Saved logits: {save_path}")

    # Plot
    out_png = os.path.join(out_dir, f"logits_hist_{args.num_x_samples}xq_{context_len}ctx.png")
    _plot_overlaid_histograms(
        logits_per_name=logits_per_name,
        title=f"Logit histograms over x_query~N(0,I), num_x_query={args.num_x_samples}, ctx_len={context_len}",
        out_path=out_png,
        bins=args.bins,
        density=args.density,
    )

    print("✓ Done.")


if __name__ == "__main__":
    main()

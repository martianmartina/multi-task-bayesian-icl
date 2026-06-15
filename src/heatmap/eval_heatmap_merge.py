#!/usr/bin/env python
import argparse
import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from eval_heatmap_core import config_fingerprint
from src.utils.eval_helpers import bernoulli_kl, bernoulli_tv

def _nan_merge(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    mask = np.isnan(dst) & ~np.isnan(src)
    dst[mask] = src[mask]
    return dst

def _fmt_df(df: float) -> str:
    return "inf" if (df >= 1e6 or np.isinf(df)) else str(df)

def _draw_meta_from_shard(shard: Dict[str, Any], q: torch.Tensor) -> tuple[list[int], int]:
    """
    Return (draw_indices, num_draws_total) for a shard.
    Backwards compatible with older shards that did not include draw sharding metadata.
    """
    if "draw_shard_id" in shard and "draw_num_shards" in shard and "num_draws_total" in shard:
        draw_shard_id = int(shard["draw_shard_id"])
        draw_num_shards = int(shard["draw_num_shards"])
        nd = int(shard["num_draws_total"])
        di = [i for i in range(nd)] if draw_num_shards <= 1 else [i for i in range(nd) if (i % draw_num_shards) == draw_shard_id]
        return di, nd

    if "draw_indices" in shard and "num_draws_total" in shard:
        di = [int(x) for x in shard["draw_indices"]]
        nd = int(shard["num_draws_total"])
        return di, nd

    nd = int(q.shape[0])
    return list(range(nd)), nd

def _merge_draw_sharded_q(
    *,
    cur: Optional[torch.Tensor],
    q_shard: torch.Tensor,
    draw_indices: list[int],
    num_draws_total: int,
    where: str,
) -> torch.Tensor:
    """
    Merge a partial q-probs shard into a full [num_draws_total, N, 1] tensor.
    """
    if not torch.is_tensor(q_shard):
        raise TypeError(f"{where}: q_shard must be a torch tensor")
    if q_shard.ndim != 3:
        raise ValueError(f"{where}: expected q_shard ndim=3, got shape={tuple(q_shard.shape)}")
    if len(draw_indices) != int(q_shard.shape[0]):
        raise ValueError(
            f"{where}: draw_indices length {len(draw_indices)} does not match "
            f"q_shard draws {int(q_shard.shape[0])}"
        )

    if cur is None:
        cur = torch.full(
            (int(num_draws_total), int(q_shard.shape[1]), int(q_shard.shape[2])),
            float("nan"),
            dtype=q_shard.dtype,
            device="cpu",
        )
    else:
        if cur.shape[1:] != q_shard.shape[1:]:
            raise ValueError(f"{where}: shape mismatch cur={tuple(cur.shape)} vs shard={tuple(q_shard.shape)}")
        if int(cur.shape[0]) != int(num_draws_total):
            raise ValueError(
                f"{where}: num_draws_total mismatch cur={int(cur.shape[0])} vs shard_meta={int(num_draws_total)}"
            )

    existing = cur[draw_indices]
    if torch.isfinite(existing).any():
        dup = torch.isfinite(existing).reshape(existing.shape[0], -1).any(dim=1).nonzero(as_tuple=False).flatten()
        if dup.numel() > 0:
            dup_draws = [draw_indices[int(k)] for k in dup.tolist()]
            raise ValueError(f"{where}: duplicate draw indices encountered: {dup_draws}")

    cur[draw_indices] = q_shard.cpu()
    return cur

def _assert_no_nan(q: torch.Tensor, where: str) -> None:
    if torch.isnan(q).any():
        bad = torch.isnan(q).reshape(q.shape[0], -1).any(dim=1).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"{where}: missing draws after merge (NaNs present) at draw indices: {bad}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--neural_glob", type=str, required=True)
    parser.add_argument(
        "--baseline_glob",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more glob patterns for baseline shard files. "
            "Example: --baseline_glob 'baseline_svi_*' 'baseline_mcmc_oracle_*'"
        ),
    )
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg_fp = config_fingerprint(cfg)

    heat = cfg["heatmap"]
    df_grid: List[float] = [float(x) for x in heat["df_grid"]]
    train_models: List[Dict[str, Any]] = list(heat["train_models"])

    neural_paths = sorted(glob.glob(args.neural_glob))
    baseline_patterns = list(args.baseline_glob)
    baseline_paths = sorted({p for pat in baseline_patterns for p in glob.glob(pat)})

    if not neural_paths:
        raise ValueError(f"No neural shards matched: {args.neural_glob}")
    if not baseline_paths:
        raise ValueError(f"No baseline shards matched any of: {baseline_patterns}")

    n_rows = len(train_models)
    n_cols = len(df_grid)

    # -------------------- merge neural --------------------
    neural_q_probs: List[List[Optional[torch.Tensor]]] = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    for p in neural_paths:
        shard = torch.load(p, map_location="cpu")
        if shard.get("cfg_fingerprint") != cfg_fp:
            pass

        if "q_probs_mat" not in shard:
            raise KeyError(f"Neural shard missing q_probs_mat: {p}. Keys={list(shard.keys())}")
        q_mat = shard["q_probs_mat"]
        if len(q_mat) != n_rows:
            raise ValueError(f"Neural shard row mismatch: {p} has {len(q_mat)} rows, expected {n_rows}")
        for i in range(n_rows):
            if len(q_mat[i]) != n_cols:
                raise ValueError(f"Neural shard col mismatch: {p} row {i} has {len(q_mat[i])} cols, expected {n_cols}")
            for j in range(n_cols):
                q = q_mat[i][j]
                if q is None:
                    continue
                draw_indices, num_draws_total = _draw_meta_from_shard(shard, q)
                neural_q_probs[i][j] = _merge_draw_sharded_q(
                    cur=neural_q_probs[i][j],
                    q_shard=q,
                    draw_indices=draw_indices,
                    num_draws_total=num_draws_total,
                    where=f"neural cell (row={i}, col={j}) shard={p}",
                )

    # -------------------- merge baselines --------------------
    baseline_q_probs: Dict[str, List[Optional[torch.Tensor]]] = {}
    mixture_q_probs: Dict[str, Dict[str, Any]] = {}

    _MIXTURE_ROW_METHODS = {
        "mcmc_mixture",
        "svi_mixture",
        "mcmc_hier_df_mix_mu",
        "svi_hier_df_mix_mu",
    }

    for p in baseline_paths:
        shard = torch.load(p, map_location="cpu")

        method = shard["method"]
        if "q_probs_vec" not in shard:
            raise KeyError(f"Baseline shard missing q_probs_vec: {p}. Keys={list(shard.keys())}")
        q_vec = shard["q_probs_vec"]
        if len(q_vec) != n_cols:
            raise ValueError(f"Baseline shard col mismatch: {p} has {len(q_vec)} cols, expected {n_cols}")

        mix_row: Optional[int] = None
        if "mixture_row_id" in shard:
            mix_row = int(shard["mixture_row_id"])
        elif str(method).strip().lower() in _MIXTURE_ROW_METHODS:
            m = re.search(r"mixrow(\d+)", os.path.basename(p))
            if m:
                mix_row = int(m.group(1))

        if mix_row is not None:
            r = int(mix_row)
            if method not in mixture_q_probs:
                mixture_q_probs[method] = {
                    "q_probs": [[None for _ in range(n_cols)] for _ in range(n_rows)],
                    "row_mixtures": [None for _ in range(n_rows)],
                }
            if r < 0 or r >= n_rows:
                raise ValueError(f"mixture_row_id {r} out of range for method={method} (n_rows={n_rows})")

            curQ = mixture_q_probs[method]["q_probs"]
            for j in range(n_cols):
                q = q_vec[j]
                if q is None:
                    continue
                draw_indices, num_draws_total = _draw_meta_from_shard(shard, q)
                curQ[r][j] = _merge_draw_sharded_q(
                    cur=curQ[r][j],
                    q_shard=q,
                    draw_indices=draw_indices,
                    num_draws_total=num_draws_total,
                    where=f"mixture method={method} row={r} col={j} shard={p}",
                )
            mixture_q_probs[method]["row_mixtures"][r] = shard.get("mixture_df_list", None)
        else:
            if method not in baseline_q_probs:
                baseline_q_probs[method] = [None for _ in range(n_cols)]
            curQ = baseline_q_probs[method]
            for j in range(n_cols):
                q = q_vec[j]
                if q is None:
                    continue
                draw_indices, num_draws_total = _draw_meta_from_shard(shard, q)
                curQ[j] = _merge_draw_sharded_q(
                    cur=curQ[j],
                    q_shard=q,
                    draw_indices=draw_indices,
                    num_draws_total=num_draws_total,
                    where=f"baseline method={method} col={j} shard={p}",
                )

    for method, pack in mixture_q_probs.items():
        row_mixtures = pack["row_mixtures"]
        for r in range(n_rows):
            if row_mixtures[r] is None:
                row_mixtures[r] = [float(x) for x in df_grid[: r + 1]]
        pack["row_mixtures"] = row_mixtures

    # -------------------- compute KL/TV to oracle --------------------
    if "mcmc_oracle" not in baseline_q_probs:
        raise ValueError(
            "Missing required baseline method 'mcmc_oracle' q_probs. "
            "Run baseline shards for --baseline_method mcmc_oracle for all df columns."
        )
    oracle_q = baseline_q_probs["mcmc_oracle"]
    missing_oracle = [j for j in range(n_cols) if oracle_q[j] is None]
    if missing_oracle:
        missing_vals = [df_grid[j] for j in missing_oracle]
        raise ValueError(f"mcmc_oracle is missing q_probs for df_grid columns: {missing_vals}")
    for j in range(n_cols):
        _assert_no_nan(oracle_q[j], where=f"mcmc_oracle col={j} df={df_grid[j]}")

    def _kl_tv_to_oracle(q_method: torch.Tensor, q_oracle: torch.Tensor) -> Tuple[float, float]:
        if not (torch.is_tensor(q_method) and torch.is_tensor(q_oracle)):
            raise TypeError("q_method and q_oracle must be torch tensors")
        if q_method.shape != q_oracle.shape:
            raise ValueError(f"Shape mismatch: method={tuple(q_method.shape)} vs oracle={tuple(q_oracle.shape)}")
        kl = float(bernoulli_kl(q_oracle, q_method).item())
        tv = float(bernoulli_tv(q_oracle, q_method).item())
        return kl, tv

    neural_KL = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    neural_TV = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            q = neural_q_probs[i][j]
            if q is None:
                continue
            _assert_no_nan(q, where=f"neural cell (row={i}, col={j}) df={df_grid[j]}")
            kl, tv = _kl_tv_to_oracle(q, oracle_q[j])
            neural_KL[i, j] = kl
            neural_TV[i, j] = tv

    baseline_vecs: Dict[str, Dict[str, List[float]]] = {}
    for method, qvec in baseline_q_probs.items():
        kl_list = [float("nan")] * n_cols
        tv_list = [float("nan")] * n_cols
        for j in range(n_cols):
            q = qvec[j]
            if q is None:
                continue
            _assert_no_nan(q, where=f"baseline method={method} col={j} df={df_grid[j]}")
            if method == "mcmc_oracle":
                kl_list[j] = 0.0
                tv_list[j] = 0.0
            else:
                kl, tv = _kl_tv_to_oracle(q, oracle_q[j])
                kl_list[j] = kl
                tv_list[j] = tv
        baseline_vecs[method] = {"KL": kl_list, "TV": tv_list}

    mixture_mats: Dict[str, Dict[str, Any]] = {}
    for method, pack in mixture_q_probs.items():
        kl_mat = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        tv_mat = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        qmat = pack["q_probs"]
        for r in range(n_rows):
            for j in range(n_cols):
                q = qmat[r][j]
                if q is None:
                    continue
                _assert_no_nan(q, where=f"mixture method={method} row={r} col={j} df={df_grid[j]}")
                kl, tv = _kl_tv_to_oracle(q, oracle_q[j])
                kl_mat[r, j] = kl
                tv_mat[r, j] = tv
        mixture_mats[method] = {
            "KL": kl_mat,
            "TV": tv_mat,
            "row_mixtures": pack.get("row_mixtures", None),
        }

    final = {
        "cfg_fingerprint": cfg_fp,
        "df_grid": df_grid,
        "df_grid_str": [_fmt_df(x) for x in df_grid],
        "train_models": train_models,
        "neural": {
            "KL": neural_KL,
            "TV": neural_TV,
        },
        "baselines": {
            "vectors": baseline_vecs,   
            "mixtures": mixture_mats,   
        },
        "neural_paths": neural_paths,
        "baseline_paths": baseline_paths,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(final, args.out)
    print(f"✓ Merged results saved to: {args.out}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------- log(df) helpers -----------------------------

def _logdf_from_linear_df(df: float) -> float:
    """Natural log by default (matches exp(3)=20.0855)."""
    df = float(df)
    if not np.isfinite(df):
        return float("inf") if df > 0 else float("nan")
    if df <= 0:
        return float("nan")
    return float(np.log(df)) 


def _fmt_logdf(x: float) -> str:
    """Pretty formatting for log(df) axis labels."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan" if (x is None or np.isnan(x)) else "inf"
    if abs(x - round(x)) < 1e-2:
        return str(int(round(x)))
    return f"{x:.2f}"


def _train_min_logdf_from_row(r: Dict[str, Any]) -> float:
    """
    Robustly interpret train min-df field:
      - Prefer explicit log fields if present
      - Else use train_min_df
        * if it looks like an integer in [-50,50], treat as log(df)
        * otherwise treat as linear df and take log()
    """
    if not isinstance(r, dict):
        return float("nan")

    for k in ("train_min_logdf", "train_min_log_df", "min_logdf", "min_log_df"):
        if k in r:
            try:
                return float(r[k])
            except Exception:
                return float("nan")

    if "train_min_df" in r:
        try:
            v = float(r["train_min_df"])
        except Exception:
            return float("nan")

        if np.isfinite(v) and abs(v - round(v)) < 1e-6 and (-50 <= v <= 50):
            return float(v)

        return _logdf_from_linear_df(v)

    return float("nan")


def _row_labels_from_train_models(train_models: List[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    for r in train_models:
        logdf = _train_min_logdf_from_row(r)
        labels.append(f"{_fmt_logdf(logdf)}")
    return labels


def _col_labels_from_df_grid(df_grid: List[float]) -> List[str]:
    return [_fmt_logdf(_logdf_from_linear_df(df)) for df in df_grid]


# ----------------------------- formatting helpers -----------------------------

def _method_label(m: str) -> str:
    return m.replace("_", "-")


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _maybe_2d(arr: Any) -> np.ndarray:
    a = _to_np(arr)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a


def _is_matrix_like(a: Any) -> bool:
    aa = _to_np(a)
    return aa.ndim == 2 and aa.shape[0] > 1 and aa.shape[1] > 1


# ----------------------------- plotting core -----------------------------

def _annotate_cells(ax: plt.Axes, mat: np.ndarray) -> None:
    """Write values in each cell in white text (2 decimals)."""
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)


def _white_grid(ax: plt.Axes, nrows: int, ncols: int) -> None:
    """Add white boundaries between cells."""
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)


def _plot_heatmap(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_path: str,
    *,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = False,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    mat = np.array(mat, dtype=np.float64)
    nrows, ncols = mat.shape

    fig_w = max(6.0, 0.85 * ncols + 3.0)
    fig_h = max(4.5, 0.60 * nrows + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if center_zero:
        finite = mat[np.isfinite(mat)]
        if finite.size == 0:
            vv = 1.0
        else:
            vv = float(np.max(np.abs(finite)))
            vv = max(vv, 1e-9)
        vmin = -vv
        vmax = vv

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(row_labels, fontsize=14)

    ax.set_xlabel(r"$\log(df)_{\mathrm{test}}$", fontsize=18)
    ax.set_ylabel(r"$\min\,\log(df)_{\mathrm{train}}$", fontsize=18)

    _white_grid(ax, nrows, ncols)
    _annotate_cells(ax, mat)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout(pad=0.75, w_pad=0.35, h_pad=0.35)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


# ----------------------------- data extraction -----------------------------

def _get_baseline_entry(baseline_rows: Dict[str, Any], method: str) -> Optional[Dict[str, Any]]:
    if not baseline_rows:
        return None
    if method in baseline_rows:
        return baseline_rows[method]
    for k in baseline_rows.keys():
        if k.strip().lower() == method.strip().lower():
            return baseline_rows[k]
    return None


def _extract_metric(entry: Dict[str, Any], metric: str) -> Optional[Any]:
    """
    metric: "KL" or "TV"
    Supported keys:
      - "KL_gt"/"TV_gt" (stripe or matrix)
      - "kl_mat"/"tv_mat" or "KL_mat"/"TV_mat"
      - "KL"/"TV" (new merged schema)
    """
    if entry is None:
        return None
    if metric.upper() == "KL":
        for k in ["KL_mat", "kl_mat", "KL_gt", "kl_gt", "KL", "kl"]:
            if k in entry:
                return entry[k]
    else:
        for k in ["TV_mat", "tv_mat", "TV_gt", "tv_gt", "TV", "tv"]:
            if k in entry:
                return entry[k]
    return None


def _get_neural_metric(payload: Dict[str, Any], metric: str) -> Any:
    """
    Support both schemas:
      - old: payload["kl_mat"], payload["tv_mat"]
      - new: payload["neural"]["KL"], payload["neural"]["TV"]
    """
    m = metric.upper()
    if "neural" in payload and isinstance(payload["neural"], dict) and m in payload["neural"]:
        return payload["neural"][m]
    if m == "KL":
        for k in ["kl_mat", "KL_mat", "KL", "kl"]:
            if k in payload:
                return payload[k]
    else:
        for k in ["tv_mat", "TV_mat", "TV", "tv"]:
            if k in payload:
                return payload[k]
    raise KeyError(f"Could not find neural metric {metric} in payload. Keys={list(payload.keys())}")


def _baseline_rows_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize baseline storage into the structure expected by this plotting script.

    Supports:
      - old: payload["baseline_rows"]
      - new: payload["baselines"]["vectors"] and payload["baselines"]["mixtures"]
    """
    if "baseline_rows" in payload and payload["baseline_rows"] is not None:
        return payload["baseline_rows"]

    baselines = payload.get("baselines", None)
    if not isinstance(baselines, dict):
        return {}

    out: Dict[str, Any] = {}

    vecs = baselines.get("vectors", {}) or {}
    if isinstance(vecs, dict):
        for method, pack in vecs.items():
            if not isinstance(pack, dict):
                continue
            out[method] = {
                "KL_gt": pack.get("KL", None),
                "TV_gt": pack.get("TV", None),
            }

    mixes = baselines.get("mixtures", {}) or {}
    if isinstance(mixes, dict):
        for method, pack in mixes.items():
            if not isinstance(pack, dict):
                continue
            out[method] = {
                "KL_mat": pack.get("KL", None),
                "TV_mat": pack.get("TV", None),
                "row_labels": pack.get("row_labels", None),  # optional
            }

    return out


# ----------------------------- main -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True, help="Merged results .pt file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save plots.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    payload = torch.load(args.in_path, map_location="cpu")

    df_grid = [float(x) for x in payload["df_grid"]]
    col_labels = _col_labels_from_df_grid(df_grid)

    train_models = list(payload.get("train_models", []))
    neural_row_labels = _row_labels_from_train_models(train_models)

    kl_neural = _to_np(_get_neural_metric(payload, "KL")).astype(np.float64)
    tv_neural = _to_np(_get_neural_metric(payload, "TV")).astype(np.float64)

    baseline_rows: Dict[str, Any] = _baseline_rows_from_payload(payload) or {}

    _plot_heatmap(
        kl_neural,
        neural_row_labels,
        col_labels,
        title="Neural ICL: KL (oracle || pred)",
        out_path=os.path.join(args.out_dir, "neural_kl.png"),
    )
    _plot_heatmap(
        tv_neural,
        neural_row_labels,
        col_labels,
        title="Neural ICL: TV (oracle, pred)",
        out_path=os.path.join(args.out_dir, "neural_tv.png"),
    )

    matrix_methods: List[str] = []
    for method, entry in baseline_rows.items():
        kl_m = _extract_metric(entry, "KL")
        tv_m = _extract_metric(entry, "TV")
        if kl_m is None or tv_m is None:
            continue
        if _is_matrix_like(kl_m) or _is_matrix_like(tv_m):
            matrix_methods.append(str(method))

    _preferred = ["mcmc_mixture", "svi_mixture"]
    matrix_methods = sorted(matrix_methods, key=lambda m: (_preferred.index(m) if m in _preferred else 999, m))

    for method in matrix_methods:
        entry = _get_baseline_entry(baseline_rows, method)
        if entry is None:
            continue
        kl_m = _extract_metric(entry, "KL")
        tv_m = _extract_metric(entry, "TV")
        if kl_m is None or tv_m is None:
            continue
        if not _is_matrix_like(kl_m):
            continue

        kl_mat = _to_np(kl_m).astype(np.float64)
        tv_mat = _to_np(tv_m).astype(np.float64)

        mix_row_labels = entry.get("row_labels", None)
        if mix_row_labels is None:
            mix_row_labels = neural_row_labels
        mix_row_labels = list(mix_row_labels)

        _plot_heatmap(
            kl_mat,
            mix_row_labels,
            col_labels,
            title=f"{_method_label(method)}: KL heatmap",
            out_path=os.path.join(args.out_dir, f"{method}_kl.png"),
        )
        _plot_heatmap(
            tv_mat,
            mix_row_labels,
            col_labels,
            title=f"{_method_label(method)}: TV heatmap",
            out_path=os.path.join(args.out_dir, f"{method}_tv.png"),
        )

    oracle_methods = ["mcmc_oracle", "svi_oracle"]
    oracle_kl_rows = []
    oracle_tv_rows = []
    oracle_labels = []

    for m in oracle_methods:
        entry = _get_baseline_entry(baseline_rows, m)
        if entry is None:
            continue
        kl = _extract_metric(entry, "KL")
        tv = _extract_metric(entry, "TV")
        if kl is None or tv is None:
            continue

        kl_1x = _maybe_2d(kl)
        tv_1x = _maybe_2d(tv)

        if kl_1x.shape[0] > 1:
            kl_1x = kl_1x[:1, :]
        if tv_1x.shape[0] > 1:
            tv_1x = tv_1x[:1, :]

        oracle_kl_rows.append(kl_1x.astype(np.float64))
        oracle_tv_rows.append(tv_1x.astype(np.float64))
        oracle_labels.append(_method_label(m))

    if oracle_kl_rows:
        kl_stack = np.vstack([kl_neural] + oracle_kl_rows)
        tv_stack = np.vstack([tv_neural] + oracle_tv_rows)
        row_labels_stack = neural_row_labels + oracle_labels

        _plot_heatmap(
            kl_stack,
            row_labels_stack,
            col_labels,
            title="Neural ICL + Oracle baselines: KL to MCMC-oracle",
            out_path=os.path.join(args.out_dir, "stack_neural_plus_oracles_kl.png"),
        )
        _plot_heatmap(
            tv_stack,
            row_labels_stack,
            col_labels,
            title="Neural ICL + Oracle baselines: TV to MCMC-oracle",
            out_path=os.path.join(args.out_dir, "stack_neural_plus_oracles_tv.png"),
        )

    def _diff_plots(method: str) -> None:
        entry = _get_baseline_entry(baseline_rows, method)
        if entry is None:
            return
        kl_m = _extract_metric(entry, "KL")
        tv_m = _extract_metric(entry, "TV")
        if kl_m is None or tv_m is None:
            return
        if not _is_matrix_like(kl_m):
            return

        kl_mat = _to_np(kl_m).astype(np.float64)
        tv_mat = _to_np(tv_m).astype(np.float64)

        if kl_mat.shape != kl_neural.shape or tv_mat.shape != tv_neural.shape:
            print(
                f"[WARN] shape mismatch for diff: {method} "
                f"kl={kl_mat.shape} vs neural={kl_neural.shape}. Skipping diffs."
            )
            return

        dkl = kl_mat - kl_neural
        dtv = tv_mat - tv_neural

        mix_row_labels = entry.get("row_labels", None)
        if mix_row_labels is None:
            mix_row_labels = neural_row_labels
        mix_row_labels = list(mix_row_labels)

        _plot_heatmap(
            dkl,
            mix_row_labels,
            col_labels,
            title=f"ΔKL: {_method_label(method)} - neural",
            out_path=os.path.join(args.out_dir, f"diff_{method}_minus_neural_kl.png"),
            cmap="coolwarm",
            center_zero=True,
        )
        _plot_heatmap(
            dtv,
            mix_row_labels,
            col_labels,
            title=f"ΔTV: {_method_label(method)} - neural",
            out_path=os.path.join(args.out_dir, f"diff_{method}_minus_neural_tv.png"),
            cmap="coolwarm",
            center_zero=True,
        )

    for method in matrix_methods:
        _diff_plots(method)

    print(f"✓ Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()

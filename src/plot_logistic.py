import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import argparse

from src.utils.eval_helpers import bernoulli_kl, bernoulli_tv

def _detach(x) -> torch.Tensor:
    return x.detach().cpu().float()

def compute_curves_from_final_results(
    final_results: List[Dict],
    *,
    oracle_method: str,
    methods: List[str],
    metric: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    curves[method] = {"K": [...], "mean": [...], "stderr": [...], "n": [...]}
    """
    assert metric in ("kl", "tv")
    metric_fn = bernoulli_kl if metric == "kl" else bernoulli_tv

    # buckets[method][K] = list of divergence values over draws
    buckets: Dict[str, Dict[int, List[float]]] = {m: {} for m in methods}

    for draw_i, d in enumerate(final_results):
        if oracle_method not in d:
            continue
        oracle = d[oracle_method]
        if "q_final" not in oracle:
            raise RuntimeError(f"Draw {draw_i}: oracle {oracle_method} missing q_final")
        q_ref = _detach(oracle["q_final"])

        for m in methods:
            if m not in d:
                continue
            md = d[m]
            if "q_probs" not in md or "K" not in md:
                continue

            K_list = list(md["K"])
            q_list = md["q_probs"]
            if len(K_list) != len(q_list):
                raise RuntimeError(f"Draw {draw_i}: {m} has len(K)!={len(q_list)}")

            for K, q_hat in zip(K_list, q_list):
                q_hat = _detach(q_hat)
                if q_hat.numel() != q_ref.numel():
                    raise RuntimeError(
                        f"Draw {draw_i}: {m}@K={K} q_hat shape {tuple(q_hat.shape)} "
                        f"!= oracle shape {tuple(q_ref.shape)}"
                    )
                val = metric_fn(q_hat, q_ref)
                buckets[m].setdefault(int(K), []).append(val)

    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for m in methods:
        if not buckets[m]:
            continue
        Ks = np.array(sorted(buckets[m].keys()), dtype=np.int64)

        means, stderrs, ns = [], [], []
        for K in Ks:
            vals = np.array(buckets[m][int(K)], dtype=np.float64)
            n = vals.size
            mu = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            means.append(mu)
            stderrs.append(se)
            ns.append(n)

        curves[m] = {
            "K": Ks,
            "mean": np.array(means, dtype=np.float64),
            "stderr": np.array(stderrs, dtype=np.float64),
            "n": np.array(ns, dtype=np.int64),
        }

    return curves

def compute_neural_divergence_vs_oracle(
    final_results: List[Dict],
    *,
    oracle_method: str,
    neural_key: str = "neural",
    metric: str = "kl",
) -> Dict[str, float]:
    """
    Returns {"mean": ..., "stderr": ..., "n": ...} over draws.
    """
    assert metric in ("kl", "tv")
    metric_fn = bernoulli_kl if metric == "kl" else bernoulli_tv

    vals = []

    for draw_i, d in enumerate(final_results):
        if oracle_method not in d:
            continue
        if neural_key not in d:
            continue
        if "q_final" not in d[oracle_method]:
            raise RuntimeError(f"Draw {draw_i}: oracle {oracle_method} missing q_final")
        if "q_probs" not in d[neural_key]:
            raise RuntimeError(f"Draw {draw_i}: {neural_key} missing q_probs")

        q_ref = _detach(d[oracle_method]["q_final"])
        q_neural = _detach(d[neural_key]["q_probs"])

        if q_neural.numel() != q_ref.numel():
            raise RuntimeError(
                f"Draw {draw_i}: neural q_probs shape {tuple(q_neural.shape)} "
                f"!= oracle shape {tuple(q_ref.shape)}"
            )

        val = metric_fn(q_neural, q_ref)
        vals.append(val)

    if len(vals) == 0:
        return {"mean": float("nan"), "stderr": float("nan"), "n": 0}

    vals = np.array(vals, dtype=np.float64)
    n = vals.size
    mean = float(vals.mean())
    stderr = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "stderr": stderr, "n": int(n)}

def compute_neural_divergences_vs_oracle(
    final_results: List[Dict],
    *,
    oracle_method: str,
    neural_keys: List[str],
    metric: str = "kl",
) -> Dict[str, Dict[str, float]]:
    """
    Returns {neural_key: {"mean": ..., "stderr": ..., "n": ...}} for each key.
    """
    out: Dict[str, Dict[str, float]] = {}
    for k in neural_keys:
        out[k] = compute_neural_divergence_vs_oracle(
            final_results,
            oracle_method=oracle_method,
            neural_key=k,
            metric=metric,
        )
    return out


PLOT_CONFIGS = {
    "mcmc": {"color": "tab:cyan"},
    "mcmc_hier": {"color": "tab:blue"},
    "mcmc_spiral_oracle": {"color": "tab:red"},
    "svi": {"color": "tab:orange"},
    "svi_hier": {"color": "tab:purple"},
    "mcmc_hier_spiral": {"color": "tab:blue"},
    "svi_hier_spiral": {"color": "tab:purple"},
}

def plot_curves(
    curves: Dict[str, Dict[str, np.ndarray]],
    *,
    metric: str,
    title: str,
    out_path: Optional[str],
    show: bool,
    neural_line: Optional[Dict[str, float]] = None,  # backward compat: one line {"mean","stderr","n"}
    neural_lines: Optional[Dict[str, Dict[str, float]]] = None,  # {label: {"mean","stderr","n"}}
    neural_band: bool = False,
    neural_label: str = "neural",
) -> None:
    plt.figure(figsize=(7.5, 4.5))

    # Plot baseline curves
    for method, d in curves.items():
        K = d["K"]
        y = d["mean"]
        se = d["stderr"]
        method_color = PLOT_CONFIGS[method]["color"]
        method_label = method.upper().replace("_", "-")
        if method == "mcmc_hier_spiral":
            method_label = "MCMC-HIER"
        elif method == "svi_hier_spiral":
            method_label = "SVI-HIER"
        plt.plot(K, y, marker="o", label=method_label, color=method_color)
        plt.fill_between(K, y - se, y + se, alpha=0.2, color=method_color)

    # Plot neural horizontal line(s)
    # Backward compatible behavior: if only neural_line is provided, convert it into a dict.
    if neural_lines is None and neural_line is not None:
        neural_lines = {neural_label: neural_line}

    if neural_lines is not None:
        # Shade across the full x-range present in curves (if requested)
        all_K = np.concatenate([d["K"] for d in curves.values()]) if len(curves) > 0 else np.array([0, 1])
        xmin, xmax = float(all_K.min()), float(all_K.max())

        # distinct colors for up to a few neural models
        color_cycle = ["green", "tab:pink", "tab:brown", "tab:gray", "tab:olive"]
        for i, (lbl, stats) in enumerate(neural_lines.items()):
            if stats is None or stats.get("n", 0) <= 0 or not np.isfinite(stats.get("mean", np.nan)):
                continue
            y = float(stats["mean"])
            c = color_cycle[i % len(color_cycle)]
            plt.axhline(y, linestyle="--", linewidth=1, label=f"{lbl} (mean={y:.4g})", color=c)
            if neural_band and np.isfinite(stats.get("stderr", np.nan)) and stats.get("n", 0) > 1:
                se = float(stats["stderr"])
                plt.fill_between([xmin, xmax], [y - se, y - se], [y + se, y + se], alpha=0.12, color=c)
    
    plt.xlabel("# samples (MCMC) / steps (SVI)", fontsize=14)

    plt.ylabel(metric.upper(), fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved plot to {out_path}")

    if show:
        plt.show()

    plt.close()


def _fmt_float(v: float, decimals: int) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{decimals}f}"


def _render_table(headers: List[str], rows: List[List[str]], table_format: str) -> str:
    if table_format == "csv":
        lines = [",".join(headers)]
        lines.extend(",".join(row) for row in rows)
        return "\n".join(lines)

    if table_format == "markdown":
        sep = ["---"] * len(headers)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    # plain text table
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _pad(row: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    divider = "  ".join("-" * w for w in widths)
    lines = [_pad(headers), divider]
    lines.extend(_pad(row) for row in rows)
    return "\n".join(lines)


NEURAL_KEY_DISPLAY = {
    "with prefix": "ICL (with prefix)",
    "no prefix": "ICL (no prefix)",
}


def print_results_tables(
    curves: Dict[str, Dict[str, np.ndarray]],
    *,
    neural_stats_by_key: Dict[str, Dict[str, float]],
    metric: str,
    oracle: str,
    table_format: str,
    decimals: int,
    only_k: Optional[int] = None,
) -> None:
    metric_upper = metric.upper()

    rows: List[List[str]] = []

    # Baseline methods (filtered to only_k if set)
    for method, d in curves.items():
        Ks = d["K"]
        means = d["mean"]
        stderrs = d["stderr"]
        for i in range(len(Ks)):
            K_val = int(Ks[i])
            if only_k is not None and K_val != only_k:
                continue
            rows.append(
                [
                    method,
                    str(K_val),
                    _fmt_float(float(means[i]), decimals),
                    _fmt_float(float(stderrs[i]), decimals),
                ]
            )

    # Neural rows appended at the bottom
    for key, st in neural_stats_by_key.items():
        display_name = NEURAL_KEY_DISPLAY.get(key, key)
        rows.append(
            [
                display_name,
                "all",
                _fmt_float(float(st.get("mean", np.nan)), decimals),
                _fmt_float(float(st.get("stderr", np.nan)), decimals),
            ]
        )

    k_label = f" @ K={only_k}" if only_k is not None else ""
    print(f"\n=== Results table ({metric_upper} vs oracle={oracle}{k_label}) ===")
    if rows:
        print(_render_table(["method", "K", "mean", "stderr"], rows, table_format))
    else:
        print("(no rows found)")


def run_plot_curves(args):
    final_results = torch.load(args.results_path, map_location="cpu")

    curves = compute_curves_from_final_results(
        final_results,
        oracle_method=args.oracle,
        methods=args.methods,
        metric=args.metric,
    )

    neural_stats_by_key = compute_neural_divergences_vs_oracle(
        final_results,
        oracle_method=args.oracle,
        neural_keys=list(args.neural_keys),
        metric=args.metric,
    )

    if args.plot_out is None:
        base_dir = os.path.dirname(args.results_path)
        args.plot_out = os.path.join(base_dir, f"{args.metric}_vs_K_logistic.png")

    if not args.skip_plot:
        title = f"{args.metric.upper()} vs K (oracle approximated by {args.oracle} with 10000 samples)"
        plot_curves(
            curves,
            metric=args.metric,
            title=title,
            out_path=args.plot_out,
            show=args.show,
            neural_lines=neural_stats_by_key,
            neural_band=args.neural_band,
        )

    for k in args.neural_keys:
        st = neural_stats_by_key.get(k, {"mean": float("nan"), "stderr": float("nan"), "n": 0})
        print(f"{k} {args.metric.upper()} vs oracle={args.oracle}: mean={st['mean']:.6g}, "
              f"stderr={st['stderr']:.6g}, n={st['n']}")

    if not args.no_table:
        print_results_tables(
            curves,
            neural_stats_by_key=neural_stats_by_key,
            metric=args.metric,
            oracle=args.oracle,
            table_format=args.table_format,
            decimals=args.table_decimals,
            only_k=args.table_only_k,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, required=True,
                    help="Path to final_results_ctx*.pt produced by run_merge()")
    parser.add_argument("--oracle", type=str, default="mcmc_hier", choices=["mcmc", "mcmc_hier", "mcmc_spiral_oracle"])
    parser.add_argument("--metric", type=str, default="kl", choices=["kl", "tv"])
    parser.add_argument("--methods", nargs="+", default=["mcmc", "mcmc_hier", "svi", "svi_hier"],
                        help="Which methods to plot curves for (must exist in final_results items).")
    parser.add_argument("--plot-out", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--neural-keys", nargs="+", default=["neural"],
                    help="One or more keys used in final_results for neural outputs (default: neural). "
                         "Example: --neural-keys neural neural2")
    parser.add_argument("--neural-band", action="store_true",
                        help="If set, show +/- stderr band for neural horizontal line")
    parser.add_argument("--skip-plot", action="store_true",
                        help="If set, do not generate/save the figure; compute/print stats only.")
    parser.add_argument("--no-table", action="store_true",
                        help="If set, suppress table printing.")
    parser.add_argument("--table-format", type=str, default="markdown", choices=["plain", "markdown", "csv"],
                        help="Output format for printed result tables.")
    parser.add_argument("--table-decimals", type=int, default=6,
                        help="Decimal places for means/stderr in printed tables.")
    parser.add_argument("--table-only-k", type=int, default=None,
                        help="If set, print baseline table rows only for this K value.")

    args = parser.parse_args()
    run_plot_curves(args)
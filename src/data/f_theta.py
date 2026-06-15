from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union, List

import torch

LikelihoodType = Literal["identity", "spline", "mlp"]


@dataclass(frozen=True)
class SplineConfig:
    """
    1D natural cubic spline f(s) defined by knots (x_i, y_i).
    x_i are fixed on [x_min, x_max], y_i are sampled.
    """
    num_knots: int = 16
    x_min: float = -8.0
    x_max: float = 8.0
    y_scale: float = 1.0
    monotone: bool = False


@dataclass(frozen=True)
class MLPConfig:
    """
    1D -> 1 MLP applied to s = w^T x (single-index).
    """
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: Literal["tanh", "relu", "gelu"] = "tanh"
    out_scale: float = 1.0


def _coerce_spline_cfg(cfg: Optional[object]) -> Optional[SplineConfig]:
    """
    Allow YAML-friendly dict configs.
    - None -> None
    - SplineConfig -> itself
    - dict -> SplineConfig(**dict)
    """
    if cfg is None or isinstance(cfg, SplineConfig):
        return cfg
    if isinstance(cfg, dict):
        return SplineConfig(**cfg)
    raise TypeError(f"spline_cfg must be SplineConfig|dict|None, got {type(cfg)}")


def _coerce_mlp_cfg(cfg: Optional[object]) -> Optional[MLPConfig]:
    """
    Allow YAML-friendly dict configs.
    - None -> None
    - MLPConfig -> itself
    - dict -> MLPConfig(**dict), with hidden_sizes coerced to tuple if provided as list
    """
    if cfg is None or isinstance(cfg, MLPConfig):
        return cfg
    if isinstance(cfg, dict):
        cfg = dict(cfg)
        if "hidden_sizes" in cfg and isinstance(cfg["hidden_sizes"], list):
            cfg["hidden_sizes"] = tuple(int(x) for x in cfg["hidden_sizes"])
        return MLPConfig(**cfg)
    raise TypeError(f"mlp_cfg must be MLPConfig|dict|None, got {type(cfg)}")


def _compute_natural_cubic_M(xk: torch.Tensor, yk: torch.Tensor) -> torch.Tensor:
    """
    Compute second derivatives M for a natural cubic spline through knots (xk, yk).
    xk: [K], strictly increasing
    yk: [K]
    returns M: [K] with M[0]=M[-1]=0
    """
    K = xk.numel()
    if K < 2:
        raise ValueError("Need at least 2 knots for spline.")
    eps = 1e-12

    if K == 2:
        return torch.zeros_like(yk)

    h = torch.clamp(xk[1:] - xk[:-1], min=eps)  # [K-1]
    n = K - 2
    h_im1 = h[:-1]  # [n]
    h_i = h[1:]  # [n]

    diag = 2.0 * (h_im1 + h_i)  # [n]
    lower = h_im1[1:]  # [n-1]
    upper = h_i[:-1]  # [n-1]

    rhs = 6.0 * ((yk[2:] - yk[1:-1]) / h_i - (yk[1:-1] - yk[:-2]) / h_im1)  # [n]

    # Thomas algorithm
    c_prime = torch.empty(n - 1, device=xk.device, dtype=yk.dtype)
    d_prime = torch.empty(n, device=xk.device, dtype=yk.dtype)

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    denom_last = diag[n - 1] - (lower[n - 2] * c_prime[n - 2] if n > 1 else 0.0)
    d_prime[n - 1] = (rhs[n - 1] - (lower[n - 2] * d_prime[n - 2] if n > 1 else 0.0)) / denom_last

    M_interior = torch.empty(n, device=xk.device, dtype=yk.dtype)
    M_interior[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        M_interior[i] = d_prime[i] - c_prime[i] * M_interior[i + 1]

    M = torch.zeros_like(yk)
    M[1:-1] = M_interior
    return M


def _sample_spline_theta(cfg: SplineConfig, generator: torch.Generator) -> Dict[str, torch.Tensor]:
    K = int(cfg.num_knots)
    if K < 2:
        raise ValueError("SplineConfig.num_knots must be >= 2")

    x_knots = torch.linspace(float(cfg.x_min), float(cfg.x_max), steps=K)

    y_knots = torch.randn(K, generator=generator) * float(cfg.y_scale)
    if bool(cfg.monotone):
        inc = torch.nn.functional.softplus(torch.randn(K, generator=generator))
        y_knots = torch.cumsum(inc, dim=0)
        y_knots = (y_knots - y_knots.mean()) * float(cfg.y_scale)

    M = _compute_natural_cubic_M(x_knots, y_knots)
    return {"x_knots": x_knots, "y_knots": y_knots, "M": M}


def _sample_mlp_theta(cfg: MLPConfig, generator: torch.Generator) -> Dict[str, Any]:
    """
    Sample MLP weights explicitly (so it's deterministic and doesn't require nn.Module
    state inside Dataset). Forward is done with matmuls.
    """
    sizes: List[int] = [1, *list(cfg.hidden_sizes), 1]
    params: Dict[str, Any] = {}
    for i in range(len(sizes) - 1):
        in_dim, out_dim = sizes[i], sizes[i + 1]
        W = torch.randn(out_dim, in_dim, generator=generator) * (1.0 / (in_dim**0.5))
        b = torch.randn(out_dim, generator=generator) * 0.01
        params[f"W{i}"] = W
        params[f"b{i}"] = b
    params["out_scale"] = torch.tensor(float(cfg.out_scale))
    params["activation"] = cfg.activation
    return params


def init_f_theta(
    *,
    likelihood: Union[str, LikelihoodType],
    generator: torch.Generator,
    spline_cfg: Optional[Union[SplineConfig, dict]] = None,
    mlp_cfg: Optional[Union[MLPConfig, dict]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Sample theta for the given likelihood type.
    Returns:
      - None for identity
      - dict for spline/mlp
    """
    lk = str(likelihood).lower()
    if lk == "identity":
        return None

    scfg = _coerce_spline_cfg(spline_cfg) or SplineConfig()
    mcfg = _coerce_mlp_cfg(mlp_cfg) or MLPConfig()

    if lk == "spline":
        return _sample_spline_theta(scfg, generator)
    if lk == "mlp":
        return _sample_mlp_theta(mcfg, generator)
    raise ValueError(f"Unknown likelihood '{likelihood}'. Expected one of: identity|spline|mlp")


def apply_spline(s: torch.Tensor, theta: Dict[str, Any]) -> torch.Tensor:
    """
    Natural cubic spline eval using precomputed second derivatives M in theta.
    s: arbitrary shape, returns same shape
    """
    xk = theta["x_knots"].to(s.device)
    yk = theta["y_knots"].to(s.device)
    M = theta["M"].to(s.device)

    K = xk.numel()
    eps = 1e-12

    s_flat = s.view(-1)
    s_clamped = torch.clamp(s_flat, xk[0].item(), xk[-1].item())

    if K == 2:
        t = (s_clamped - xk[0]) / (xk[1] - xk[0] + eps)
        y = yk[0] + t * (yk[1] - yk[0])
        return y.view(s.shape)

    idx = torch.bucketize(s_clamped, xk)
    idx = torch.clamp(idx, 1, K - 1)
    j = idx - 1

    xj = xk[j]
    xj1 = xk[j + 1]
    hj = torch.clamp(xj1 - xj, min=eps)

    yj = yk[j]
    yj1 = yk[j + 1]
    Mj = M[j]
    Mj1 = M[j + 1]

    a = (xj1 - s_clamped) / hj
    b = (s_clamped - xj) / hj

    y = a * yj + b * yj1 + ((a**3 - a) * Mj + (b**3 - b) * Mj1) * (hj**2) / 6.0
    return y.view(s.shape)


def apply_mlp(s: torch.Tensor, theta: Dict[str, Any]) -> torch.Tensor:
    """
    Manual MLP forward. Accepts arbitrary shape; internally flattens to [N,1].
    Returns same shape as input.
    """
    s2 = s.view(-1, 1)
    act = theta["activation"]
    h = s2
    layer = 0
    while f"W{layer}" in theta:
        W = theta[f"W{layer}"].to(h.device)
        b = theta[f"b{layer}"].to(h.device)
        h = h @ W.t() + b
        is_last = (f"W{layer+1}" not in theta)
        if not is_last:
            if act == "tanh":
                h = torch.tanh(h)
            elif act == "relu":
                h = torch.relu(h)
            elif act == "gelu":
                h = torch.nn.functional.gelu(h)
            else:
                raise ValueError(f"Unknown activation: {act}")
        layer += 1
    h = h * theta["out_scale"].to(h.device)
    return h.view(s.shape)


def apply_f_theta(
    s: torch.Tensor,
    *,
    likelihood: Union[str, LikelihoodType],
    theta: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Apply f_theta to the single-index scores s and return same-shaped tensor.
    """
    lk = str(likelihood).lower()
    if lk == "identity":
        return s
    if theta is None:
        raise ValueError("theta must be provided for non-identity likelihoods.")
    if lk == "spline":
        return apply_spline(s, theta)
    if lk == "mlp":
        return apply_mlp(s, theta)
    raise ValueError(f"Unknown likelihood '{likelihood}'. Expected one of: identity|spline|mlp")


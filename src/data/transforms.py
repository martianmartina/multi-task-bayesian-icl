import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Optional


class RadialFlow(nn.Module):
    """
    Expands or contracts space radially around a center point 'z_0'.
    Can turn a blob into a ring.
    Equation: f(z) = z + beta * h(alpha, r) * (z - z_0)
    """
    def __init__(self, dim, generator: torch.Generator = None):
        super().__init__()
        gen_kwargs = {} if generator is None else {"generator": generator}
        self.z0 = nn.Parameter(torch.randn(dim, **gen_kwargs))      # Center of the radial distortion
        self.log_alpha = nn.Parameter(torch.randn(1, **gen_kwargs)) # Smoothness of distortion
        self.beta = nn.Parameter(torch.randn(1, **gen_kwargs))      # Magnitude of distortion
        
    def forward(self, z):
        
        # r is distance from center z0
        r = torch.norm(z - self.z0, dim=1, keepdim=True)

        # Ensure parameters allow invertibility (h' > -1)
        # We constrain alpha > 0 and beta >= -alpha
        alpha = F.softplus(self.log_alpha)
        beta = -alpha + F.softplus(self.beta) # constraint to ensure invertibility

        # h(r) = 1 / (alpha + r)
        h = 1.0 / (alpha + r)

        # Transformation
        return z + beta * h * (z - self.z0)

class PlanarFlow(nn.Module):
    """
    Slices space along a plane defined by 'w' and 'b' and warps it.
    Can create folds and creases.
    Equation: f(z) = z + u * tanh(w^T z + b)
    """
    def __init__(self, dim, generator: torch.Generator = None):
        super().__init__()
        gen_kwargs = {} if generator is None else {"generator": generator}
        self.u = nn.Parameter(torch.randn(1, dim, **gen_kwargs) * 2.0) # Scale of warp (init high for chaos)
        self.w = nn.Parameter(torch.randn(1, dim, **gen_kwargs) * 2.0) # Direction of cut
        self.b = nn.Parameter(torch.randn(1, **gen_kwargs))            # Position of cut

    def forward(self, z):
        # We need to enforce w^T * u > -1 for invertibility.
        # This is a standard trick to reparameterize u:
        dot = torch.mm(self.u, self.w.t())
        u_hat = self.u
        if dot < -1:
            # If constraint violated, adjust u parallel to w
            w_norm_sq = torch.sum(self.w ** 2)
            u_hat = self.u + ((-1 + F.softplus(dot)) / w_norm_sq - dot) * self.w

        # Linear projection: w^T z + b
        lin = F.linear(z, self.w, self.b)

        # Non-linear activation: tanh
        return z + u_hat * torch.tanh(lin)

class ChaoticFlow(nn.Module):
    def __init__(self, dim=8, depth=20, generator: torch.Generator = None):
        super().__init__()
        self.layers = nn.ModuleList()

        print(f"Building Chaotic Flow with {depth} mixed layers...")
        for i in range(depth):
            # Alternate between Planar (Folding) and Radial (Expanding)
            if i % 2 == 0:
                self.layers.append(PlanarFlow(dim, generator=generator))
            else:
                self.layers.append(RadialFlow(dim, generator=generator))

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z



class SpiralFlow(nn.Module):
    r"""
    Dimension-preserving spiral:
        z ↦ R(z) z,   where R(z) = initial @ exp(generator * ||z||^2).

    generator: skew-symmetric matrix in so(D)
    initial:   any invertible matrix (we'll just use identity by default)
    """
    def __init__(
        self,
        dim: int,
        speed: float = 1.0,
        skew_matrix: torch.Tensor = None,
        initial: torch.Tensor = None,
        generator: torch.Generator = None,
        verbose: bool = False,
        random_initial: bool = False,
        initial_scale: float = 0.1,   # how far from identity
    ):
        super().__init__()
        rng = generator
        # TODO: make sure test time diff param from train param used for ICL
        gen_kwargs = {} if rng is None else {"generator": rng}
        self.verbose = bool(verbose)

        # 1) Skew-symmetric generator
        if skew_matrix is None:
            A = torch.randn(dim, dim, **gen_kwargs) # sample from nomal with nonzero mean
            A = A - A.T  # make it skew-symmetric
            if self.verbose:
                print(f"[SpiralFlow] sampled skew generator A (shape={tuple(A.shape)})")
        else:
            A = torch.as_tensor(skew_matrix, dtype=torch.get_default_dtype())
            if A.shape != (dim, dim):
                raise ValueError(f"skew_matrix has shape {A.shape}, expected ({dim}, {dim})")

        A = A * speed
        self.register_buffer("generator_matrix", A)

        # 2) Initial matrix
        if initial is None:
            if random_initial:
                # Invertible by construction: lower-triangular with positive-ish diagonal near 1
                L = torch.tril(torch.randn(dim, dim, **gen_kwargs) * initial_scale)
                L.diagonal().add_(1.0)   # diag ~ 1 => well-conditioned, invertible
                I = L.to(torch.get_default_dtype())
            else:
                I = torch.eye(dim, dtype=torch.get_default_dtype())
        else:
            I = torch.as_tensor(initial, dtype=torch.get_default_dtype())
            if I.shape != (dim, dim):
                raise ValueError(f"initial has shape {I.shape}, expected ({dim}, {dim})")
        self.register_buffer("initial", I)

        self.dim = dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,D] or [D]
        squeezed = False
        if z.dim() == 1:
            z = z.unsqueeze(0)  # [1,D]
            squeezed = True

        z = z.to(self.generator_matrix.dtype)
        # r^2 per sample
        r2 = (z ** 2).sum(dim=1)  # [B]
        G = self.generator_matrix * r2.view(-1, 1, 1)  # [B,D,D]

        # batched matrix exponential
        R = torch.matrix_exp(G)  # [B,D,D]

        R_total = torch.matmul(self.initial, R)  # [B,D,D], broadcasts initial

        z_out = torch.matmul(R_total, z.unsqueeze(-1)).squeeze(-1)  # [B,D]

        if squeezed:
            z_out = z_out.squeeze(0)
        return z_out

def _spiral_params_from_seed(
    x_dim: int,
    transform_seed: int,
    *,
    speed: float = 1.0,
    random_initial: bool = False,
    initial_scale: float = 0.1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Reconstruct the exact SpiralFlow parameters used in `src/data/transforms.py::SpiralFlow`
    when called as `SpiralFlow(dim=x_dim, speed=speed, generator=g, random_initial=..., initial_scale=...)`.

    This is needed for "oracle-known-theta" baselines where theta is identified by `transform_seed`.
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.get_default_dtype()

    # SpiralFlow samples parameters on CPU via the provided generator.
    g = torch.Generator(device="cpu").manual_seed(int(transform_seed))

    # 1) Skew-symmetric generator matrix A (then multiplied by speed)
    A = torch.randn(int(x_dim), int(x_dim), generator=g, device="cpu", dtype=dtype)
    A = (A - A.t()) * float(speed)  # skew-symmetric

    # 2) Initial matrix (identity by default unless random_initial=True)
    if random_initial:
        L = torch.tril(torch.randn(int(x_dim), int(x_dim), generator=g, device="cpu", dtype=dtype) * float(initial_scale))
        L.diagonal().add_(1.0)
        initial = L
    else:
        initial = torch.eye(int(x_dim), device="cpu", dtype=dtype)

    return A.to(device=device, dtype=dtype), initial.to(device=device, dtype=dtype)


def spiral_pushforward(
    z: torch.Tensor,            # [D] or [B,D]
    A_skew: torch.Tensor,       # [D,D] skew
    initial: torch.Tensor = None,  # [D,D] or None
) -> torch.Tensor:
    """
    SpiralFlow-like push-forward:
        w = (initial @ exp(A * ||z||^2)) @ z
    """
    squeezed = False
    if z.ndim == 1:
        z = z.unsqueeze(0)
        squeezed = True

    D = z.shape[-1]
    if initial is None:
        initial = torch.eye(D, device=z.device, dtype=z.dtype)

    r2 = (z ** 2).sum(dim=-1)  # [B]
    G = A_skew.unsqueeze(0) * r2.view(-1, 1, 1)  # [B,D,D]
    R = torch.matrix_exp(G)  # [B,D,D]
    R_total = torch.matmul(initial.unsqueeze(0), R)  # [B,D,D]
    w = torch.matmul(R_total, z.unsqueeze(-1)).squeeze(-1)  # [B,D]

    if squeezed:
        return w.squeeze(0)
    return w

class ComposedFlow(nn.Module):
    """
    A stronger, more chaotic flow:
    composition of Planar, Radial, and Spiral layers.
    """
    def __init__(self, dim: int, depth: int = 20, generator: torch.Generator = None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList()
        rng = generator

        for i in range(depth):
            mod = i % 3
            if mod == 0:
                self.layers.append(PlanarFlow(dim, generator=rng))
            elif mod == 1:
                self.layers.append(RadialFlow(dim, generator=rng))
            else:
                # Spiral with moderate twisting
                self.layers.append(SpiralFlow(dim, speed=1.5, generator=rng))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: [B, dim] or [dim]
        """
        squeezed = False

        # Handle [dim]
        if w.dim() == 1:
            w = w.unsqueeze(0)
            squeezed = True

        # If someone passes [B, dim, 1], drop the last singleton
        if w.dim() == 3 and w.size(-1) == 1:
            w = w.squeeze(-1)

        # Now w is [B, dim]
        for layer in self.layers:
            w = layer(w)

        if squeezed:
            w = w.squeeze(0)

        return w

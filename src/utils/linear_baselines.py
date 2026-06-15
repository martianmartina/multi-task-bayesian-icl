# src/utils/linear_baselines.py
from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import torch
from src.data.multi_task_synthetic import PRIOR_TOKEN_X, TARGET_TOKEN_X
from src.utils.my_pyro_models import _simple_regression_model_eval, _hier_regression_model_eval
# -------------------------
# Gaussian KL helper
# -------------------------
def gaussian_kl(mean1, logvar1, mean2, logvar2):
    """
    Elementwise KL:
      KL( N(mean1, var1) || N(mean2, var2) )
    Supports broadcasting.
    """
    var1 = np.exp(logvar1)
    var2 = np.exp(logvar2)
    return 0.5 * (logvar2 - logvar1 + (var1 + (mean1 - mean2) ** 2) / var2 - 1.0)

# -------------------------
# Ridge Oracle
# -------------------------
from models.ridge import Ridge

def ridge_oracle_predict(
    *,
    x_ctx: torch.Tensor,       # [N, D]
    y_ctx: torch.Tensor,       # [N, 1] or [N]
    x_query: torch.Tensor,     # [Q, D]
    prior_mean: float,
    prior_std: float,
    noise_std: float,
    noise_mean: float = 0.0,   # Added for the generalized case
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the predictive mean and standard deviation for the test points.
    
    Returns:
        mu:  [Q] array of predictive means
        std: [Q] array of predictive standard deviations
    """
    # Initialize the oracle with the correct Bayesian parameters
    ridge = Ridge(
        mu_w=float(prior_mean),
        std_w=float(prior_std),
        mu_eps=float(noise_mean),
        std_eps=float(noise_std),
    )

    # Move to CPU for safety/numpy compatibility and add Batch dimension [1, ...]
    # Inputs: (1, N, D), (1, N, 1), (1, Q, D)
    mu_tensor, var_tensor = ridge(
        x_ctx.unsqueeze(0).cpu(),
        y_ctx.unsqueeze(0).cpu(),
        x_query.unsqueeze(0).cpu(),
        return_var=True
    )

    # Handle outputs: (1, Q) -> (Q,)
    mu = mu_tensor.squeeze(0).detach().numpy() 
    var = var_tensor.squeeze(0).detach().numpy()

    logvar = np.log(var + 1e-12) 
    
    return mu, logvar

# -------------------------
# Neural Prediction Logic
# -------------------------
def predict_with_bootstrap(model, prior_tasks, x_target_context_pool, y_target_context_pool, x_target_query):
    model_device = next(model.parameters()).device
    x_dim = x_target_query.shape[-1]

    x_prior_seqs, y_prior_seqs = [], []
    for x_p, y_p in prior_tasks:
        x_prior_seqs.append(torch.full((1, x_dim), PRIOR_TOKEN_X, device=model_device))
        y_prior_seqs.append(torch.zeros(1, 1, device=model_device))
        x_prior_seqs.append(x_p.to(model_device))
        y_prior_seqs.append(y_p.to(model_device))

    x_context_prior = torch.cat(x_prior_seqs, dim=0) if x_prior_seqs else torch.empty(0, x_dim, device=model_device)
    y_context_prior = torch.cat(y_prior_seqs, dim=0) if y_prior_seqs else torch.empty(0, 1, device=model_device)

    with torch.no_grad():
        x_full_context = torch.cat([x_context_prior, torch.full((1, x_dim), TARGET_TOKEN_X, device=model_device), x_target_context_pool], dim=0)
        y_full_context = torch.cat([y_context_prior, torch.zeros(1, 1, device=model_device), y_target_context_pool], dim=0)

        batch_size = len(x_target_query)
        x_seq_batch = torch.cat([x_full_context.unsqueeze(0).expand(batch_size, -1, -1), x_target_query.unsqueeze(1)], dim=1)
        y_seq_batch = torch.cat([y_full_context.unsqueeze(0).expand(batch_size, -1, -1), torch.zeros(batch_size, 1, model.hparams.y_dim, device=model_device)], dim=1)
        
        model_input, _ = model._prepare_input_sequence(x_seq_batch, y_seq_batch)
        mu, logvar, _ = model(model_input)
        return mu[:, -1, :].cpu().numpy(), logvar[:, -1, :].cpu().numpy() # [Q, y_dim]
    
# -------------------------
# Pyro hierarchical baselines
# -------------------------
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from pyro.optim import Adam


def predict_hierarchical_mcmc(
    prior_tasks_data: List[Tuple[torch.Tensor, torch.Tensor]],
    x_target: torch.Tensor,
    y_target: torch.Tensor,
    x_query: torch.Tensor,
    x_dim: int,
    noise_std: float,
    *,
    num_samples: int,
    thinning: int,
    warmup_steps: int,
    hier_prior_mean_min: float,
    hier_prior_mean_max: float,
    disable_progbar: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns predictive (mu, logvar) for y(x_query).
    """
    pyro.clear_param_store()

    device = torch.device("cpu")
    dtype = torch.float32
    prior_tasks_X = [xp.to(device=device, dtype=dtype) for xp, _ in prior_tasks_data]
    prior_tasks_y = [yp.to(device=device, dtype=dtype) for _, yp in prior_tasks_data]
    x_target = x_target.to(device=device, dtype=dtype)
    y_target = y_target.to(device=device, dtype=dtype)
    x_query = x_query.to(device=device, dtype=dtype)

    nuts_kernel = NUTS(_hier_regression_model_eval)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=int(num_samples),
        warmup_steps=int(warmup_steps),
        disable_progbar=bool(disable_progbar),
    )

    mcmc.run(
        prior_tasks_X,
        prior_tasks_y,
        x_target,
        y_target,
        int(x_dim),
        float(noise_std),
        float(hier_prior_mean_min),
        float(hier_prior_mean_max),
    )

    samples = mcmc.get_samples()
    w_target = samples["w_target"]        # [S, D]
    
    # apply thinning to w_target
    w_target = w_target[::thinning]

    y_samp = (x_query @ w_target.T).T # [S, Q]
    mu = y_samp.mean(0).detach().cpu().numpy()
    var = y_samp.var(0).detach().cpu().numpy() + float(noise_std**2)
    return mu, np.log(var)


def predict_hierarchical_svi(
    prior_tasks_data: List[Tuple[torch.Tensor, torch.Tensor]],
    x_target: torch.Tensor,
    y_target: torch.Tensor,
    x_query: torch.Tensor,
    x_dim: int,
    noise_std: float,
    *,
    steps: int,
    posterior_samples: int,
    hier_prior_mean_min: float,
    hier_prior_mean_max: float,
    lr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same output as MCMC version, using SVI.
    """
    pyro.clear_param_store()


    device = torch.device("cpu")
    dtype = torch.float32
    prior_tasks_X = [xp.to(device=device, dtype=dtype) for xp, _ in prior_tasks_data]
    prior_tasks_y = [yp.to(device=device, dtype=dtype) for _, yp in prior_tasks_data]
    x_target = x_target.to(device=device, dtype=dtype)
    y_target = y_target.to(device=device, dtype=dtype)
    x_query = x_query.to(device=device, dtype=dtype)

    guide = pyro.infer.autoguide.AutoDiagonalNormal(_hier_regression_model_eval)
    svi = SVI(
        _hier_regression_model_eval,
        guide,
        Adam({"lr": float(lr)}),
        loss=Trace_ELBO(),
    )

    for _ in range(int(steps)):
        svi.step(
            prior_tasks_X,
            prior_tasks_y,
            x_target,
            y_target,
            int(x_dim),
            float(noise_std),
            float(hier_prior_mean_min),
            float(hier_prior_mean_max),
        )

    predictive = Predictive(
        _hier_regression_model_eval,
        guide=guide,
        num_samples=int(posterior_samples),
    )
    preds = predictive(
        prior_tasks_data,
        x_target,
        y_target,
        int(x_dim),
        float(noise_std),
        float(hier_prior_mean_min),
        float(hier_prior_mean_max),
    )

    w_samples = preds["w_target"]  
    if w_samples.ndim == 3:
        w_samples = w_samples.reshape(-1, x_dim) # [S, D]
        
    y_samp = (x_query @ w_samples.T).T # [S, Q]
    mu = y_samp.mean(0).detach().cpu().numpy() # [Q]
    var = y_samp.var(0).detach().cpu().numpy() + float(noise_std**2) # [Q]
    return mu, np.log(var)

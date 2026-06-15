"""
Helper functions for evaluating the performance of the models at inference time.
"""
import time
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
import pyro.optim as pyro_optim
from src.utils.my_pyro_models import (
    oracle_model_for_mcmc,
    oracle_model_for_hierarchical_mcmc,
    oracle_model_for_hierarchical_df_mcmc,  
    oracle_model_for_hierarchical_df_mixture_mu_mcmc,  
    oracle_model_for_hierarchical_mcmc_spiral,         
    oracle_model_for_mcmc_spiral_known_theta,          
    _simple_classification_model_eval,
    _hier_classification_model_eval,
    _hier_df_model_eval,                     
    _hier_df_mixture_mu_model_eval,           
    _hier_spiral_model_eval,                  
)
from src.data.transforms import spiral_pushforward, _spiral_params_from_seed
from typing import List, Optional, Union, Iterable
from src.data.multi_task_logistic_regression import PRIOR_TOKEN_X, TARGET_TOKEN_X, Y_PADDING
import numpy as np

ORACLE_NUM_SAMPLES = 10000
ORACLE_WARMUP_STEPS = 1000
ORACLE_THINNING = 10
NUM_CHAINS = 1
X_DIM = 8

"""
------------------------------------ Helpers ------------------------------------
"""

def sample_normal_vec(mean_scalar: float, std_scalar: float, dim: int) -> torch.Tensor:
    return torch.randn(dim, 1) * std_scalar + mean_scalar

def sample_laplace_vec(mean_scalar: float, scale_scalar: float, dim: int) -> torch.Tensor:
    dist = torch.distributions.Laplace(loc=mean_scalar, scale=scale_scalar)
    return dist.sample((dim, 1))

def sample_student_t_vec(df: float, mean_scalar: float, scale_scalar: float, dim: int) -> torch.Tensor:
    """
    Sample a (dim, 1) vector from a Student-t distribution with
    degrees of freedom df, location mean_scalar, and scale scale_scalar.
    """
    # Treat df=inf (tailedness=0) as Normal.
    if df is None or (isinstance(df, float) and (np.isinf(df) or df >= 1e6)):
        return sample_normal_vec(mean_scalar=mean_scalar, std_scalar=scale_scalar, dim=dim)
    dist = torch.distributions.StudentT(df=float(df), loc=mean_scalar, scale=scale_scalar)
    return dist.sample((dim, 1))


def generate_logistic_task(w, sequence_length, noise_std, x_mean=0.0, x_std=1.0):
    """Generate a single input sequence (corresponding to a single logistic task vector w)."""
    x = torch.empty(sequence_length, w.shape[0]).normal_(x_mean, x_std)
    logits = x @ w
    if noise_std > 0:
        logits = logits + torch.randn_like(logits) * noise_std
    probabilities = torch.sigmoid(logits)
    y = torch.bernoulli(probabilities)
    return x, y

def bernoulli_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL( Bern(p) || Bern(q) ) elementwise; safe at 0/1"""
    assert p.shape == q.shape and p.shape[-1] == 1, "p and q must have the same shape and last dimension must be 1"
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    kl = torch.special.xlogy(p, p / q) + torch.special.xlogy(1 - p, (1 - p) / (1 - q))
    return kl.mean() if kl.ndim > 0 else kl

def bernoulli_tv(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    TV( Bern(p), Bern(q) ) = 0.5 * sum_x |P(x)-Q(x)| = |p - q|
    Returns elementwise absolute difference; same shape as p/q.
    """
    assert p.shape == q.shape and p.shape[-1] == 1, "p and q must have the same shape and last dimension must be 1"
    tv = (p - q).abs()
    return tv.mean() if tv.ndim > 0 else tv

def _mcmc_predict_proba_xquery(x_query: torch.Tensor, w_samples: torch.Tensor) -> torch.Tensor:
    """
    x_query: [N, D] torch.Tensor
    w_samples: [S, D] or any shape that flattens per-sample to D (e.g. [S,1,D], [S,D,1], [S,1,D,1])
    Returns: probs [N] torch.Tensor on the same device as w_samples
    """
    xq = x_query.to(w_samples.device)
    X_DIM = xq.shape[-1]
    # Normalize w_samples to shape [S, D]
    if w_samples.ndim >= 3:
        w_samples = w_samples.reshape(-1, X_DIM)  # [S, D]
    elif w_samples.ndim == 2:
        assert w_samples.shape[-1] == X_DIM
        pass  # already [S, D]
    else:  # [D]
        assert w_samples.shape[-1] == X_DIM
        w_samples = w_samples.view(1, X_DIM)
    logits = xq @ w_samples.T      # [N, S]
    probs  = torch.sigmoid(logits).mean(dim=1).clamp(1e-6, 1 - 1e-6)  # [N]
    return probs.unsqueeze(-1) # [N, 1]

# device helpers
def as_tensor_on(x, device, dtype=None):
    """Make x a torch.Tensor on `device` (no-op if already correct)."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype) if (x.device != device or (dtype and x.dtype != dtype)) else x
    return torch.as_tensor(x, device=device, dtype=dtype)

def same_device(*tensors_or_models):
    """Infer a device from the first torch.nn.Module or Tensor provided."""
    for obj in tensors_or_models:
        if isinstance(obj, torch.nn.Module):
            return next(obj.parameters()).device
        if isinstance(obj, torch.Tensor):
            return obj.device
    return torch.device("cpu")

"""
------------------------------------ Baseline wrappers ------------------------------------
"""

def run_oracle_mcmc_predictive(x_ctx, y_ctx, x_query, w_mean_oracle: float, w_std_oracle: float = 1.0,
                               num_thinned=ORACLE_NUM_SAMPLES, warmup=ORACLE_WARMUP_STEPS,
                               thinning=ORACLE_THINNING, num_chains=1,
                               prior_family: str = "normal",
                               student_t_df: Optional[float] = None,
                               student_t_df_list: Optional[List[float]] = None):
    """
    "Oracle" simple MCMC baseline.

    Supports:
      - Normal / Laplace with (w_mean_oracle, w_std_oracle)
      - StudentT with either:
          (a) known df: student_t_df
          (b) mixture over dfs: student_t_df_list  (uniform weights) (to match neural icl)
    """
    device = torch.device("cpu")
    dtype = torch.float32
    X = x_ctx.to(device=device, dtype=dtype); y = y_ctx.to(device=device, dtype=dtype); Xq = x_query.to(device=device, dtype=dtype)
    X_DIM = X.shape[-1]
    
    fam = prior_family.lower()
    if fam == "student_t":
        if (student_t_df is None) and (student_t_df_list is None or len(student_t_df_list) == 0):
            raise ValueError("Student-t prior requires student_t_df or non-empty student_t_df_list.")

    nuts = NUTS(oracle_model_for_mcmc, adapt_step_size=True, jit_compile=True)
    num_raw_samples = num_thinned * thinning
    mcmc = MCMC(nuts, num_samples=num_raw_samples, warmup_steps=warmup, num_chains=num_chains)
    pyro.clear_param_store()
    
    mcmc.run(X, y, X_DIM, 
             w_mean_oracle, w_std_oracle, 
             prior_family, 
             student_t_df,
             student_t_df_list)

    w_samples = mcmc.get_samples()["w"]
    w_thin  = w_samples[::thinning].contiguous() if thinning > 1 else w_samples.contiguous()
    w_thin  = w_thin[:num_thinned]

    probs = _mcmc_predict_proba_xquery(Xq, w_thin)    # [N, 1]
    p_star = probs
    return w_thin, p_star


def run_oracle_mcmc_predictive_spiral_known_theta(
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    x_query: torch.Tensor,
    base_mean: float,
    base_scale: float,
    transform_seed: int,
    *,
    num_thinned: int = ORACLE_NUM_SAMPLES,
    warmup: int = ORACLE_WARMUP_STEPS,
    thinning: int = ORACLE_THINNING,
    num_chains: int = 1,
    speed: float = 1.0,
    random_initial: bool = False,
    initial_scale: float = 0.1,
    jit_compile: bool = False,  
):
    """
    SpiralFlow oracle baseline where the transform parameters (theta) are known.

    Uses `oracle_model_for_mcmc_spiral_known_theta`, which:
      - samples z ~ Normal(base_mean, base_scale)
      - sets w = SpiralFlow_theta(z) with theta determined by transform_seed
      - conditions on (x_ctx, y_ctx)

    Returns:
      w_thin: [S, D]
      probs:  [N, 1] posterior predictive mean Bernoulli probs over x_query
    """
    device = torch.device("cpu")
    dtype = torch.float32

    X = x_ctx.to(device=device, dtype=dtype)
    Y = y_ctx.to(device=device, dtype=dtype)
    Xq = x_query.to(device=device, dtype=dtype)
    X_DIM = int(X.shape[-1])

    nuts = NUTS(
        oracle_model_for_mcmc_spiral_known_theta,
        adapt_step_size=True,
        jit_compile=bool(jit_compile),
    )
    num_raw_samples = int(num_thinned * thinning)
    mcmc = MCMC(
        nuts,
        num_samples=int(num_raw_samples),
        warmup_steps=int(warmup),
        num_chains=int(num_chains),
    )

    pyro.clear_param_store()
    mcmc.run(
        X,
        Y,
        int(X_DIM),
        float(base_mean),
        float(base_scale),
        int(transform_seed),
        float(speed),
        bool(random_initial),
        float(initial_scale),
    )

    samples = mcmc.get_samples()
    if "w" in samples:
        w_samples = samples["w"]  # [S, D]
    elif "z_main" in samples:
        z_samples = samples["z_main"]  # [S, D]
        A_skew, initial = _spiral_params_from_seed(
            int(X_DIM),
            int(transform_seed),
            speed=float(speed),
            random_initial=bool(random_initial),
            initial_scale=float(initial_scale),
            device=device,
            dtype=dtype,
        )
        w_samples = spiral_pushforward(z_samples, A_skew, initial=initial)  # [S, D]
    else:
        raise KeyError(
            f"MCMC samples keys: {list(samples.keys())}. Expected 'w' or 'z_main'."
        )

    w_thin = w_samples[::thinning].contiguous()[: int(num_thinned)]

    probs = _mcmc_predict_proba_xquery(Xq, w_thin)  # [N,1]
    return w_thin, probs

# Hierarchical 'oracle' that *does not* know the true test prior
def run_oracle_hierarchical_mcmc_predictive(
    prior_tasks,                     # list[(X_p, y_p)]
    x_ctx, y_ctx,                    # target context
    x_query,                         # query grid
    prior_mean_min: float = -8.0,    # hyperprior range for mu
    prior_mean_max: float =  8.0,
    num_thinned=ORACLE_NUM_SAMPLES, 
    warmup=ORACLE_WARMUP_STEPS, 
    thinning=ORACLE_THINNING, 
    num_chains=1,
    prior_family: str = "normal",
    student_t_df: Optional[float] = None,
    pyro_model_function = oracle_model_for_hierarchical_mcmc,
):
    """
    A fairer comparison with the neural models.
    Near-oracle that *learns* the prior distribution p(w) (mu) from all prior tasks + target context
    using the hierarchical model. 
    Returns mean posterior predictive probabilities [N,1].
    """
    device = torch.device("cpu")
    dtype = torch.float32
    prior_tasks_X = [x_p.to(device=device, dtype=dtype) for x_p, _ in prior_tasks]
    prior_tasks_y = [y_p.to(device=device, dtype=dtype) for _, y_p in prior_tasks]
    
    X  = x_ctx.to(device=device, dtype=dtype)
    Y  = y_ctx.to(device=device, dtype=dtype)
    Xq = x_query.to(device=device, dtype=dtype)
    X_DIM = X.shape[-1]
    nuts = NUTS(pyro_model_function,
                adapt_step_size=True, jit_compile=True)
    num_raw_samples = num_thinned * thinning
    mcmc = MCMC(nuts, num_samples=num_raw_samples, warmup_steps=warmup, num_chains=num_chains)
    pyro.clear_param_store()
    if prior_family == "student_t" and student_t_df is None:
        raise ValueError("student_t_df must be provided when using a Student-t prior.")
    mcmc.run(prior_tasks_X, prior_tasks_y, X, Y, X_DIM, prior_mean_min, prior_mean_max,
             prior_family, student_t_df)
    w_samples = mcmc.get_samples()["w"]   # [S, D]
    w_thin  = w_samples[::thinning].contiguous()
    w_thin  = w_thin[:num_thinned]
        
    probs = _mcmc_predict_proba_xquery(Xq, w_thin)    # [N, 1]
    p_star = probs
    return w_thin, p_star


def run_oracle_hierarchical_df_mcmc_predictive(
    prior_tasks,                      # list[(X_p, y_p)]
    x_ctx,
    y_ctx,
    x_query,
    tau_eps: float = 1e-3,
    prior_mean_min: float = -8.0,
    prior_mean_max: float = 8.0,
    num_thinned: int = ORACLE_NUM_SAMPLES,
    warmup: int = ORACLE_WARMUP_STEPS,
    thinning: int = ORACLE_THINNING,
    num_chains: int = 1,
):
    """
    Hierarchical Student-t df inference baseline (MCMC-hier-t).
      - infers tau ~ Uniform(0,1), df = 1/(tau + tau_eps)
      - infers mu_scalar ~ Uniform(prior_mean_min, prior_mean_max), broadcasts to mu in R^D
      - fixes w scale=1 
      - df shared across tasks in the episode
    """
    device = torch.device("cpu")
    dtype = torch.float32

    prior_tasks_X = [x_p.to(device=device, dtype=dtype) for x_p, _ in prior_tasks]
    prior_tasks_y = [y_p.to(device=device, dtype=dtype) for _, y_p in prior_tasks]

    X = x_ctx.to(device=device, dtype=dtype)
    Y = y_ctx.to(device=device, dtype=dtype)
    Xq = x_query.to(device=device, dtype=dtype)
    X_DIM = X.shape[-1]

    nuts = NUTS(oracle_model_for_hierarchical_df_mcmc, adapt_step_size=True)
    num_raw_samples = int(num_thinned) * int(thinning)
    mcmc = MCMC(nuts, num_samples=num_raw_samples, warmup_steps=int(warmup), num_chains=int(num_chains))
    pyro.clear_param_store()

    mcmc.run(
        prior_tasks_X, prior_tasks_y,
        X, Y,
        int(X_DIM),
        float(tau_eps),
        float(prior_mean_min),
        float(prior_mean_max),
    )

    samples = mcmc.get_samples()  # contains 'w' and also 'tau' and 'mu_scalar'
    w_samples = samples["w"]
    w_thin = w_samples[::thinning].contiguous() if thinning > 1 else w_samples.contiguous()
    w_thin = w_thin[:num_thinned]

    probs = _mcmc_predict_proba_xquery(Xq, w_thin)
    return w_thin, probs, samples  


def run_oracle_hierarchical_df_mixture_mu_mcmc_predictive(
    prior_tasks,                      # list[(X_p, y_p)]
    x_ctx,
    y_ctx,
    x_query,
    student_t_df_list: List[float],
    prior_mean_min: float = -8.0,
    prior_mean_max: float = 8.0,
    num_thinned: int = ORACLE_NUM_SAMPLES,
    warmup: int = ORACLE_WARMUP_STEPS,
    thinning: int = ORACLE_THINNING,
    num_chains: int = 1,
):
    """
    Hierarchical baseline that jointly infers:
      - mu_scalar ~ Uniform(prior_mean_min, prior_mean_max)
      - df shared across tasks, selected from a discrete df list (mixture), marginalized via MixtureSameFamily
    and returns posterior predictive probabilities for the *target task* w.
    """
    if student_t_df_list is None or len(student_t_df_list) == 0:
        raise ValueError("student_t_df_list must be a non-empty list of df values.")

    device = torch.device("cpu")
    dtype = torch.float32

    prior_tasks_X = [x_p.to(device=device, dtype=dtype) for x_p, _ in prior_tasks]
    prior_tasks_y = [y_p.to(device=device, dtype=dtype) for _, y_p in prior_tasks]

    X = x_ctx.to(device=device, dtype=dtype)
    Y = y_ctx.to(device=device, dtype=dtype)
    Xq = x_query.to(device=device, dtype=dtype)
    X_DIM = int(X.shape[-1])

    nuts = NUTS(oracle_model_for_hierarchical_df_mixture_mu_mcmc, adapt_step_size=True, jit_compile=True)
    num_raw_samples = int(num_thinned) * int(thinning)
    mcmc = MCMC(nuts, num_samples=num_raw_samples, warmup_steps=int(warmup), num_chains=int(num_chains))
    pyro.clear_param_store()

    mcmc.run(
        prior_tasks_X, prior_tasks_y,
        X, Y,
        int(X_DIM),
        [float(x) for x in student_t_df_list],
        float(prior_mean_min),
        float(prior_mean_max),
    )

    samples = mcmc.get_samples()
    w_all = samples["w_all"]  # [S, T*D]

    # Extract target-task weights: last block of size D
    D = int(X_DIM)
    w_target = w_all[..., -D:]  # [S, D]

    w_thin = w_target[::thinning].contiguous() if thinning > 1 else w_target.contiguous()
    w_thin = w_thin[: int(num_thinned)]

    probs = _mcmc_predict_proba_xquery(Xq, w_thin)
    return w_thin, probs, samples
    
def run_svi_curve(
    x_ctx,
    y_ctx,
    x_query,
    prior_mean,
    prior_scale,
    steps_set,
    svi_posterior_samples=200,
    prior_family: str = "normal",
    student_t_df: Optional[float] = None,
    student_t_df_list: Optional[List[float]] = None,
):
    # Prepare data
    device = torch.device("cpu")
    dtype = torch.float32
    X = x_ctx.to(device=device, dtype=dtype); Y = y_ctx.to(device=device, dtype=dtype)
    X_DIM = X.shape[-1]
    
    # Define model
    def model_fn_simple():
        return _simple_classification_model_eval(
            X,
            Y,
            X_DIM,
            prior_mean,
            prior_scale,
            prior_family=prior_family,
            student_t_df=student_t_df,
            student_t_df_list=student_t_df_list,
        )
    
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model_fn_simple)
    svi = SVI(model_fn_simple, guide, pyro_optim.Adam({"lr": 1e-2}), loss=Trace_ELBO())
    q_probs_svi = []
    for t in range(1, max(steps_set) + 1):
        svi.step()
        if t in steps_set:
            predictive = Predictive(
                model_fn_simple,
                guide=guide,
                num_samples=svi_posterior_samples,
                return_sites=("w",),
            )
            samples = predictive()
            w = samples["w"]
       
            if w.ndim == 3:
                if w.size(-1) == 1:
                    w = w.squeeze(-1)
                elif w.size(-2) == 1:
                    w = w.squeeze(-2)
            q_probs_svi.append(_mcmc_predict_proba_xquery(
                x_query.to(device=device, dtype=dtype), w))

    return q_probs_svi # [len(K_LIST), num_queries, 1]

def run_hier_svi_curve(
    x_ctx,
    y_ctx,
    x_query,
    prior_tasks_data,
    steps_set,
    hier_prior_mean_min=-8.0,
    hier_prior_mean_max=8.0,
    svi_posterior_samples=200,
    prior_family: str = "normal",
    student_t_df: Optional[float] = None,
):
    # Prepare data
    device = torch.device("cpu")
    dtype = torch.float32
    X = x_ctx.to(device=device, dtype=dtype); Y = y_ctx.to(device=device, dtype=dtype)
    X_DIM = X.shape[-1]
    
    prior_tasks_X = [xp.to(device=device, dtype=dtype) for xp, _ in prior_tasks_data]
    prior_tasks_y = [yp.to(device=device, dtype=dtype) for _, yp in prior_tasks_data]
    
    # Define model
    def model_fn_hier():
        return _hier_classification_model_eval(
            prior_tasks_X,
            prior_tasks_y,
            X,
            Y,
            X_DIM,
            hier_prior_mean_min,
            hier_prior_mean_max,
            prior_family=prior_family,
            student_t_df=student_t_df,
        )
        
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model_fn_hier)
    svi = SVI(model_fn_hier, guide, pyro_optim.Adam({"lr": 1e-2}), loss=Trace_ELBO())
    
    q_probs_svi_hier = []
    for t in range(1, max(steps_set) + 1):
        svi.step()
        if t in steps_set:
            predictive = Predictive(
                model_fn_hier,
                guide=guide,
                num_samples=svi_posterior_samples,
                return_sites=("w",),
            )
            samples = predictive()
            w = samples["w"]
            if w.ndim == 3:
                if w.size(-1) == 1:
                    w = w.squeeze(-1)
                elif w.size(-2) == 1:
                    w = w.squeeze(-2)
            q_probs_svi_hier.append(_mcmc_predict_proba_xquery(
                x_query.to(device=device, dtype=dtype), w
            ))
    return q_probs_svi_hier # [len(K_LIST), num_queries, 1]

def run_hier_df_svi_curve(
    x_ctx,
    y_ctx,
    x_query,
    prior_tasks_data,
    steps_set,
    tau_eps: float = 1e-3,
    prior_mean_min: float = -8.0,
    prior_mean_max: float = 8.0,
    svi_posterior_samples: int = 500,
):
    """
    Hierarchical df inference SVI baseline (SVI-hier-t), matching _hier_df_model_eval.
      - infers tau -> df
      - infers mu_scalar (shared across dims/tasks)
      - fixes scale=1
    """
    device = torch.device("cpu")
    dtype = torch.float32  

    X = x_ctx.to(device=device, dtype=dtype)  
    Y = y_ctx.to(device=device, dtype=dtype)  
    X_DIM = X.shape[-1]

    prior_tasks_X = [xp.to(device=device, dtype=dtype) for xp, _ in prior_tasks_data]  
    prior_tasks_y = [yp.to(device=device, dtype=dtype) for _, yp in prior_tasks_data]  

    def model_fn_hier_df():
        return _hier_df_model_eval(
            prior_tasks_X, prior_tasks_y,
            X, Y,
            x_dim=int(X_DIM),
            tau_eps=float(tau_eps),
            prior_mean_min=float(prior_mean_min),
            prior_mean_max=float(prior_mean_max),
            prior_family="student_t",
        )
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model_fn_hier_df)
    svi = SVI(model_fn_hier_df, guide, pyro_optim.Adam({"lr": 1e-2}), loss=Trace_ELBO())

    q_probs = []
    max_step = int(max(steps_set))
    steps_set = set(int(s) for s in steps_set)  
    for t in range(1, max_step + 1):
        svi.step()
        if t in steps_set:
            predictive = Predictive(
                model_fn_hier_df,
                guide=guide,
                num_samples=int(svi_posterior_samples),
                return_sites=("w",),
            )
            samples = predictive()
            w = samples["w"]
            if w.ndim == 3:
                if w.size(-1) == 1:
                    w = w.squeeze(-1)
                elif w.size(-2) == 1:
                    w = w.squeeze(-2)
            q_probs.append(_mcmc_predict_proba_xquery(x_query.to(device=device, dtype=dtype), w))  

    return q_probs


def run_hier_df_mixture_mu_svi_curve(
    x_ctx,
    y_ctx,
    x_query,
    prior_tasks_data,
    student_t_df_list: List[float],
    steps_set,
    prior_mean_min: float = -8.0,
    prior_mean_max: float = 8.0,
    svi_posterior_samples: int = 200,
):
    """
    SVI baseline matching _hier_df_mixture_mu_model_eval (latent mu + df-from-discrete-list, shared across tasks).
    """
    if student_t_df_list is None or len(student_t_df_list) == 0:
        raise ValueError("student_t_df_list must be a non-empty list of df values.")

    device = torch.device("cpu")
    dtype = torch.float32

    X = x_ctx.to(device=device, dtype=dtype)
    Y = y_ctx.to(device=device, dtype=dtype)
    X_DIM = int(X.shape[-1])

    prior_tasks_X = [xp.to(device=device, dtype=dtype) for xp, _ in prior_tasks_data]
    prior_tasks_y = [yp.to(device=device, dtype=dtype) for _, yp in prior_tasks_data]

    df_list_f = [float(x) for x in student_t_df_list]

    def model_fn_hier_df_mix_mu():
        return _hier_df_mixture_mu_model_eval(
            prior_tasks_X, prior_tasks_y,
            X, Y,
            x_dim=int(X_DIM),
            student_t_df_list=df_list_f,
            prior_mean_min=float(prior_mean_min),
            prior_mean_max=float(prior_mean_max),
        )

    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model_fn_hier_df_mix_mu)
    svi = SVI(model_fn_hier_df_mix_mu, guide, pyro_optim.Adam({"lr": 1e-2}), loss=Trace_ELBO())

    q_probs = []
    max_step = int(max(steps_set))
    steps_set = set(int(s) for s in steps_set)
    for t in range(1, max_step + 1):
        svi.step()
        if t in steps_set:
            predictive = Predictive(
                model_fn_hier_df_mix_mu,
                guide=guide,
                num_samples=int(svi_posterior_samples),
                return_sites=("w_all",),
            )
            samples = predictive()
            w_all = samples["w_all"]  # [S, T*D]
            w_target = w_all[..., -int(X_DIM):]  # [S, D]
            q_probs.append(_mcmc_predict_proba_xquery(x_query.to(device=device, dtype=dtype), w_target))

    return q_probs

def run_oracle_hierarchical_mcmc_predictive_spiral(
    prior_tasks,                     # list[(X_p, y_p)]
    x_ctx, y_ctx,                    # target context
    x_query,                         # query grid
    prior_mean_min: float = -8.0,
    prior_mean_max: float =  8.0,
    num_thinned: int = 1000,
    warmup: int = 1000,
    thinning: int = 10,
    num_chains: int = 1,
    A_scale: float = 1.0,
    speed: float = 1.0,
    jit_compile: bool = False,       
    pyro_model_function = oracle_model_for_hierarchical_mcmc_spiral,
):
    device = torch.device("cpu")
    dtype = torch.float32

    prior_tasks_X = [x_p.to(device=device, dtype=dtype) for x_p, _ in prior_tasks]
    prior_tasks_y = [y_p.to(device=device, dtype=dtype) for _, y_p in prior_tasks]

    X  = x_ctx.to(device=device, dtype=dtype)
    Y  = y_ctx.to(device=device, dtype=dtype)
    Xq = x_query.to(device=device, dtype=dtype)
    X_DIM = X.shape[-1]

    nuts = NUTS(
        pyro_model_function,
        adapt_step_size=True,
        jit_compile=bool(jit_compile),
    )
    num_raw_samples = int(num_thinned * thinning)
    mcmc = MCMC(
        nuts,
        num_samples=num_raw_samples,
        warmup_steps=int(warmup),
        num_chains=int(num_chains),
    )

    pyro.clear_param_store()
    mcmc.run(
        prior_tasks_X, prior_tasks_y,
        X, Y,
        int(X_DIM), float(prior_mean_min), float(prior_mean_max),
        float(A_scale), float(speed),
    )

    samples = mcmc.get_samples()
    if "w" in samples:
        w_samples = samples["w"]  # [S, D]
    elif ("z_main" in samples) and ("M" in samples):
        # reconstruct w via the same push-forward used in the model:
        #   A_skew = (M - M^T) * speed
        #   w = exp(A_skew * ||z||^2) @ z      (initial = I in the hierarchical spiral model)
        z = samples["z_main"].to(device=device, dtype=dtype)  # [S, D]
        M = samples["M"].to(device=device, dtype=dtype)       # [S, D, D]
        A_skew = (M - M.transpose(-1, -2)) * float(speed)     # [S, D, D]
        r2 = (z ** 2).sum(dim=-1)                             # [S]
        G = A_skew * r2.view(-1, 1, 1)                        # [S, D, D]
        R = torch.matrix_exp(G)                               # [S, D, D]
        w_samples = torch.matmul(R, z.unsqueeze(-1)).squeeze(-1)  # [S, D]
    else:
        raise KeyError(
            f"MCMC samples keys: {list(samples.keys())}. Expected 'w' or ('z_main' and 'M'). "
            f"(If 'w' is only recorded via pyro.deterministic, your Pyro version may omit it from get_samples().)"
        )

    w_thin = w_samples[::thinning].contiguous()[:num_thinned]

    probs = _mcmc_predict_proba_xquery(Xq, w_thin)  # [N,1]
    return w_thin, probs


def run_hier_svi_curve_spiral(
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    x_query: torch.Tensor,
    prior_tasks_data,                     # list[(X_p, y_p)]
    steps_set: Iterable[int],             # e.g. set(K_LIST) or list(K_LIST)
    *,
    hier_prior_mean_min: float = -8.0,
    hier_prior_mean_max: float = 8.0,
    svi_posterior_samples: int = 200,
    lr: float = 1e-2,
    A_scale: float = 1.0,
    speed: float = 1.0,
) -> List[torch.Tensor]:
    """
    SVI "curve" for hierarchical SpiralFlow baseline.

    Returns:
      q_probs_list: list of tensors, one per step in sorted(steps_set).
        each tensor: [num_queries, 1]
    """
    steps = sorted(set(int(s) for s in steps_set))
    if len(steps) == 0:
        raise ValueError("steps_set must be non-empty")
    if steps[0] < 1:
        raise ValueError(f"steps_set must contain positive integers, got min={steps[0]}")

    # Prepare data
    device = torch.device("cpu")
    dtype = torch.float32

    X = x_ctx.to(device=device, dtype=dtype)
    Y = y_ctx.to(device=device, dtype=dtype)
    Xq = x_query.to(device=device, dtype=dtype)
    X_DIM = int(X.shape[-1])

    prior_tasks_X = [xp.to(device=device, dtype=dtype) for xp, _ in prior_tasks_data]
    prior_tasks_y = [yp.to(device=device, dtype=dtype) for _, yp in prior_tasks_data]

    # Model closure with no args
    def model_fn_hier_spiral():
        return _hier_spiral_model_eval(
            prior_tasks_X,
            prior_tasks_y,
            X,
            Y,
            X_DIM,
            float(hier_prior_mean_min),
            float(hier_prior_mean_max),
            A_scale=float(A_scale),
            speed=float(speed),
        )

    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model_fn_hier_spiral)
    svi = SVI(model_fn_hier_spiral, guide, pyro_optim.Adam({"lr": float(lr)}), loss=Trace_ELBO())

    q_probs_list: List[torch.Tensor] = []
    max_steps = int(steps[-1])

    time_start = time.time()
    # Main SVI loop
    for t in range(1, max_steps + 1):
        svi.step()
        if t in steps:
            predictive = Predictive(
                model_fn_hier_spiral,
                guide=guide,
                num_samples=int(svi_posterior_samples),
                return_sites=("w",),   # <-- w is deterministic in the model
            )
            samples = predictive()
            w = samples["w"]  # typically [S, D]

            if w.ndim == 3:
                if w.size(-1) == 1:
                    w = w.squeeze(-1)
                elif w.size(-2) == 1:
                    w = w.squeeze(-2)

            q_probs_list.append(_mcmc_predict_proba_xquery(Xq, w))  # [N,1]
            time_end = time.time()
            print(f"Time taken for step {t}: {time_end - time_start} seconds")

    return q_probs_list  # list length == len(sorted(steps_set))

"""
------------------------------------ Neural model wrappers ------------------------------------
"""
def predict_with_bootstrap(model, prior_tasks, x_target_context_pool, y_target_context_pool,
                           x_target_query, use_task_ids: bool = False, num_bootstraps: int = 100, return_logvar: bool = False):
    model_device = next(model.parameters()).device
    x_dim = x_target_query.shape[-1]
    y_dim = int(getattr(model.hparams, "y_dim", 1))

    task_type = str(getattr(model.hparams, "task_type", "regression")).lower()
    y_special_token = float(Y_PADDING) if task_type == "classification" else 0.0
    
    all_predictions = []
    if return_logvar:                                
        all_mu = []                                  
        all_lv = []  

    # Prepare prior sequences
    x_prior_seqs, y_prior_seqs = [], []
    # If using task ids, also build seg_ids and task_ids in parallel
    if use_task_ids:
        seg_prior_seqs, task_prior_seqs = [], []
        num_prior_tasks = len(prior_tasks)
        
    for t, (x_p, y_p) in enumerate(prior_tasks):
        x_prior_seqs.append(torch.full((1, x_dim), PRIOR_TOKEN_X, device=model_device))
        # Pad y for special-token rows (see task_type note above).
        y_prior_seqs.append(torch.full((1, 1), y_special_token, dtype=torch.float32, device=model_device))
        x_prior_seqs.append(x_p.to(model_device))
        y_prior_seqs.append(y_p.to(model_device))

        if use_task_ids:
            # segment ids: 0 for prior; task ids: 0..K-1 for prior tasks
            seg_prior_seqs.append(torch.zeros(1, dtype=torch.long, device=model_device)) # seg label for special token
            seg_prior_seqs.append(torch.zeros(len(x_p), dtype=torch.long, device=model_device)) # seg label for prior data
            task_prior_seqs.append(torch.full((1,), t, dtype=torch.long, device=model_device)) # task label for special token
            task_prior_seqs.append(torch.full((len(x_p),), t, dtype=torch.long, device=model_device)) # task label for prior data

    if x_prior_seqs:
        x_context_prior = torch.cat(x_prior_seqs, dim=0)
        y_context_prior = torch.cat(y_prior_seqs, dim=0)
        if use_task_ids:
            seg_context_prior = torch.cat(seg_prior_seqs, dim=0)
            task_context_prior = torch.cat(task_prior_seqs, dim=0)
    else: # CAUTION: no prior given
        x_context_prior = torch.empty(0, x_dim, device=model_device)
        y_context_prior = torch.empty(0, 1, device=model_device)
        if use_task_ids:
            seg_context_prior = torch.empty(0, dtype=torch.long, device=model_device)
            task_context_prior = torch.empty(0, dtype=torch.long, device=model_device)

    x_target_context_pool = x_target_context_pool.to(model_device)
    y_target_context_pool = y_target_context_pool.to(model_device)
    x_target_query = x_target_query.to(model_device)

    with torch.no_grad():
        for _ in range(num_bootstraps):
            if num_bootstraps == 1:
                x_boot_context = x_target_context_pool
                y_boot_context = y_target_context_pool
            else:
                indices = np.random.choice(len(x_target_context_pool), size=len(x_target_context_pool), replace=True)
                x_boot_context = x_target_context_pool[indices]
                y_boot_context = y_target_context_pool[indices]

            # Build context (prior + target separator + target context)
            x_full_context = torch.cat([
                x_context_prior,
                torch.full((1, x_dim), TARGET_TOKEN_X, device=model_device),
                x_boot_context
            ], dim=0)
            y_full_context = torch.cat([
                y_context_prior,
                # Pad y for special-token rows (see task_type note above).
                torch.full((1, 1), y_special_token, dtype=torch.float32, device=model_device),
                y_boot_context
            ], dim=0)

            if not use_task_ids or getattr(model.hparams, 'identity_dim', 0) <= 0:
                # Fallback to standard independent prediction without seg/task ids
                
                y_pred = model.predict_independent(x_full_context, y_full_context, x_target_query)
                if return_logvar:                     
                    # Inline forward pass to extract logvar
                    batch_size = len(x_target_query)
                    x_context_batch = x_full_context.unsqueeze(0).expand(batch_size, -1, -1)
                    y_context_batch = y_full_context.unsqueeze(0).expand(batch_size, -1, -1)
                    x_seq_batch = torch.cat([x_context_batch, x_target_query.unsqueeze(1)], dim=1)
                    # Placeholder y at query position (not consumed by _prepare_input_sequence for inputs).
                    y_placeholder_batch = torch.full(
                        (batch_size, 1, y_dim),
                        y_special_token,
                        dtype=torch.float32,
                        device=model_device,
                    )
                    y_seq_batch = torch.cat([y_context_batch, y_placeholder_batch], dim=1)

                    model_input, _ = model._prepare_input_sequence(x_seq_batch, y_seq_batch)
                    mu_all, logvar_all, _ = model(model_input)

                    mu_pred = mu_all[:, -1, :]
                    lv_pred = logvar_all[:, -1, :]
                    if mu_pred.ndim == 1:
                        mu_pred = mu_pred.unsqueeze(-1)
                    if lv_pred.ndim == 1:
                        lv_pred = lv_pred.unsqueeze(-1)
                    # Keep tensors (not numpy) so torch.stack works.
                    all_mu.append(mu_pred.detach().cpu())
                    all_lv.append(lv_pred.detach().cpu())
            else:
                # NOTE: use_task_ids and identity_dim > 0
                # Build seg_ids (0 for prior region, 1 for target region inc. separator/query)
                seg_full_context = torch.cat([
                    seg_context_prior,
                    torch.ones(1, dtype=torch.long, device=model_device),
                    torch.ones(len(x_boot_context), dtype=torch.long, device=model_device)
                ], dim=0)
                # Build task_ids (0..K-1 for prior tasks, K for target)
                K = len(prior_tasks)
                task_full_context = torch.cat([
                    task_context_prior,
                    torch.full((1,), K, dtype=torch.long, device=model_device),
                    torch.full((len(x_boot_context),), K, dtype=torch.long, device=model_device)
                ], dim=0)

                # Batch queries as in model.predict_independent
                batch_size = len(x_target_query)
                x_context_batch = x_full_context.unsqueeze(0).expand(batch_size, -1, -1)
                y_context_batch = y_full_context.unsqueeze(0).expand(batch_size, -1, -1)
                seg_batch = seg_full_context.unsqueeze(0).expand(batch_size, -1)
                task_batch = task_full_context.unsqueeze(0).expand(batch_size, -1)

                # Append query step
                x_seq_batch = torch.cat([x_context_batch, x_target_query.unsqueeze(1)], dim=1)
                # Placeholder y at query position (not consumed by _prepare_input_sequence for inputs).
                y_placeholder_batch = torch.full(
                    (batch_size, 1, getattr(model.hparams, 'y_dim', 1)),
                    y_special_token,
                    dtype=torch.float32,
                    device=model_device,
                )
                y_seq_batch = torch.cat([y_context_batch, y_placeholder_batch], dim=1)
                seg_seq_batch = torch.cat([seg_batch, torch.ones(batch_size, 1, dtype=torch.long, device=model_device)], dim=1)
                task_seq_batch = torch.cat([task_batch, torch.full((batch_size, 1), K, dtype=torch.long, device=model_device)], dim=1)

                # Build model input with seg/task ids and predict
                model_input, _ = model._prepare_input_sequence(x_seq_batch, y_seq_batch, seg_seq_batch, task_seq_batch)
                output1, _, _ = model(model_input)
                # Extract last-step prediction
                if getattr(model.hparams, 'task_type', 'regression') == 'regression':
                    y_pred = output1[:, -1, :]
                else:
                    y_pred = output1[:, -1, :]
                if y_pred.ndim == 1:
                    y_pred = y_pred.unsqueeze(-1)
            ### end of conditional logic for seg/task ids ###
            all_predictions.append(y_pred)

    if not return_logvar:
        return torch.stack(all_predictions, dim=0) # [num_bootstraps, num_queries, y_dim] (logits for classification)

    mu_hat = torch.stack(all_mu, dim=0).mean(dim=0)
    lv_hat = torch.stack(all_lv, dim=0).mean(dim=0)
    return mu_hat, lv_hat

# -------------------------
# Helpers for permutation-sensitivity KL only
# -------------------------
from src.utils.linear_baselines import gaussian_kl
from typing import List, Dict, Any
def _compute_gaussian_kl_vs_oracle(
    *,
    oracle_mu: np.ndarray,
    oracle_lv: np.ndarray,
    pred_mu: np.ndarray,
    pred_lv: np.ndarray,
) -> float:
    """Mean KL(oracle || pred) across query points."""
    return float(np.mean(gaussian_kl(oracle_mu, oracle_lv, pred_mu, pred_lv)))


def _compute_symmetric_gaussian_kl_between_preds(
    *,
    mu_a: np.ndarray,
    lv_a: np.ndarray,
    mu_b: np.ndarray,
    lv_b: np.ndarray,
) -> float:
    """
    Symmetric mean KL between two Gaussian PPDs:
      0.5 * ( KL(a || b) + KL(b || a) ),
    averaged across query points.
    """
    kl_ab = float(np.mean(gaussian_kl(mu_a, lv_a, mu_b, lv_b)))
    kl_ba = float(np.mean(gaussian_kl(mu_b, lv_b, mu_a, lv_a)))
    return 0.5 * (kl_ab + kl_ba)


def _summarize_prediction_set_kl_only(
    trial_preds: List[Dict[str, Any]],
    oracle_mu: np.ndarray,
    oracle_lv: np.ndarray,
) -> Dict[str, float]:
    """
    For one fixed original example, summarize permutation sensitivity over N permuted prefixes.

    trial_preds: list of dicts, each with
      {
        "trial_idx": int,
        "seed": int,
        "mu": np.ndarray [Q],
        "lv": np.ndarray [Q],
      }

    Returns per-example summary:
      - oracle_kl_mean / oracle_kl_std:
          KL to oracle across permutations
      - pairwise_sym_kl_mean / pairwise_sym_kl_std:
          average pairwise symmetric KL among model PPDs from different permutations
    """
    out: Dict[str, float] = {"num_trials": int(len(trial_preds))}

    if len(trial_preds) == 0:
        out["oracle_kl_mean"] = float("nan")
        out["oracle_kl_std"] = float("nan")
        out["pairwise_sym_kl_mean"] = float("nan")
        out["pairwise_sym_kl_std"] = float("nan")
        return out

    oracle_kls = []
    for pred in trial_preds:
        kl = _compute_gaussian_kl_vs_oracle(
            oracle_mu=oracle_mu,
            oracle_lv=oracle_lv,
            pred_mu=pred["mu"],
            pred_lv=pred["lv"],
        )
        oracle_kls.append(kl)

    oracle_kls = np.asarray(oracle_kls, dtype=np.float64)
    out["oracle_kl_mean"] = float(oracle_kls.mean())
    out["oracle_kl_std"] = float(oracle_kls.std(ddof=1)) if len(oracle_kls) > 1 else float("nan")

    pairwise_sym_kls = []
    T = len(trial_preds)
    for i in range(T):
        for j in range(i + 1, T):
            skl = _compute_symmetric_gaussian_kl_between_preds(
                mu_a=trial_preds[i]["mu"],
                lv_a=trial_preds[i]["lv"],
                mu_b=trial_preds[j]["mu"],
                lv_b=trial_preds[j]["lv"],
            )
            pairwise_sym_kls.append(skl)

    if len(pairwise_sym_kls) == 0:
        out["pairwise_sym_kl_mean"] = float("nan")
        out["pairwise_sym_kl_std"] = float("nan")
    else:
        pairwise_sym_kls = np.asarray(pairwise_sym_kls, dtype=np.float64)
        out["pairwise_sym_kl_mean"] = float(pairwise_sym_kls.mean())
        out["pairwise_sym_kl_std"] = (
            float(pairwise_sym_kls.std(ddof=1)) if len(pairwise_sym_kls) > 1 else float("nan")
        )

    return out
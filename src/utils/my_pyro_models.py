import torch
import pyro
import pyro.distributions as dist
from typing import Optional, List, Union, Dict, Any, Tuple

from src.data.transforms import spiral_pushforward, _spiral_params_from_seed
from src.data.f_theta import LikelihoodType, apply_f_theta

SUPPORTED_PRIOR_FAMILIES = {"normal", "laplace", "student_t"}

# ---------------------------------- prior helper functions ----------------------------------
def _make_prior_distribution(
    mu: torch.Tensor,
    scale: torch.Tensor,
    prior_family: str,
    student_t_df: Optional[Union[float, torch.Tensor]] = None,
    student_t_df_list: Optional[List[float]] = None,
) -> dist.Distribution:
    family = prior_family.lower()
    if family not in SUPPORTED_PRIOR_FAMILIES:
        raise ValueError(
            f"Unsupported prior distribution '{prior_family}'. "
            f"Supported families: {sorted(SUPPORTED_PRIOR_FAMILIES)}"
        )
    if family == "normal":
        return dist.Normal(mu, scale).to_event(1)

    if family == "laplace":
        return dist.Laplace(mu, scale).to_event(1)

    # student_t
    if student_t_df_list is not None and len(student_t_df_list) > 0:
        return _make_student_t_mixture_prior_distribution(mu, scale, student_t_df_list)

    if student_t_df is None:
        raise ValueError("student_t_df must be provided when using a Student-t prior (non-mixture).")

    # IMPORTANT: keep df as a tensor if it already is one (for latent df inference)
    if isinstance(student_t_df, torch.Tensor):
        df_tensor = student_t_df.to(device=mu.device, dtype=mu.dtype)
    else:
        df_tensor = torch.as_tensor(float(student_t_df), device=mu.device, dtype=mu.dtype)

    return dist.StudentT(df_tensor, loc=mu, scale=scale).to_event(1)

def _make_student_t_mixture_prior_distribution(
    mu: torch.Tensor, # [D,]
    scale: torch.Tensor, # [D,]
    student_t_df_list: List[float],
) -> dist.Distribution:
    """
    Continuous mixture prior over w:
      w ~ Sum_k pi_k StudentT(df_k, loc=mu, scale=scale)

    Returns distribution with event_dim=1 over D.
    """
    
    device = mu.device
    dtype = mu.dtype

    df = torch.tensor([float(x) for x in student_t_df_list], device=device, dtype=dtype)  # [K]

    K = df.numel()
    mix_probs = torch.full((K,), 1.0 / K, device=device, dtype=dtype)

    # Component distribution: batch_shape [K], event_shape [D]
    loc = mu.unsqueeze(0).expand(K, -1)      # [K, D]
    sc  = scale.unsqueeze(0).expand(K, -1)   # [K, D]

    comp = dist.StudentT(df=df.unsqueeze(-1), loc=loc, scale=sc).to_event(1)  # event over D
    mix  = dist.Categorical(probs=mix_probs)  # over K

    return dist.MixtureSameFamily(mix, comp)

# ---------------------------------- basic classification model ----------------------------------
def _simple_classification_model_eval(target_task_X,
                       target_task_y,
                       x_dim: int,
                       w_mean: float,
                       w_std: float = 1.0,
                       prior_family: str = "normal",
                       student_t_df: Optional[float] = None,
                       student_t_df_list: Optional[List[float]] = None,
                       likelihood: LikelihoodType = "identity",
                       theta: Dict[str, Any] = None,
                       logit_scale: float = 1.0):
    device = target_task_X.device
    dtype = target_task_X.dtype
    
    mu = torch.full((x_dim,), float(w_mean), device=device, dtype=dtype)
    sigma = torch.full((x_dim,), float(w_std), device=device, dtype=dtype)
    w_prior = _make_prior_distribution(mu, sigma, prior_family, student_t_df, student_t_df_list)
    w = pyro.sample("w", w_prior)
    
    y_obs = target_task_y.squeeze(-1) if target_task_y.ndim == 2 else target_task_y
    with pyro.plate("data_main", target_task_X.size(0)):
        s = target_task_X @ w
        logits = float(logit_scale) * apply_f_theta(s, likelihood=likelihood, theta=theta)
        pyro.sample("obs_main", dist.Bernoulli(logits=logits), obs=y_obs)
        
def _hier_classification_model_eval(prior_tasks_X, prior_tasks_y, target_task_X, target_task_y,
                     x_dim: int, prior_mean_min: float, prior_mean_max: float,
                     prior_family: str = "normal",
                     student_t_df: Optional[float] = None,
                     likelihood: LikelihoodType = "identity",
                     theta: Dict[str, Any] = None,
                     logit_scale: float = 1.0,
                     w_std: float = 1.0):
    device = target_task_X.device
    dtype = target_task_X.dtype
    
    mu_scalar = pyro.sample("mu_scalar", dist.Uniform(
        torch.tensor(prior_mean_min, device=device),
        torch.tensor(prior_mean_max, device=device)
    ))
    mu = mu_scalar.expand(x_dim)
    sigma = torch.full((x_dim,), float(w_std), device=device, dtype=dtype)
    # NOTE: extra (correct) assumptions in hierarchical mcmc:
    # (1. Normal distribution w)
    # 2. mu is a scalar, expanded to a vector
    # 3. Fixed sigma as identity matrix
    
    K = len(prior_tasks_X)
    for k in range(K):
        Xk = prior_tasks_X[k]
        yk = prior_tasks_y[k]
        if yk.ndim == 2:
            yk = yk.squeeze(-1)
        yk = yk.to(dtype=Xk.dtype)
        w_prior = _make_prior_distribution(mu, sigma, prior_family, student_t_df)
        w_k = pyro.sample(f"w_k_{k}", w_prior)
        with pyro.plate(f"data_k_{k}", Xk.size(0)):
            s = Xk @ w_k
            logits_k = float(logit_scale) * apply_f_theta(s, likelihood=likelihood, theta=theta)
            pyro.sample(f"obs_k_{k}", dist.Bernoulli(logits=logits_k), obs=yk)

    y_main = target_task_y
    if y_main.ndim == 2:
        y_main = y_main.squeeze(-1)
    y_main = y_main.to(dtype=target_task_X.dtype)
    main_w_prior = _make_prior_distribution(mu, sigma, prior_family, student_t_df)
    w = pyro.sample("w", main_w_prior)
    with pyro.plate("data_main", target_task_X.size(0)):
        s = target_task_X @ w
        logits = float(logit_scale) * apply_f_theta(s, likelihood=likelihood, theta=theta)
        pyro.sample("obs_main", dist.Bernoulli(logits=logits), obs=y_main)


# ---------------------------------- basic regression model ----------------------------------
def _simple_regression_model_eval(
    target_task_X,
    target_task_y,
    x_dim: int,
    w_mean: float,
    w_std: float = 1.0,
    prior_family: str = "normal",
    student_t_df: Optional[float] = None,
    likelihood: LikelihoodType = "identity",
    theta: Dict[str, Any] = None,
    noise_std: float = 0.5, 
):
    device = target_task_X.device
    dtype = target_task_X.dtype
    
    mu = torch.full((x_dim,), float(w_mean), device=device, dtype=dtype)
    sigma = torch.full((x_dim,), float(w_std), device=device, dtype=dtype)
    w_prior = _make_prior_distribution(mu, sigma, prior_family, student_t_df)
    w = pyro.sample("w", w_prior)
    
    y_obs = target_task_y.squeeze(-1) if target_task_y.ndim == 2 else target_task_y
    with pyro.plate("data_main", target_task_X.size(0)):
        s = target_task_X @ w
        y_loc = apply_f_theta(s, likelihood=likelihood, theta=theta)
        pyro.sample("obs_main", dist.Normal(y_loc, float(noise_std)), obs=y_obs)

def _hier_regression_model_eval(
    prior_tasks_X, prior_tasks_y,
    x_target: torch.Tensor,
    y_target: torch.Tensor,
    x_dim: int,
    noise_std: float,
    hier_prior_mean_min: float,
    hier_prior_mean_max: float,
    likelihood: LikelihoodType = "identity",
    theta: Dict[str, Any] = None,
    w_std: float = 1.0,
):
    """
    Hierarchical model:
        mu_scalar ~ Uniform(min, max)
        w_k ~ Normal(mu_scalar * 1, I)
        y | X, w ~ Normal(apply_f_theta(X w), noise_std)
    """
    device = x_target.device
    dtype = x_target.dtype

    mu_scalar = pyro.sample(
        "mu_scalar",
        dist.Uniform(
            torch.tensor(hier_prior_mean_min, device=device),
            torch.tensor(hier_prior_mean_max, device=device),
        ),
    )
    mu = mu_scalar.expand(x_dim)
    sigma = torch.full((x_dim,), float(w_std), device=device, dtype=dtype)

    # Prior tasks
    K = len(prior_tasks_X)
    for k in range(K):
        Xk = prior_tasks_X[k]
        yk = prior_tasks_y[k]
        if yk.ndim == 2:
            yk = yk.squeeze(-1)
        yk = yk.to(dtype=Xk.dtype)
        w_k = pyro.sample(f"w_k_{k}", dist.Normal(mu, sigma).to_event(1))
        assert w_k.shape == (x_dim,), f"w_k_{k} shape should be ({x_dim},), but got {w_k.shape}"
        with pyro.plate(f"data_k_{k}", Xk.size(0)):
            s_k = Xk @ w_k
            y_loc_k = apply_f_theta(s_k, likelihood=likelihood, theta=theta)
            pyro.sample(
                f"obs_k_{k}",
                dist.Normal(y_loc_k, noise_std),
                obs=yk,
            )

    # Target task
    w_target = pyro.sample("w_target", dist.Normal(mu, sigma).to_event(1))
    y_obs_target = y_target.squeeze(-1) if y_target.ndim == 2 else y_target
    with pyro.plate("data_target", x_target.size(0)):
        s_target = x_target @ w_target
        y_loc_target = apply_f_theta(s_target, likelihood=likelihood, theta=theta)
        pyro.sample(
            "obs_target",
            dist.Normal(y_loc_target, noise_std),
            obs=y_obs_target,
        )

# ---------------------------------- baseline model for prior steering ----------------------------------

def _hier_model_pooled_eval(
    prior_tasks_X, prior_tasks_y,
    target_task_X, target_task_y,
    x_dim: int,
    prior_mean_min: float = -8.0,
    prior_mean_max: float = 8.0,
    prior_family: str = "normal",
    student_t_df=None,
):
    """
    Wrapper-compatible pooled+latent-mu model.

    Assumptions:
      - One *shared* w generates ALL datapoints from (prior tasks + target context).
      - Prior mean mu_scalar is unknown and inferred from all datapoints.
      - Prior covariance is fixed to I (std=1 per dimension).

    Signature matches:
      mcmc.run(prior_tasks_X, prior_tasks_y, X, Y, X_DIM, prior_mean_min, prior_mean_max,
               prior_family, student_t_df)
    """

    device = target_task_X.device
    dtype = target_task_X.dtype

    # ---- pool all data points into a single dataset ----
    Xs = [target_task_X]
    Ys = [target_task_y]
    for Xk, yk in zip(prior_tasks_X, prior_tasks_y):
        Xs.append(Xk)
        Ys.append(yk)

    X_all = torch.cat(Xs, dim=0)  # [N_total, D]
    y_all = torch.cat(
        [(yy if yy.ndim == 1 else yy.squeeze(-1)) for yy in Ys],
        dim=0
    ).to(dtype=X_all.dtype)       # [N_total]

    # ---- infer scalar mu over the range ----
    mu_scalar = pyro.sample(
        "mu_scalar",
        dist.Uniform(
            torch.tensor(float(prior_mean_min), device=device, dtype=dtype),
            torch.tensor(float(prior_mean_max), device=device, dtype=dtype),
        ),
    )
    mu = mu_scalar.expand(int(x_dim))  # [D]

    # ---- fixed std = 1 (I covariance) ----
    sigma = torch.ones(int(x_dim), device=device, dtype=dtype)

    # ---- prior over w, with family support ----
    fam = str(prior_family).lower()
    if fam == "normal":
        w_prior = dist.Normal(mu, sigma).to_event(1)
    elif fam == "laplace":
        w_prior = dist.Laplace(mu, sigma).to_event(1)
    elif fam == "student_t":
        if student_t_df is None:
            raise ValueError("student_t_df must be provided for prior_family='student_t'.")
        # Pyro StudentT uses df, loc, scale; broadcast to [D], then event-dim 1
        df = torch.tensor(float(student_t_df), device=device, dtype=dtype)
        w_prior = dist.StudentT(df=df, loc=mu, scale=sigma).to_event(1)
    else:
        raise ValueError(f"Unknown prior_family={prior_family!r}. Expected normal/laplace/student_t.")

    # ---- single shared w for all pooled datapoints ----
    w = pyro.sample("w", w_prior)  # [D]

    with pyro.plate("data_all", X_all.size(0)):
        logits = X_all @ w  # [N_total]
        pyro.sample("obs_all", dist.Bernoulli(logits=logits), obs=y_all)


# ---------------------------------- skyline model for hierarchical df inference ----------------------------------
def _hier_df_model_eval(
    prior_tasks_X, prior_tasks_y,
    target_task_X, target_task_y,
    x_dim: int,
    tau_eps: float = 1e-3,
    prior_mean_min: float = -8.0,
    prior_mean_max: float = 8.0,
    prior_family: str = "student_t",
):
    """
    Hierarchical model that infers BOTH:
      - Student-t df via tau (tailedness) ~ Uniform(0,1), df = 1/(tau + eps)
      - a shared location (mean) mu_scalar ~ Uniform(prior_mean_min, prior_mean_max)
        which is broadcast across dimensions: mu = mu_scalar * 1_D

    df and mu are shared across all tasks in the episode.

    Assumes:
      - scale = 1
      - prior family is Student-t (for this baseline)
    """
    if prior_family.lower() != "student_t":
        raise ValueError("_hier_df_model_eval is intended for prior_family='student_t' only.")

    device = target_task_X.device
    dtype = target_task_X.dtype

    # ---- hyperprior over tailedness ----
    # tau in (0,1). Sampling exactly 0 has prob 0, but eps prevents blow-ups.
    tau = pyro.sample(
        "tau",
        dist.Uniform(
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(1.0, device=device, dtype=dtype),
        )
    )
    tau_eps = torch.tensor(float(tau_eps), device=device, dtype=dtype)
    df = 1.0 / (tau + tau_eps)  # df in (1/(1+eps), 1/eps]

    # ---- shared loc + fixed scale for w ----
    mu_scalar = pyro.sample(
        "mu_scalar",
        dist.Uniform(
            torch.tensor(float(prior_mean_min), device=device, dtype=dtype),
            torch.tensor(float(prior_mean_max), device=device, dtype=dtype),
        ),
    )
    mu = mu_scalar.expand(int(x_dim))  # [D]
    sigma = torch.ones(x_dim, device=device, dtype=dtype)

    # ---- prior tasks ----
    K = len(prior_tasks_X)
    for k in range(K):
        Xk = prior_tasks_X[k].to(device=device, dtype=dtype)
        yk = prior_tasks_y[k].to(device=device, dtype=dtype)
        if yk.ndim == 2:
            yk = yk.squeeze(-1)

        w_prior = _make_prior_distribution(mu, sigma, prior_family="student_t", student_t_df=df)
        w_k = pyro.sample(f"w_k_{k}", w_prior)

        with pyro.plate(f"data_k_{k}", Xk.size(0)):
            logits_k = Xk @ w_k
            pyro.sample(f"obs_k_{k}", dist.Bernoulli(logits=logits_k), obs=yk)

    # ---- target task ----
    Xt = target_task_X.to(device=device, dtype=dtype)
    yt = target_task_y.to(device=device, dtype=dtype)
    if yt.ndim == 2:
        yt = yt.squeeze(-1)

    w_prior_main = _make_prior_distribution(mu, sigma, prior_family="student_t", student_t_df=df)
    w = pyro.sample("w", w_prior_main)

    with pyro.plate("data_main", Xt.size(0)):
        logits = Xt @ w
        pyro.sample("obs_main", dist.Bernoulli(logits=logits), obs=yt)


# ---------------------------------- hierarchical df-from-mixture + latent loc inference ----------------------------------
def _hier_df_mixture_mu_model_eval(
    prior_tasks_X,
    prior_tasks_y,
    target_task_X,
    target_task_y,
    x_dim: int,
    student_t_df_list: List[float],
    prior_mean_min: float = -8.0,
    prior_mean_max: float = 8.0,
):
    """
    Hierarchical model matching the neural setup where, per *episode* (input sequence):
      - mu_scalar ~ Uniform(prior_mean_min, prior_mean_max)      (shared across all dims)
      - df is selected from a *discrete set* (student_t_df_list), shared across all tasks
      - each task gets its own w_k ~ StudentT(df, loc=mu, scale=1)

    IMPORTANT:
      - To keep this model differentiable for NUTS and simple Auto* guides, we *marginalize*
        the discrete df choice by sampling a single concatenated vector of all task weights
        from a MixtureSameFamily distribution. This is equivalent to a single latent
        component z shared across tasks.
    """
    if student_t_df_list is None or len(student_t_df_list) == 0:
        raise ValueError("student_t_df_list must be a non-empty list of df values.")

    device = target_task_X.device
    dtype = target_task_X.dtype

    # ---- hyperprior over shared loc ----
    mu_scalar = pyro.sample(
        "mu_scalar",
        dist.Uniform(
            torch.tensor(float(prior_mean_min), device=device, dtype=dtype),
            torch.tensor(float(prior_mean_max), device=device, dtype=dtype),
        ),
    )
    mu = mu_scalar.expand(int(x_dim))  # [D]
    sigma = torch.ones(int(x_dim), device=device, dtype=dtype)  # fixed scale=1 per dim

    # ---- shared df from discrete list (marginalized as a joint mixture over all task weights) ----
    df_vals = torch.tensor([float(x) for x in student_t_df_list], device=device, dtype=dtype)  # [K]
    Kmix = int(df_vals.numel())
    mix_probs = torch.full((Kmix,), 1.0 / float(Kmix), device=device, dtype=dtype)

    num_prior_tasks = int(len(prior_tasks_X))
    num_tasks_total = num_prior_tasks + 1  # prior tasks + target task
    w_dim_total = int(num_tasks_total) * int(x_dim)

    loc_all = mu.repeat(int(num_tasks_total))         # [T*D]
    sc_all = sigma.repeat(int(num_tasks_total))       # [T*D]

    # Component distribution: batch [Kmix], event [T*D]
    loc = loc_all.unsqueeze(0).expand(Kmix, -1)       # [Kmix, T*D]
    sc = sc_all.unsqueeze(0).expand(Kmix, -1)         # [Kmix, T*D]
    comp = dist.StudentT(df=df_vals.unsqueeze(-1), loc=loc, scale=sc).to_event(1)
    mix = dist.Categorical(probs=mix_probs)

    w_all_flat = pyro.sample("w_all", dist.MixtureSameFamily(mix, comp))  # [T*D]
    w_all = w_all_flat.reshape(int(num_tasks_total), int(x_dim))          # [T, D]
 
    # ---- prior tasks likelihood ----
    for k in range(num_prior_tasks):
        Xk = prior_tasks_X[k].to(device=device, dtype=dtype)
        yk = prior_tasks_y[k].to(device=device, dtype=dtype)
        if yk.ndim == 2:
            yk = yk.squeeze(-1)
        with pyro.plate(f"data_k_{k}", Xk.size(0)):
            logits_k = Xk @ w_all[k]
            pyro.sample(f"obs_k_{k}", dist.Bernoulli(logits=logits_k), obs=yk)

    # ---- target task likelihood ----
    Xt = target_task_X.to(device=device, dtype=dtype)
    yt = target_task_y.to(device=device, dtype=dtype)
    if yt.ndim == 2:
        yt = yt.squeeze(-1)
    with pyro.plate("data_main", Xt.size(0)):
        logits = Xt @ w_all[-1]
        pyro.sample("obs_main", dist.Bernoulli(logits=logits), obs=yt)


# -------------------------- Bayesian baseline for spiral flow transformed prior ----------------------------------

def _hier_spiral_model_eval(
    prior_tasks_X, prior_tasks_y,
    target_task_X, target_task_y,
    x_dim: int,
    prior_mean_min: float,
    prior_mean_max: float,
    *,
    A_scale: float = 1.0,     # std for entries of M (before skew)
    speed: float = 1.0,       # multiplies A_skew (optional)
):
    device = target_task_X.device
    dtype = target_task_X.dtype

    # ---- episode-level latent: mu_scalar ----
    mu_scalar = pyro.sample(
        "mu_scalar",
        dist.Uniform(
            torch.tensor(prior_mean_min, device=device, dtype=dtype),
            torch.tensor(prior_mean_max, device=device, dtype=dtype),
        ),
    )
    mu = mu_scalar.expand(x_dim)  # [D]

    # ---- episode-level latent: full DxD matrix (via M, then skew) ----
    M = pyro.sample(
        "M",
        dist.Normal(
            torch.zeros(x_dim, x_dim, device=device, dtype=dtype),
            torch.full((x_dim, x_dim), float(A_scale), device=device, dtype=dtype),
        ).to_event(2),
    )
    A_skew = (M - M.transpose(-1, -2)) * float(speed)  # [D,D], skew-symmetric

    initial = torch.eye(x_dim, device=device, dtype=dtype)  # keep fixed for now

    # ---- prior tasks ----
    K = len(prior_tasks_X)
    for k in range(K):
        Xk = prior_tasks_X[k]
        yk = prior_tasks_y[k]
        if yk.ndim == 2:
            yk = yk.squeeze(-1)
        yk = yk.to(dtype=Xk.dtype)

        z_k = pyro.sample(
            f"z_k_{k}",
            dist.Normal(mu, torch.ones(x_dim, device=device, dtype=dtype)).to_event(1),
        )
        w_k = spiral_pushforward(z_k, A_skew, initial=initial)  # [D]
        pyro.deterministic(f"w_k_{k}", w_k)

        with pyro.plate(f"data_k_{k}", Xk.size(0)):
            logits_k = Xk @ w_k
            pyro.sample(f"obs_k_{k}", dist.Bernoulli(logits=logits_k), obs=yk)

    # ---- target task ----
    y_main = target_task_y
    if y_main.ndim == 2:
        y_main = y_main.squeeze(-1)
    y_main = y_main.to(dtype=target_task_X.dtype)

    z_main = pyro.sample(
        "z_main",
        dist.Normal(mu, torch.ones(x_dim, device=device, dtype=dtype)).to_event(1),
    )
    w_main = spiral_pushforward(z_main, A_skew, initial=initial)  # [D]
    pyro.deterministic("w", w_main)  
    pyro.deterministic("w_main", w_main)

    with pyro.plate("data_main", target_task_X.size(0)):
        logits = target_task_X @ w_main
        pyro.sample("obs_main", dist.Bernoulli(logits=logits), obs=y_main)


def _oracle_spiral_known_theta_model_eval(
    target_task_X: torch.Tensor,
    target_task_y: torch.Tensor,
    x_dim: int,
    base_mean: float,
    base_scale: float,
    transform_seed: int,
    *,
    speed: float = 1.0,
    random_initial: bool = False,
    initial_scale: float = 0.1,
):
    """
    Oracle SpiralFlow baseline (known theta).

    Data-generating prior:
      z ~ Normal(base_mean * 1_D, base_scale * 1_D)
      w = SpiralFlow_theta(z)   where theta is fully determined by `transform_seed`

    Inference:
      infer z (and thereby w) given (X, y) for the target task only.

    Notes:
      - We record `w` as a deterministic site so MCMC can return it directly
        without treating it as an HMC parameter.
      - This intentionally does NOT infer theta or the base mean/scale.
    """
    device = target_task_X.device
    dtype = target_task_X.dtype

    A_skew, initial = _spiral_params_from_seed(
        int(x_dim),
        int(transform_seed),
        speed=float(speed),
        random_initial=bool(random_initial),
        initial_scale=float(initial_scale),
        device=device,
        dtype=dtype,
    )

    mu = torch.full((int(x_dim),), float(base_mean), device=device, dtype=dtype)
    sigma = torch.full((int(x_dim),), float(base_scale), device=device, dtype=dtype)

    y_obs = target_task_y.squeeze(-1) if target_task_y.ndim == 2 else target_task_y
    y_obs = y_obs.to(dtype=target_task_X.dtype)

    z = pyro.sample("z_main", dist.Normal(mu, sigma).to_event(1))
    w_main = spiral_pushforward(z, A_skew, initial=initial)  # [D]

    # Make w available in MCMC samples (without sampling it).
    # Using pyro.deterministic avoids NUTS trying to treat `w` as a latent parameter.
    pyro.deterministic("w", w_main)

    with pyro.plate("data_main", target_task_X.size(0)):
        logits = target_task_X @ w_main
        pyro.sample("obs_main", dist.Bernoulli(logits=logits), obs=y_obs)


def oracle_model_for_mcmc_spiral_known_theta(
    target_task_X: torch.Tensor,
    target_task_y: torch.Tensor,
    x_dim: int,
    base_mean: float,
    base_scale: float,
    transform_seed: int,
    speed: float = 1.0,
    random_initial: bool = False,
    initial_scale: float = 0.1,
):
    """
    Picklable top-level wrapper around `_oracle_spiral_known_theta_model_eval`
    for use with Pyro's MCMC/NUTS.
    """
    _oracle_spiral_known_theta_model_eval(
        target_task_X,
        target_task_y,
        int(x_dim),
        float(base_mean),
        float(base_scale),
        int(transform_seed),
        speed=float(speed),
        random_initial=bool(random_initial),
        initial_scale=float(initial_scale),
    )

# This is the top-level function that Pyro's MCMC will call in the child process.
def oracle_model_for_mcmc(target_task_X, target_task_y, x_dim, w_mean, w_std,
                          prior_family: str = "normal",
                          student_t_df: Optional[float] = None,
                          student_t_df_list: Optional[List[float]] = None):
    """
    A picklable, top-level wrapper around _simple_model_eval for use with MCMC.
    """
    _simple_classification_model_eval(target_task_X, target_task_y, x_dim, w_mean, w_std,
                       prior_family=prior_family, student_t_df=student_t_df,
                       student_t_df_list=student_t_df_list)
    
def oracle_model_for_hierarchical_mcmc(prior_tasks_X, prior_tasks_y, target_task_X, target_task_y,
                                       x_dim, prior_mean_min, prior_mean_max,
                                       prior_family: str = "normal",
                                       student_t_df: Optional[float] = None):
    """
    A picklable, top-level wrapper around _hier_model_eval for use with MCMC.
    """
    _hier_classification_model_eval(prior_tasks_X, prior_tasks_y, target_task_X, target_task_y,
                     x_dim, prior_mean_min, prior_mean_max,
                     prior_family=prior_family, student_t_df=student_t_df)

def oracle_model_for_hierarchical_df_mcmc(
    prior_tasks_X, prior_tasks_y,
    target_task_X, target_task_y,
    x_dim: int,
    tau_eps: float = 1e-3,
    prior_mean_min: float = -8.0,
    prior_mean_max: float = 8.0,
):
    _hier_df_model_eval(
        prior_tasks_X, prior_tasks_y,
        target_task_X, target_task_y,
        x_dim=x_dim,
        tau_eps=tau_eps,
        prior_mean_min=float(prior_mean_min),
        prior_mean_max=float(prior_mean_max),
        prior_family="student_t",
    )


def oracle_model_for_hierarchical_df_mixture_mu_mcmc(
    prior_tasks_X,
    prior_tasks_y,
    target_task_X,
    target_task_y,
    x_dim: int,
    student_t_df_list: List[float],
    prior_mean_min: float = -8.0,
    prior_mean_max: float = 8.0,
):
    _hier_df_mixture_mu_model_eval(
        prior_tasks_X,
        prior_tasks_y,
        target_task_X,
        target_task_y,
        x_dim=int(x_dim),
        student_t_df_list=student_t_df_list,
        prior_mean_min=float(prior_mean_min),
        prior_mean_max=float(prior_mean_max),
    )
    
def oracle_model_for_hierarchical_mcmc_spiral(
    prior_tasks_X, prior_tasks_y,
    target_task_X, target_task_y,
    x_dim, prior_mean_min, prior_mean_max,
    A_scale: float = 1.0,
    speed: float = 1.0,
):
    _hier_spiral_model_eval(
        prior_tasks_X, prior_tasks_y,
        target_task_X, target_task_y,
        int(x_dim), float(prior_mean_min), float(prior_mean_max),
        A_scale=float(A_scale),
        speed=float(speed),
    )
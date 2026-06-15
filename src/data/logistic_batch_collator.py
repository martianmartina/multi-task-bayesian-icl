import torch
import torch.nn as nn
from data.f_theta import init_f_theta, apply_f_theta
from data.logistic_helpers import *
from typing import List, Optional, Tuple, Union, Dict, Any, Literal

class LogisticBatchCollator:
    """
    Collate indices -> generate one batch (X, Y, role) or (X, Y, seg, task, role) on CPU.
    """
    def __init__(
        self,
        *,
        w_bank: Optional[torch.Tensor],  # [N, K+1, D, 1] or None
        K: int,
        sample_num_prior_tasks: bool,
        min_num_prior_tasks: int,
        max_num_prior_tasks: int,
        T: int,
        T_ctx: int,
        x_dim: int,
        x_mean: float,
        x_std: float,
        num_evidence_tokens: int,
        num_thinking_tokens: int,
        add_task_ids: bool,
        X_tpl: torch.Tensor,
        Y_tpl: torch.Tensor,
        seg_tpl: torch.Tensor,
        task_tpl: torch.Tensor,
        role_tpl: torch.Tensor,
        # If w_bank is None, we sample w on the fly using these:
        distribution_names: List[str],
        distribution_probs: Optional[torch.Tensor],
        student_t_df: Optional[float],
        student_t_df_list: Optional[List[float]],
        w_mean: float,
        w_std: float,
        w_mean_min: Optional[float],
        w_mean_max: Optional[float],
        w_std_min: Optional[float],
        w_std_max: Optional[float],
        # Transform handling (applied to w)
        transform_name: str,
        transform_resample: str,  # "dataset" | "batch" | "sample"
        transform_seed: Optional[int],
        transform_chunk_size: int,
        transform_module: Optional[nn.Module] = None,  # only for transform_resample="dataset"
        w_bank_is_transformed: bool = False,
        # Likelihood / link function f_theta applied to s = x @ w
        likelihood: str = "identity",  # identity|spline|mlp
        spline_cfg: Optional[object] = None,
        mlp_cfg: Optional[object] = None,
        theta_mode: Literal["global", "per_batch", "per_sample"] = "global",
        theta_seed: int = 0,
        logit_scale: float = 1.0,
        theta_global: Optional[Dict[str, Any]] = None,
    ):
        self.w_bank = w_bank
        self.K = K
        self.sample_num_prior_tasks = bool(sample_num_prior_tasks)
        self.min_num_prior_tasks = int(min_num_prior_tasks)
        self.max_num_prior_tasks = int(max_num_prior_tasks)
        self.T = T
        self.T_ctx = T_ctx
        self.x_dim = x_dim
        self.x_mean = float(x_mean)
        self.x_std = float(x_std)
        self.num_evidence_tokens = int(num_evidence_tokens)
        self.num_thinking_tokens = int(num_thinking_tokens)
        self.add_task_ids = bool(add_task_ids)

        self.X_tpl = X_tpl
        self.Y_tpl = Y_tpl
        self.seg_tpl = seg_tpl
        self.task_tpl = task_tpl
        self.role_tpl = role_tpl
        
        self._template_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {
            int(K): (self.X_tpl, self.Y_tpl, self.seg_tpl, self.task_tpl, self.role_tpl)
        }

        self.distribution_names = distribution_names
        self.distribution_probs = distribution_probs
        self.student_t_df = student_t_df
        self.student_t_df_list = student_t_df_list
        self.w_mean = float(w_mean)
        self.w_std = float(w_std)
        self.w_mean_min = w_mean_min
        self.w_mean_max = w_mean_max
        self.w_std_min = w_std_min
        self.w_std_max = w_std_max

        self.transform_name = str(transform_name)
        self.transform_resample = str(transform_resample).lower()
        self.transform_seed = transform_seed
        self.transform_chunk_size = int(transform_chunk_size)
        self.transform_module = transform_module
        self.w_bank_is_transformed = bool(w_bank_is_transformed)

        self._call_idx = 0

        self.likelihood = str(likelihood).lower()
        self.spline_cfg = spline_cfg
        self.mlp_cfg = mlp_cfg
        self.theta_mode = theta_mode
        self.theta_seed = int(theta_seed)
        self.logit_scale = float(logit_scale)
        self.theta_global = theta_global

        allowed = {"dataset", "batch", "sample"}
        if self.transform_resample not in allowed:
            raise ValueError(f"transform_resample must be one of {sorted(allowed)}, got '{transform_resample}'")
        if self.transform_resample == "dataset" and self.transform_module is None and self.transform_name.lower() != "identity":
            raise ValueError("transform_module must be provided when transform_resample='dataset' and transform_name != 'identity'")
        if self.transform_resample != "dataset" and self.transform_module is not None:
            raise ValueError("transform_module should be None unless transform_resample='dataset'")

        if self.theta_mode not in ("global", "per_batch", "per_sample"):
            raise ValueError(f"theta_mode must be one of ['global','per_batch','per_sample'], got '{self.theta_mode}'")
        if self.likelihood != "identity" and self.theta_mode == "global" and self.theta_global is None:
            raise ValueError("theta_global must be provided when likelihood != 'identity' and theta_mode == 'global'")
        if not (0 <= self.min_num_prior_tasks <= self.max_num_prior_tasks <= self.K):
            raise ValueError(
                f"Invalid K sampling range [{self.min_num_prior_tasks}, {self.max_num_prior_tasks}] for K={self.K}"
            )

    def _sample_k_for_batch(self, B: int) -> torch.Tensor:
        if not self.sample_num_prior_tasks:
            return torch.full((B,), self.K, dtype=torch.long)
        return torch.randint(
            low=self.min_num_prior_tasks,
            high=self.max_num_prior_tasks + 1,
            size=(B,),
            dtype=torch.long,
        )

    def _get_template_for_k(
        self,
        K: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if K not in self._template_cache:
            self._template_cache[K] = _build_templates(
                K=K,
                T=self.T,
                T_ctx=self.T_ctx,
                x_dim=self.x_dim,
                num_evidence_tokens=self.num_evidence_tokens,
                num_thinking_tokens=self.num_thinking_tokens,
                add_task_ids=self.add_task_ids,
            )
        return self._template_cache[K]

    def _apply_link(self, s: torch.Tensor, *, idx: torch.Tensor, call_idx: int) -> torch.Tensor:
        """
        Apply optional f_theta to s and then apply logit_scale.
        s: [B, ...] tensor
        idx: [B] episode indices (used for per-sample theta seeding)
        """
        if self.likelihood == "identity":
            return s * self.logit_scale

        if self.theta_mode == "global":
            out = apply_f_theta(s, likelihood=self.likelihood, theta=self.theta_global)
            return out * self.logit_scale

        if self.theta_mode == "per_batch":
            g = torch.Generator(device="cpu").manual_seed(self.theta_seed + int(call_idx))
            theta = init_f_theta(
                likelihood=self.likelihood,
                generator=g,
                spline_cfg=self.spline_cfg,
                mlp_cfg=self.mlp_cfg,
            )
            out = apply_f_theta(s, likelihood=self.likelihood, theta=theta)
            return out * self.logit_scale

        # per_sample: theta differs per episode; loop over batch
        outs = []
        for b in range(int(s.shape[0])):
            seed = self.theta_seed + int(idx[b].item())
            g = torch.Generator(device="cpu").manual_seed(seed)
            theta = init_f_theta(
                likelihood=self.likelihood,
                generator=g,
                spline_cfg=self.spline_cfg,
                mlp_cfg=self.mlp_cfg,
            )
            outs.append(apply_f_theta(s[b], likelihood=self.likelihood, theta=theta))
        out = torch.stack(outs, dim=0)
        return out * self.logit_scale

    @staticmethod
    def _sample_student_t(df: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Vectorized standard Student-t sampler with potentially different df per batch item.

        df:   [B] positive tensor
        shape: sample shape for each batch item, e.g. (K+1, D, 1)

        Returns: [B, *shape]
        """
        if df.ndim != 1:
            raise ValueError(f"df must be 1D [B], got shape {tuple(df.shape)}")
        if torch.any(df <= 0):
            raise ValueError("All df must be > 0 for Student-t sampling.")

        dist = torch.distributions.StudentT(df=df, loc=0.0, scale=1.0)
        t = dist.sample(shape) # [*shape, B]
        t = t.permute(-1, *range(t.ndim - 1)) # [B, *shape]
        return t
    
    def _sample_w_for_batch(self, B: int) -> torch.Tensor:
        """
        Sample w ~ mixture over families, with per-sequence mu (scalar shared across dims),
        returning [B, K+1, D, 1].
        """
        K = self.K
        D = self.x_dim
        names = self.distribution_names
        num_families = len(names)

        # Per-sequence mu
        if self.w_mean_min is not None and self.w_mean_max is not None:
            mu = torch.empty(B, 1).uniform_(self.w_mean_min, self.w_mean_max)
        else:
            mu = torch.full((B, 1), self.w_mean)

        # Per-sequence std
        if self.w_std_min is not None and self.w_std_max is not None:
            std = torch.empty(B, 1).uniform_(self.w_std_min, self.w_std_max)
        else:
            std = torch.full((B, 1), self.w_std)

        # Family choice
        if self.distribution_probs is not None:
            family_idx = torch.multinomial(self.distribution_probs, num_samples=B, replacement=True)
        else:
            family_idx = torch.randint(low=0, high=num_families, size=(B,))

        w = torch.empty(B, K + 1, D, 1, dtype=torch.float32)

        for j, name in enumerate(names):
            mask = (family_idx == j)
            n_j = int(mask.sum().item())
            if n_j == 0:
                continue
            idx_j = mask.nonzero(as_tuple=True)[0]
            mu_j = mu[idx_j]  # [n_j, 1]
            std_j = std[idx_j]  # [n_j, 1]
            if name == "normal":
                base = torch.randn(n_j, K + 1, D, 1)
            elif name == "laplace":
                lap = torch.distributions.Laplace(loc=0.0, scale=1.0)
                base = lap.sample((n_j, K + 1, D, 1))
            elif name == "student_t":
                # Choose df per sequence from a discrete grid (preferred), or fall back to a single df.
                if self.student_t_df_list is not None and len(self.student_t_df_list) > 0:
                    df_choices = torch.tensor(self.student_t_df_list, dtype=torch.float32)
                    df_idx = torch.randint(low=0, high=df_choices.numel(), size=(n_j,))
                    df_sel = df_choices[df_idx]  # [n_j]
                elif self.student_t_df is not None:
                    df_sel = torch.full((n_j,), float(self.student_t_df), dtype=torch.float32)
                else:
                    df_sel = torch.tensor(
                        [random.choice([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 1000.0]) for _ in range(n_j)],
                        dtype=torch.float32,
                    )

                base = torch.empty(n_j, K + 1, D, 1, dtype=torch.float32)
                inf_mask = torch.isinf(df_sel) | (df_sel >= 1e6)
                fin_mask = ~inf_mask
                if int(inf_mask.sum().item()) > 0:
                    base[inf_mask] = torch.randn(int(inf_mask.sum().item()), K + 1, D, 1)
                if int(fin_mask.sum().item()) > 0:
                    base[fin_mask] = self._sample_student_t(df_sel[fin_mask], (K + 1, D, 1))
            else:
                raise ValueError(f"Unsupported w distribution family: {name}")

            std_j_ = std_j.view(n_j, 1, 1, 1)
            mu_j_  = mu_j.view(n_j, 1, 1, 1)
            w[idx_j] = base * std_j_ + mu_j_

        return w

    def _apply_transform_to_w(self, w: torch.Tensor) -> torch.Tensor:
        """
        Apply a (possibly resampled) transform to w.
        w: [B, K+1, D, 1] float32
        """
        if self.transform_name.lower() == "identity":
            return w

        if self.w_bank is not None and self.transform_resample == "dataset" and self.w_bank_is_transformed:
            # precompute == true and transform_resample == dataset
            return w

        B = w.shape[0]
        flat = w.view(-1, self.x_dim)  # [B*(K+1), D]

        # Select transform instance
        if self.transform_resample == "dataset":
            transform = self.transform_module
            assert transform is not None
            with torch.no_grad():
                out = self._transform_in_chunks(transform, flat)
            return out.view(B, -1, self.x_dim, 1)

        # Resample transform params per batch or per sample.
        from torch.utils.data import get_worker_info
        wi = get_worker_info()
        worker_id = 0 if wi is None else int(wi.id)

        base_seed = self.transform_seed
        if base_seed is None:
            # Non-deterministic; rely on process RNG state.
            base_seed = int(torch.seed() % (2**31 - 1))

        if self.transform_resample == "batch":
            seed = int(base_seed + worker_id * 1_000_000 + self._call_idx)
            g = torch.Generator(device="cpu").manual_seed(seed)
            transform = _make_transform(self.transform_name, self.x_dim, generator=g)
            if transform is None:
                return w
            with torch.no_grad():
                out = self._transform_in_chunks(transform, flat)
            return out.view(B, -1, self.x_dim, 1)

        # "sample": different transform per sequence
        outs = []
        per_seq = w.view(B, -1, self.x_dim)  # [B, K+1, D]
        for i in range(B):
            seed = int(base_seed + worker_id * 1_000_000 + self._call_idx * B + i)
            g = torch.Generator(device="cpu").manual_seed(seed)
            transform = _make_transform(self.transform_name, self.x_dim, generator=g)
            if transform is None:
                outs.append(per_seq[i])
                continue
            with torch.no_grad():
                outs.append(transform(per_seq[i]))
        out = torch.stack(outs, dim=0)  # [B, K+1, D]
        return out.view(B, -1, self.x_dim, 1)

    def _transform_in_chunks(self, transform: nn.Module, x2: torch.Tensor) -> torch.Tensor:
        """
        Apply transform to x2: [N, D] in chunks.
        """
        assert x2.ndim == 2 and x2.shape[1] == self.x_dim
        N = x2.size(0)
        chunk = self.transform_chunk_size
        if N <= chunk:
            return transform(x2)
        out = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            out.append(transform(x2[s:e]))
        return torch.cat(out, dim=0)

    def __call__(self, batch: List[Union[int, torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
        # batch is a list of indices (ints) or 0-d tensors
        if isinstance(batch[0], torch.Tensor):
            idx = torch.stack([b.view(1) for b in batch], dim=0).view(-1).long()
        else:
            idx = torch.tensor(batch, dtype=torch.long)
        B = idx.numel()

        # Get w: either from bank or sample fresh
        if self.w_bank is not None:
            w = self.w_bank[idx].to(torch.float32)  # [B, K+1, D, 1]
            if not torch.isfinite(w).all():
                print("w has non-finite!",
                      "inf:", torch.isinf(w).sum().item(),
                      "nan:", torch.isnan(w).sum().item(),
                      "max_abs:", w.abs().max().item())
                raise RuntimeError(f"Non-finite w")
        else:
            w = self._sample_w_for_batch(B)         # [B, K+1, D, 1]

        # Potentially apply flow transform to w (fixed or resampled)
        w = self._apply_transform_to_w(w)
        call_idx = int(self._call_idx)
        self._call_idx += 1

        k_per_seq = self._sample_k_for_batch(B)
        seq_lens = k_per_seq * (self.T + 1) + 1 + self.T_ctx + self.num_thinking_tokens
        L_max = int(seq_lens.max().item())

        X = torch.zeros(B, L_max, self.x_dim, dtype=torch.float32)
        Y = torch.full((B, L_max, 1), Y_PADDING, dtype=torch.float32)
        seg = torch.zeros(B, L_max, dtype=torch.long)
        task = torch.zeros(B, L_max, dtype=torch.long)
        role = torch.full((B, L_max), ROLE_PADDING, dtype=torch.long)

        unique_k = torch.unique(k_per_seq, sorted=True)
        for k_val_t in unique_k:
            k_val = int(k_val_t.item())
            batch_mask = (k_per_seq == k_val)
            idx_group = batch_mask.nonzero(as_tuple=True)[0]
            Bk = int(idx_group.numel())
            if Bk == 0:
                continue

            Lk = int(k_val * (self.T + 1) + 1 + self.T_ctx + self.num_thinking_tokens)
            X_tpl_k, Y_tpl_k, seg_tpl_k, task_tpl_k, role_tpl_k = self._get_template_for_k(k_val)
            Xk = X_tpl_k.unsqueeze(0).repeat(Bk, 1, 1).clone()
            Yk = Y_tpl_k.unsqueeze(0).repeat(Bk, 1, 1).clone()
            segk = seg_tpl_k.unsqueeze(0).repeat(Bk, 1)
            taskk = task_tpl_k.unsqueeze(0).repeat(Bk, 1)
            rolek = role_tpl_k.unsqueeze(0).repeat(Bk, 1)

            # ----- Prior tasks -----
            if k_val > 0:
                x_prior = torch.empty(Bk, k_val, self.T, self.x_dim).normal_(self.x_mean, self.x_std)
                w_prior = w[idx_group, :k_val, :, :]
                logits = torch.matmul(x_prior, w_prior)  # [Bk, k_val, T, 1]
                logits = self._apply_link(logits, idx=idx[idx_group], call_idx=call_idx)

                if not torch.isfinite(logits).all():
                    bad = logits[~torch.isfinite(logits)]
                    print("Non-finite logits!", bad[:10], "min", logits.min().item(), "max", logits.max().item())
                    raise RuntimeError("Non-finite logits in data generation")

                y_prior = torch.bernoulli(torch.sigmoid(logits))
                pos = 0
                for t in range(k_val):
                    pos += 1
                    Xk[:, pos:pos+self.T, :] = x_prior[:, t, :, :]
                    Yk[:, pos:pos+self.T, :] = y_prior[:, t, :, :]
                    pos += self.T
            else:
                pos = 0

            # ----- Target task -----
            pos += 1  # skip TARGET token row

            x_tgt = torch.empty(Bk, self.T_ctx, self.x_dim).normal_(self.x_mean, self.x_std)
            w_tgt = w[idx_group, k_val, :, :]  # target task depends on sampled K
            logits_tgt = torch.matmul(x_tgt, w_tgt)
            logits_tgt = self._apply_link(logits_tgt, idx=idx[idx_group], call_idx=call_idx)
            y_tgt = torch.bernoulli(torch.sigmoid(logits_tgt))

            if self.num_evidence_tokens > 0:
                e = self.num_evidence_tokens
                Xk[:, pos:pos+e, :] = x_tgt[:, :e, :]
                Yk[:, pos:pos+e, :] = y_tgt[:, :e, :]
                pos += e

            if self.num_thinking_tokens > 0:
                pos += self.num_thinking_tokens

            num_pred = self.T_ctx - self.num_evidence_tokens
            if num_pred > 0:
                Xk[:, pos:pos+num_pred, :] = x_tgt[:, self.num_evidence_tokens:, :]
                Yk[:, pos:pos+num_pred, :] = y_tgt[:, self.num_evidence_tokens:, :]

            X[idx_group, :Lk, :] = Xk
            Y[idx_group, :Lk, :] = Yk
            seg[idx_group, :Lk] = segk
            task[idx_group, :Lk] = taskk
            role[idx_group, :Lk] = rolek

        if self.add_task_ids:
            return X, Y, seg, task, role
        else:
            return X, Y, role

from typing import Tuple, Optional, List, Union, Callable, Literal, Any, Dict

import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
from data.f_theta import init_f_theta, apply_f_theta
from data.logistic_helpers import *
from data.logistic_batch_collator import LogisticBatchCollator

class MultiTaskSyntheticLogisticDataset(Dataset):

    def __init__(
        self,
        num_samples: int,
        num_prior_tasks: int,
        sequence_length: int,
        sample_num_prior_tasks: bool = False,
        min_num_prior_tasks: int = 0,
        max_num_prior_tasks: Optional[int] = None,
        context_length: int = None,
        num_evidence_tokens: int = 0,
        num_thinking_tokens: int = 0,
        x_dim: int = 1,
        distribution_names: List[str] = ["normal"],
        distribution_probs: Optional[List[float]] = None,
        transform_name: str = "identity",
        student_t_df: Optional[float] = None,
        student_t_df_list: Optional[List[float]] = None,
        w_mean: float = 0.0,
        w_std: float = 1.0,
        w_mean_min: Optional[float] = None,
        w_mean_max: Optional[float] = None,
        w_std_min: Optional[float] = None,
        w_std_max: Optional[float] = None,
        x_mean: float = 0.0,
        x_std: float = 1.0,
        add_task_ids: bool = False,
        precompute: bool = True,              # now means "precompute w only"
        w_bank_dtype: torch.dtype = torch.bfloat16,  # prevent very heavy tail overflow
        transform_seed: Optional[int] = 122,
        transform_resample: str = "dataset",  # "dataset" (old behavior) | "batch" | "sample"
        transform_chunk_size: int = 8192,

        # ---- optional harder likelihood f_theta applied to s = x@w ----
        likelihood: str = "identity",  # identity|spline|mlp
        spline_cfg: Optional[object] = None,
        mlp_cfg: Optional[object] = None,
        theta_mode: Literal["global", "per_batch", "per_sample"] = "global",
        theta_seed: int = 0,
        logit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert x_dim >= 1, "x_dim must be >= 1"
        self.num_samples = int(num_samples)
        self.num_prior_tasks = int(num_prior_tasks)
        self.sample_num_prior_tasks = bool(sample_num_prior_tasks)
        self.min_num_prior_tasks = int(min_num_prior_tasks)
        self.max_num_prior_tasks = int(max_num_prior_tasks) if max_num_prior_tasks is not None else self.num_prior_tasks
        self.sequence_length = int(sequence_length)
        self.context_length = int(context_length) if context_length is not None else int(sequence_length)
        self.num_evidence_tokens = int(num_evidence_tokens)
        self.num_thinking_tokens = int(num_thinking_tokens)
        assert self.num_evidence_tokens < self.context_length, "Evidence points must be less than total context length"
        if not (0 <= self.min_num_prior_tasks <= self.max_num_prior_tasks <= self.num_prior_tasks):
            raise ValueError(
                f"Invalid K sampling range [{self.min_num_prior_tasks}, {self.max_num_prior_tasks}] "
                f"for num_prior_tasks={self.num_prior_tasks}"
            )

        self.x_dim = int(x_dim)
        self.x_mean = float(x_mean)
        self.x_std = float(x_std)

        self.w_mean = float(w_mean)
        self.w_std = float(w_std)
        self.w_mean_min = w_mean_min
        self.w_mean_max = w_mean_max

        self.w_std_min = w_std_min
        self.w_std_max = w_std_max
        self.distribution_names = distribution_names
        self.distribution_probs = self._normalize_distribution_probs(distribution_probs)

        self.student_t_df = student_t_df
        self.student_t_df_list = student_t_df_list

        self.add_task_ids = bool(add_task_ids)
        self.precompute = bool(precompute)

        self.transform_name = transform_name
        self.transform_seed = transform_seed
        self.transform_resample = str(transform_resample).lower()
        self.transform_chunk_size = int(transform_chunk_size)
        self.transform = None  # may be None even when transform_name != identity if we resample later

        self.likelihood = str(likelihood).lower()
        self.spline_cfg = spline_cfg
        self.mlp_cfg = mlp_cfg
        self.theta_mode = theta_mode
        self.theta_seed = int(theta_seed)
        self.logit_scale = float(logit_scale)
        self._theta_global: Optional[Dict[str, Any]] = None

        if self.likelihood != "identity" and self.theta_mode == "global":
            g = torch.Generator(device="cpu").manual_seed(self.theta_seed)
            self._theta_global = init_f_theta(
                likelihood=self.likelihood,
                generator=g,
                spline_cfg=self.spline_cfg,
                mlp_cfg=self.mlp_cfg,
            )

        if self.transform_resample == "dataset":
            self.transform = self._create_transform()
        else:
            # In resampling modes we create transforms inside collate_fn.
            self.transform = None

        # Templates
        self._X_tpl, self._Y_tpl, self._seg_tpl, self._task_tpl, self._role_tpl = build_templates(
            K=self.num_prior_tasks,
            T=self.sequence_length,
            T_ctx=self.context_length,
            x_dim=self.x_dim,
            num_evidence_tokens=self.num_evidence_tokens,
            num_thinking_tokens=self.num_thinking_tokens,
            add_task_ids=self.add_task_ids,
        )

        # Optional: precompute w_bank only
        self.w_bank: Optional[torch.Tensor] = None
        w_bank_is_transformed = False
        if self.precompute:
            print("Precomputing w_bank...")
            w = self._sample_w_bank(num_samples=self.num_samples)          # [N, K+1, D, 1]
            # Only apply transform here if transform is fixed for the whole dataset.
            if self.transform is not None and self.transform_resample == "dataset":
                w = self._apply_transform_for_w_dataset(w)
                w_bank_is_transformed = True
            self.w_bank = w.to(dtype=w_bank_dtype).contiguous()
        else:
            print("Not precomputing w_bank...")

        self.collate_fn: Callable = LogisticBatchCollator(
            w_bank=self.w_bank,
            K=self.num_prior_tasks,
            sample_num_prior_tasks=self.sample_num_prior_tasks,
            min_num_prior_tasks=self.min_num_prior_tasks,
            max_num_prior_tasks=self.max_num_prior_tasks,
            T=self.sequence_length,
            T_ctx=self.context_length,
            x_dim=self.x_dim,
            x_mean=self.x_mean,
            x_std=self.x_std,
            num_evidence_tokens=self.num_evidence_tokens,
            num_thinking_tokens=self.num_thinking_tokens,
            add_task_ids=self.add_task_ids,
            X_tpl=self._X_tpl,
            Y_tpl=self._Y_tpl,
            seg_tpl=self._seg_tpl,
            task_tpl=self._task_tpl,
            role_tpl=self._role_tpl,
            distribution_names=self.distribution_names,
            distribution_probs=self.distribution_probs,
            student_t_df=self.student_t_df,
            student_t_df_list=self.student_t_df_list,
            w_mean=self.w_mean,
            w_std=self.w_std,
            w_mean_min=self.w_mean_min,
            w_mean_max=self.w_mean_max,
            w_std_min=self.w_std_min,
            w_std_max=self.w_std_max,
            transform_name=self.transform_name,
            transform_resample=self.transform_resample,
            transform_seed=self.transform_seed,
            transform_chunk_size=self.transform_chunk_size,
            transform_module=self.transform,
            w_bank_is_transformed=w_bank_is_transformed,
            likelihood=self.likelihood,
            spline_cfg=self.spline_cfg,
            mlp_cfg=self.mlp_cfg,
            theta_mode=self.theta_mode,
            theta_seed=self.theta_seed,
            logit_scale=self.logit_scale,
            theta_global=self._theta_global,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # return only index; collate_fn builds full tensors for the batch
        return idx

    def _normalize_distribution_probs(self, distribution_probs: Optional[List[float]]) -> Optional[torch.Tensor]:
        if distribution_probs is None:
            return None
        if len(distribution_probs) != len(self.distribution_names):
            raise ValueError("distribution_probs must have the same length as distribution_names.")
        probs = torch.tensor(distribution_probs, dtype=torch.float64)
        if torch.any(probs < 0):
            raise ValueError("distribution_probs must be non-negative.")
        total = probs.sum()
        if total <= 0:
            raise ValueError("distribution_probs must sum to a positive value.")
        return (probs / total).to(torch.float32)

    def _create_transform(self) -> Optional[nn.Module]:
        generator = None
        if self.transform_seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.transform_seed)
        return make_transform(self.transform_name, self.x_dim, generator=generator)

    def _sample_w_bank(self, num_samples: int) -> torch.Tensor:
        """
        Pre-sample w for all sequences: [N, K+1, D, 1]
        """
        K = self.num_prior_tasks
        D = self.x_dim
        names = self.distribution_names
        num_families = len(names)

        # Per-sequence mu
        if self.w_mean_min is not None and self.w_mean_max is not None:
            mu = torch.empty(num_samples, 1).uniform_(self.w_mean_min, self.w_mean_max)
        else:
            mu = torch.full((num_samples, 1), self.w_mean)

        # Per-sequence std  
        if self.w_std_min is not None and self.w_std_max is not None: 
            std = torch.empty(num_samples, 1).uniform_(self.w_std_min, self.w_std_max) 
        else:  
            std = torch.full((num_samples, 1), self.w_std)  


        # Family selection
        if self.distribution_probs is not None:
            family_idx = torch.multinomial(self.distribution_probs, num_samples=num_samples, replacement=True)
        else:
            family_idx = torch.randint(low=0, high=num_families, size=(num_samples,))

        w = torch.empty(num_samples, K + 1, D, 1, dtype=torch.float32)

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
                laplace = torch.distributions.Laplace(loc=0.0, scale=1.0)
                base = laplace.sample((n_j, K + 1, D, 1))
            elif name == "student_t":
                # Choose df per sequence from a discrete grid (preferred), or fall back to a single df.
                if self.student_t_df_list is not None and len(self.student_t_df_list) > 0:
                    df_choices = torch.tensor(self.student_t_df_list, dtype=torch.float32)
                    df_idx = torch.randint(low=0, high=df_choices.numel(), size=(n_j,))
                    df_sel = df_choices[df_idx]  # [n_j]
                elif self.student_t_df is not None:
                    df_sel = torch.full((n_j,), float(self.student_t_df), dtype=torch.float32)
                else:
                    raise ValueError("student_t_df_list or student_t_df must be provided.")

                base = torch.empty(n_j, K + 1, D, 1, dtype=torch.float32)
                inf_mask = torch.isinf(df_sel) | (df_sel >= 1e6)
                fin_mask = ~inf_mask
                if int(inf_mask.sum().item()) > 0:
                    base[inf_mask] = torch.randn(int(inf_mask.sum().item()), K + 1, D, 1)
                if int(fin_mask.sum().item()) > 0:
                    base[fin_mask] = _LogisticBatchCollator._sample_student_t(df_sel[fin_mask], (K + 1, D, 1))
            else:
                raise ValueError(f"Unsupported w distribution family: {name}")


            std_j_ = std_j.view(n_j, 1, 1, 1)
            mu_j_  = mu_j.view(n_j, 1, 1, 1)
            w[idx_j] = base * std_j_ + mu_j_

        return w

    def _apply_transform_for_w_dataset(self, w: torch.Tensor) -> torch.Tensor:
        """
        Apply transform to w in chunks to avoid OOM.
        Input:  [N, K+1, D, 1]
        Output: [N, K+1, D, 1]
        """
        assert w.ndim == 4 and w.shape[-2] == self.x_dim and w.shape[-1] == 1
        N = w.shape[0]
        w2 = w.view(-1, self.x_dim)  # [N*(K+1), D]

        with torch.no_grad():
            B = w2.size(0)
            chunk = self.transform_chunk_size
            if B > chunk:
                out = []
                for s in range(0, B, chunk):
                    e = min(s + chunk, B)
                    out.append(self.transform(w2[s:e]))
                w2 = torch.cat(out, dim=0)
            else:
                w2 = self.transform(w2)

        return w2.view(N, -1, self.x_dim, 1)


# ------------------------------------------------------------
# MultiTaskLogisticDataModule
# ------------------------------------------------------------


from typing import Optional, List
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch


class MultiTaskLogisticDataModule(pl.LightningDataModule):
    """
    Lightning DataModule wrapper for MultiTaskSyntheticLogisticDataset.
    """

    def __init__(
        self,
        num_prior_tasks: int,
        sequence_length: int,
        x_dim: int,
        num_train_samples: int,
        num_val_samples: int,
        sample_num_prior_tasks: bool = False,
        min_num_prior_tasks: int = 0,
        max_num_prior_tasks: Optional[int] = None,
        context_length: int = None,
        num_evidence_tokens: int = 0,
        num_thinking_tokens: int = 0,
        batch_size: int = 32,
        num_workers: int = 4,
        distribution_names: List[str] = ["normal"],
        distribution_probs: Optional[List[float]] = None,
        transform_name: str = "identity",
        student_t_df: Optional[float] = None,
        student_t_df_list: Optional[List[float]] = None,
        w_mean: float = 0.0,
        w_std: float = 1.0,
        w_mean_min: Optional[float] = None,
        w_mean_max: Optional[float] = None,
        w_std_min: Optional[float] = None,
        w_std_max: Optional[float] = None,
        x_mean: float = 0.0,
        x_std: float = 1.0,
        add_task_ids: bool = False,
        transform_seed: Optional[int] = 122,
        # Optional split-specific transform seeds (override transform_seed)
        train_transform_seed: Optional[int] = None,
        val_transform_seed: Optional[int] = None,

        precompute: bool = True,                  # precompute w_bank only
        w_bank_dtype: str = "bfloat16",            
        transform_resample: str = "dataset",       # "dataset" (old) | "batch" | "sample"
        transform_chunk_size: int = 8192,

        # ---- optional harder likelihood f_theta applied to s = x@w ----
        likelihood: str = "identity",  # identity|spline|mlp
        spline_cfg: Optional[object] = None,
        mlp_cfg: Optional[object] = None,
        theta_mode: str = "global",  # global|per_batch|per_sample
        theta_seed: int = 0,
        logit_scale: float = 1.0,

        # ---- loader knobs ----
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["spline_cfg", "mlp_cfg"])
        self._spline_cfg = spline_cfg
        self._mlp_cfg = mlp_cfg

        self.train_dataset: Optional[MultiTaskSyntheticLogisticDataset] = None
        self.val_dataset: Optional[MultiTaskSyntheticLogisticDataset] = None

        # Resolve per-split seeds (defaults to transform_seed for backward compatibility)
        self._train_transform_seed = int(train_transform_seed) if train_transform_seed is not None else self.hparams.transform_seed
        self._val_transform_seed = int(val_transform_seed) if val_transform_seed is not None else self.hparams.transform_seed

        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        if isinstance(self.hparams.w_bank_dtype, str):
            key = self.hparams.w_bank_dtype.lower()
            if key not in dtype_map:
                raise ValueError(f"Unsupported w_bank_dtype='{self.hparams.w_bank_dtype}'. Options: {sorted(dtype_map)}")
            self._w_bank_dtype = dtype_map[key]
        else:
            self._w_bank_dtype = self.hparams.w_bank_dtype

    def setup(self, stage: Optional[str] = None) -> None:
        common_args = dict(
            num_prior_tasks=self.hparams.num_prior_tasks,
            sample_num_prior_tasks=self.hparams.sample_num_prior_tasks,
            min_num_prior_tasks=self.hparams.min_num_prior_tasks,
            max_num_prior_tasks=self.hparams.max_num_prior_tasks,
            sequence_length=self.hparams.sequence_length,
            context_length=self.hparams.context_length,
            num_evidence_tokens=self.hparams.num_evidence_tokens,
            num_thinking_tokens=self.hparams.num_thinking_tokens,
            x_dim=self.hparams.x_dim,
            distribution_names=self.hparams.distribution_names,
            distribution_probs=self.hparams.distribution_probs,
            transform_name=self.hparams.transform_name,
            student_t_df=self.hparams.student_t_df,
            student_t_df_list=self.hparams.student_t_df_list,
            w_mean=self.hparams.w_mean,
            w_std=self.hparams.w_std,
            w_mean_min=self.hparams.w_mean_min,
            w_mean_max=self.hparams.w_mean_max,
            w_std_min=self.hparams.w_std_min,
            w_std_max=self.hparams.w_std_max,
            x_mean=self.hparams.x_mean,
            x_std=self.hparams.x_std,
            add_task_ids=self.hparams.add_task_ids,
            transform_resample=self.hparams.transform_resample,

            # new dataset knobs
            precompute=self.hparams.precompute,
            w_bank_dtype=self._w_bank_dtype,
            transform_chunk_size=self.hparams.transform_chunk_size,

            likelihood=self.hparams.likelihood,
            spline_cfg=self._spline_cfg,
            mlp_cfg=self._mlp_cfg,
            theta_mode=self.hparams.theta_mode,
            theta_seed=self.hparams.theta_seed,
            logit_scale=self.hparams.logit_scale,
        )

        if stage in (None, "fit"):
            self.train_dataset = MultiTaskSyntheticLogisticDataset(
                num_samples=self.hparams.num_train_samples,
                transform_seed=self._train_transform_seed,
                **common_args,
            )
            self.val_dataset = MultiTaskSyntheticLogisticDataset(
                num_samples=self.hparams.num_val_samples,
                transform_seed=self._val_transform_seed,
                **common_args,
            )

    def _make_loader(self, ds, shuffle: bool) -> DataLoader:
        assert ds is not None
        persistent = bool(self.hparams.persistent_workers) and self.hparams.num_workers > 0

        kwargs = dict(
            dataset=ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=bool(self.hparams.pin_memory),
            persistent_workers=persistent,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            drop_last=(bool(self.hparams.drop_last) if shuffle else False),
            collate_fn=ds.collate_fn,  
        )
        if self.hparams.num_workers == 0:
            kwargs.pop("prefetch_factor", None)

        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup('fit') before requesting train_dataloader()."
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call setup('fit') before requesting val_dataloader()."
        return self._make_loader(self.val_dataset, shuffle=False)
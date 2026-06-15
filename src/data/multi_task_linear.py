from typing import Tuple, Optional, Union, Literal, Any, Dict

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from src.data.f_theta import init_f_theta, apply_f_theta
PRIOR_TOKEN_X = -10.0
TARGET_TOKEN_X = -11.0

ROLE_PRIOR = 0
ROLE_EVIDENCE = 1
ROLE_THINKING = 2
ROLE_PREDICTION = 3

SEG_PRIOR = 0
SEG_TARGET = 1


class MultiTaskSyntheticLinearDataset(Dataset):

    def __init__(
        self,
        num_samples: int,
        num_prior_tasks: int,
        sequence_length: int,
        x_dim: int = 1,
        noise_mean: float = 0.0,
        noise_std: float = 0.5,
        w_mean: float = 0.0,
        w_std: float = 1.0,
        w_mean_min: Optional[float] = None,
        w_mean_max: Optional[float] = None,
        x_mean: float = 0.0,
        x_std: float = 1.0,
        add_task_ids: bool = False,
        likelihood: str = "identity",  
        spline_cfg: Optional[object] = None,
        mlp_cfg: Optional[object] = None,
        theta_mode: Literal["global", "per_sample"] = "global",
        theta_seed: int = 0,
    ) -> None:
        super().__init__()
        assert x_dim >= 1, "x_dim must be >= 1"
        self.num_samples = num_samples
        self.num_prior_tasks = num_prior_tasks
        self.sequence_length = sequence_length
        self.x_dim = x_dim

        self.noise_mean = float(noise_mean)
        self.noise_std = float(noise_std)
        self.w_mean = float(w_mean)
        self.w_std = float(w_std)
        self.w_mean_min = w_mean_min
        self.w_mean_max = w_mean_max

        self.x_mean = float(x_mean)
        self.x_std = float(x_std)

        self.add_task_ids = bool(add_task_ids)

        self.likelihood = str(likelihood).lower()
        self.spline_cfg = spline_cfg
        self.mlp_cfg = mlp_cfg
        self.theta_mode = theta_mode
        self.theta_seed = int(theta_seed)
        self._theta_global: Optional[Dict[str, Any]] = None

        if self.theta_mode not in ("global", "per_sample"):
            raise ValueError(f"theta_mode must be one of ['global','per_sample'], got '{self.theta_mode}'")

        if self.likelihood != "identity" and self.theta_mode == "global":
            g = torch.Generator(device="cpu").manual_seed(self.theta_seed)
            self._theta_global = init_f_theta(
                likelihood=self.likelihood,
                generator=g,
                spline_cfg=self.spline_cfg,
                mlp_cfg=self.mlp_cfg,
            )

    def __len__(self) -> int:
        return self.num_samples

    def _sample_w(self, mu_scalar: float) -> torch.Tensor:
        """
        Sample a coefficient vector w ~ N(mu_scalar * 1, w_std^2 I) with shape (x_dim, 1).
        """
        return torch.randn(self.x_dim, 1) * self.w_std + mu_scalar

    def _sample_task(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample one linear task with fixed w: draw x ~ N(x_mean, x_std^2 I)
        If likelihood == "identity": y = x @ w + noise.
        Else: y = f_theta(x @ w) + noise.
        Returns x: (T, x_dim), y: (T, 1).
        """
        x = torch.empty(self.sequence_length, self.x_dim).normal_(self.x_mean, self.x_std)
        s = x @ w  # [T, 1]
        if self.likelihood == "identity":
            y = s
        else:
            if self.theta_mode == "global":
                assert self._theta_global is not None
                y = apply_f_theta(s, likelihood=self.likelihood, theta=self._theta_global)
            else:
                # in per_sample mode, theta is initialized inside __getitem__
                assert self._theta_global is not None
                y = apply_f_theta(s, likelihood=self.likelihood, theta=self._theta_global)
        if self.noise_std > 0.0:
            y = y + torch.randn_like(y) * self.noise_std + self.noise_mean
        return x, y

    def __getitem__(self, idx: int):
        # Sample theta per episode if requested
        if self.likelihood != "identity" and self.theta_mode == "per_sample":
            g = torch.Generator(device="cpu").manual_seed(self.theta_seed + int(idx))
            self._theta_global = init_f_theta(
                likelihood=self.likelihood,
                generator=g,
                spline_cfg=self.spline_cfg,
                mlp_cfg=self.mlp_cfg,
            )

        # Choose the shared prior mean for this *sample*.
        if self.w_mean_min is not None and self.w_mean_max is not None:
            mu = torch.empty(1).uniform_(self.w_mean_min, self.w_mean_max).item()
        else:
            mu = self.w_mean

        x_chunks, y_chunks = [], []
        role_chunks = []

        seg_chunks = []
        task_chunks = []

        # Prior tasks
        for t_idx in range(self.num_prior_tasks):
            w = self._sample_w(mu)
            x_task, y_task = self._sample_task(w)

            # PRIOR token row
            x_chunks.append(torch.full((1, self.x_dim), PRIOR_TOKEN_X))
            y_chunks.append(torch.zeros(1, 1))
            role_chunks.append(torch.full((1,), ROLE_PRIOR, dtype=torch.long))

            if self.add_task_ids:
                seg_chunks.append(torch.full((1,), SEG_PRIOR, dtype=torch.long))
                task_chunks.append(torch.full((1,), t_idx, dtype=torch.long))

            # Prior data rows
            x_chunks.append(x_task)
            y_chunks.append(y_task)
            role_chunks.append(torch.full((self.sequence_length,), ROLE_PRIOR, dtype=torch.long))

            if self.add_task_ids:
                seg_chunks.append(torch.full((self.sequence_length,), SEG_PRIOR, dtype=torch.long))
                task_chunks.append(torch.full((self.sequence_length,), t_idx, dtype=torch.long))

        # Target task
        w_tgt = self._sample_w(mu)
        x_tgt, y_tgt = self._sample_task(w_tgt)

        # TARGET token row
        x_chunks.append(torch.full((1, self.x_dim), TARGET_TOKEN_X))
        y_chunks.append(torch.zeros(1, 1))
        role_chunks.append(torch.full((1,), ROLE_EVIDENCE, dtype=torch.long))

        if self.add_task_ids:
            seg_chunks.append(torch.full((1,), SEG_TARGET, dtype=torch.long))
            task_chunks.append(torch.full((1,), self.num_prior_tasks, dtype=torch.long))

        # Target data rows: these are the only positions we train on
        x_chunks.append(x_tgt)
        y_chunks.append(y_tgt)
        role_chunks.append(torch.full((self.sequence_length,), ROLE_PREDICTION, dtype=torch.long))

        if self.add_task_ids:
            seg_chunks.append(torch.full((self.sequence_length,), SEG_TARGET, dtype=torch.long))
            task_chunks.append(torch.full((self.sequence_length,), self.num_prior_tasks, dtype=torch.long))

        x_out = torch.cat(x_chunks, dim=0)   # [L, x_dim]
        y_out = torch.cat(y_chunks, dim=0)   # [L, 1]
        role_ids = torch.cat(role_chunks, dim=0)  # [L]

        if self.add_task_ids:
            seg_ids = torch.cat(seg_chunks, dim=0)    # [L]
            task_ids = torch.cat(task_chunks, dim=0)  # [L]
            return x_out, y_out, seg_ids, task_ids, role_ids
        else:
            return x_out, y_out, role_ids


class MultiTaskLinearDataModule(pl.LightningDataModule):
    """
    Lightning DataModule wrapper for MultiTaskSyntheticLinearDataset.
    """

    def __init__(
        self,
        num_prior_tasks: int,
        sequence_length: int,
        x_dim: int,
        num_train_samples: int,
        num_val_samples: int,
        batch_size: int = 32,
        num_workers: int = 4,
        noise_mean: float = 0.0,
        noise_std: float = 0.5,
        w_mean: float = 0.0,
        w_std: float = 1.0,
        w_mean_min: Optional[float] = None,
        w_mean_max: Optional[float] = None,
        x_mean: float = 0.0,
        x_std: float = 1.0,
        add_task_ids: bool = False,

        likelihood: str = "identity",  # identity|spline|mlp
        spline_cfg: Optional[object] = None,
        mlp_cfg: Optional[object] = None,
        theta_mode: str = "global",  # global|per_sample
        theta_seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["spline_cfg", "mlp_cfg"])
        self._spline_cfg = spline_cfg
        self._mlp_cfg = mlp_cfg

        self.train_dataset: Optional[MultiTaskSyntheticLinearDataset] = None
        self.val_dataset: Optional[MultiTaskSyntheticLinearDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        common_args = dict(
            num_prior_tasks=self.hparams.num_prior_tasks,
            sequence_length=self.hparams.sequence_length,
            x_dim=self.hparams.x_dim,
            noise_mean=self.hparams.noise_mean,
            noise_std=self.hparams.noise_std,
            w_mean=self.hparams.w_mean,
            w_std=self.hparams.w_std,
            w_mean_min=self.hparams.w_mean_min,
            w_mean_max=self.hparams.w_mean_max,
            x_mean=self.hparams.x_mean,
            x_std=self.hparams.x_std,
            add_task_ids=self.hparams.add_task_ids,  

            likelihood=self.hparams.likelihood,
            spline_cfg=self._spline_cfg,
            mlp_cfg=self._mlp_cfg,
            theta_mode=self.hparams.theta_mode,
            theta_seed=self.hparams.theta_seed,
        )
        if stage in (None, "fit"):
            self.train_dataset = MultiTaskSyntheticLinearDataset(
                num_samples=self.hparams.num_train_samples, **common_args
            )
            self.val_dataset = MultiTaskSyntheticLinearDataset(
                num_samples=self.hparams.num_val_samples, **common_args
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Call setup('fit') before requesting train_dataloader()."
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=(self.hparams.num_workers > 0),
            prefetch_factor=4 if self.hparams.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Call setup('fit') before requesting val_dataloader()."
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=(self.hparams.num_workers > 0),
            prefetch_factor=4 if self.hparams.num_workers > 0 else None,
        )

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.gpt2 import GPTConfig, Block as GPTBlock, LayerNorm as GPTLayerNorm
from src.data.multi_task_era5 import (
    ROLE_PRIOR,
    ROLE_EVIDENCE,
    ROLE_PREDICTION,
)


Y_PADDING = -1


class SinusoidalPositionalEncoding(nn.Module):
    """
    Absolute positional encoding, used only if rope=False.
    Expects x with shape [B, T, D].
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return self.dropout(x + self.pe[:T].unsqueeze(0))


class MultiTaskERA5ImplicitLearner(pl.LightningModule):
    """
    Minimal masked-loss decoder model for multi-task ERA5 Bayesian ICL.
    """

    def __init__(
        self,
        x_dim: int = 4,
        y_dim: int = 1,
        d_model: int = 128,
        n_layer: int = 6,
        n_head: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 500,
        max_steps: int = 50_000,
        rope: bool = True,
        use_seg_ids: bool = True,
        use_task_ids: bool = True,
        task_id_scale: float = 1.0,
        seg_id_scale: float = 1.0,
        min_logvar: float = -8.0,
        max_logvar: float = 8.0,
        input_noise: float = 0.0, # fixed to 0
    ):
        super().__init__()
        self.save_hyperparameters()

        identity_dim = int(use_seg_ids) + int(use_task_ids)
        input_dim = x_dim + y_dim + identity_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.input_noise = input_noise

        if rope:
            self.pos_encoder = None
        else:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        config = GPTConfig(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=d_model,
            dropout=dropout,
            bias=True,
        )
        self.blocks = nn.ModuleList([GPTBlock(config, rope=rope) for _ in range(n_layer)])
        self.ln_f = GPTLayerNorm(d_model, bias=True)

        # Gaussian output: mean + logvar
        self.output_proj = nn.Linear(d_model, 2 * y_dim)

    def _prepare_input_sequence(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        seg_ids: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forcing style:
        model sees [x_t, y_{t-1}, optional ids] and predicts y_t.
        """
        B, T, _ = x.shape

        y0 = torch.full(
            (B, 1, self.hparams.y_dim),
            float(Y_PADDING),
            dtype=x.dtype,
            device=x.device,
        )
        y_shifted = torch.cat([y0, y[:, :-1, :]], dim=1)

        # If training, add noise to inputs for regularization
        # Add noise only to the continuous inputs, not the identity features.
        if self.training and self.hparams.input_noise > 0.0:
            x = x + torch.randn_like(x) * self.hparams.input_noise
            y_shifted = y_shifted + torch.randn_like(y_shifted) * self.hparams.input_noise

        pieces = [x, y_shifted]

        if seg_ids is not None and self.hparams.use_seg_ids:
            pieces.append(seg_ids.unsqueeze(-1).to(dtype=x.dtype) * self.hparams.seg_id_scale)

        if task_ids is not None and self.hparams.use_task_ids:
            pieces.append(task_ids.unsqueeze(-1).to(dtype=x.dtype) * self.hparams.task_id_scale)

        model_input = torch.cat(pieces, dim=-1)
        targets = y
        return model_input, targets

    def _encode(self, src: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(src)
        x = x * math.sqrt(self.hparams.d_model)

        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        x = self.input_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        seg_ids: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, _ = self._prepare_input_sequence(x, y, seg_ids=seg_ids, task_ids=task_ids)
        hidden = self._encode(src)
        pred = self.output_proj(hidden)
        mu, logvar = pred.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=self.hparams.min_logvar, max=self.hparams.max_logvar)
        return mu, logvar

    @staticmethod
    def masked_gaussian_nll(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        mask: [B,T] boolean, True where loss is included
        """
        var = torch.exp(logvar)
        nll = 0.5 * (math.log(2.0 * math.pi) + logvar + (y_true - mu) ** 2 / var)  # [B,T,1]
        nll = nll.squeeze(-1)  # [B,T]

        mask = mask.to(dtype=nll.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (nll * mask).sum() / denom

    @staticmethod
    def masked_mse(
        mu: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mse = ((mu - y_true) ** 2).squeeze(-1)  # [B,T]
        mask = mask.to(dtype=mse.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (mse * mask).sum() / denom

    def _unpack_batch(self, batch):
        if len(batch) == 5:
            x, y, seg_ids, task_ids, role_ids = batch
        elif len(batch) == 3:
            x, y, role_ids = batch
            seg_ids, task_ids = None, None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        return x, y, seg_ids, task_ids, role_ids

    def _shared_step(self, batch, stage: str):
        x, y, seg_ids, task_ids, role_ids = self._unpack_batch(batch)

        mu, logvar = self(x, y, seg_ids=seg_ids, task_ids=task_ids)

        # Train/eval only on target datapoints, not prior datasets and not special tokens
        pred_mask = (role_ids == ROLE_PREDICTION)  # [B,T]

        loss = self.masked_gaussian_nll(mu, logvar, y, pred_mask)
        mse = self.masked_mse(mu, y, pred_mask)

        self.log(
            f"{stage}_nll",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{stage}_mse",
            mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        stage = "test" if dataloader_idx == 1 else "val"
        self._shared_step(batch, stage)

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_steps = int(self.hparams.warmup_steps)
        max_steps = int(self.hparams.max_steps)
        cosine_steps = max(1, max_steps - warmup_steps)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[max(1, warmup_steps)],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
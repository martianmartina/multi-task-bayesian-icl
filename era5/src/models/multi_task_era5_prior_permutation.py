import math
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.multi_task_era5 import (
    ROLE_PREDICTION,
    SEG_PRIOR,
    SEG_TARGET,
)
from src.models.gpt2 import GPTConfig, LayerNorm as GPTLayerNorm, MLP
from src.models.multi_task_era5_implicit import Y_PADDING


class LocalSinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding addressed by caller-provided local positions.
    This lets every dataset restart at position zero.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if positions.max().item() >= self.pe.size(0):
            raise ValueError(
                f"Local position {int(positions.max())} exceeds max_len={self.pe.size(0)}."
            )
        return x + self.pe[positions].to(dtype=x.dtype)


class LocalRotaryEmbedding(nn.Module):
    """
    RoPE with caller-provided local positions.

    position_ids may contain negative entries; those tokens are left unrotated.
    This is useful in the final block, where prior tokens are already encoded
    with local order and only target tokens should receive a fresh order signal.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.rotary_dim = dim - (dim % 2)
        if self.rotary_dim == 0:
            inv_freq = torch.empty(0)
        else:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_dim == 0:
            return q, k

        bsz, _n_head, seq_len, _head_dim = q.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(bsz, -1)
        else:
            if position_ids.shape != (bsz, seq_len):
                raise ValueError(
                    f"position_ids must have shape {(bsz, seq_len)}, got {tuple(position_ids.shape)}."
                )
            position_ids = position_ids.to(device=q.device)

        active = position_ids >= 0
        positions = position_ids.clamp_min(0).to(dtype=self.inv_freq.dtype)
        freqs = torch.einsum("bt,d->btd", positions, self.inv_freq)
        cos = freqs.cos().to(dtype=q.dtype).unsqueeze(1)
        sin = freqs.sin().to(dtype=q.dtype).unsqueeze(1)
        active = active.unsqueeze(1).unsqueeze(-1)
        cos = torch.where(active, cos, torch.ones_like(cos))
        sin = torch.where(active, sin, torch.zeros_like(sin))

        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        rotated = torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        ).flatten(-2)
        return torch.cat((rotated, x_pass), dim=-1)


class MaskedSelfAttention(nn.Module):
    """
    Self-attention with caller-provided boolean allow masks.

    attn_allow_mask shape is [T, T], where True means query row may attend to key column.
    If no mask is given, attention is fully bidirectional.
    """

    def __init__(self, config: GPTConfig, rope: bool = False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.rope = LocalRotaryEmbedding(self.head_dim) if rope else None

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_allow_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, channels = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k, position_ids=position_ids)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_mask = None
            if attn_allow_mask is not None:
                attn_mask = attn_allow_mask.to(device=x.device, dtype=torch.bool)
                attn_mask = attn_mask.view(1, 1, seq_len, seq_len)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attn_allow_mask is not None:
                mask = attn_allow_mask.to(device=x.device, dtype=torch.bool)
                att = att.masked_fill(~mask.view(1, 1, seq_len, seq_len), float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MaskedTransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig, rope: bool = False):
        super().__init__()
        self.ln_1 = GPTLayerNorm(config.n_embd, bias=config.bias)
        self.attn = MaskedSelfAttention(config, rope=rope)
        self.ln_2 = GPTLayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attn_allow_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln_1(x),
            attn_allow_mask=attn_allow_mask,
            position_ids=position_ids,
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiTaskERA5PriorPermutationInvariantLearner(pl.LightningModule):
    """
    Ablation model with permutation invariance to the order of prior datasets.
    """

    def __init__(
        self,
        x_dim: int = 4,
        y_dim: int = 1,
        d_model: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        prior_n_layer: Optional[int] = None,
        mix_n_layer: Optional[int] = None,
        final_n_layer: Optional[int] = None,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 500,
        max_steps: int = 50_000,
        use_seg_ids: bool = True,
        use_task_ids: bool = False,
        task_id_scale: float = 1.0,
        seg_id_scale: float = 1.0,
        rope: bool = True,
        max_local_len: int = 4096,
        min_logvar: float = -8.0,
        max_logvar: float = 8.0,
        input_noise: float = 0.0, # fixed to 0
    ):
        super().__init__()
        if use_task_ids:
            raise ValueError(
                "use_task_ids=True would expose prior dataset order to this ablation. "
                "task_ids are used only as boundary metadata."
            )
        self.save_hyperparameters()

        prior_n_layer = n_layer if prior_n_layer is None else int(prior_n_layer)
        mix_n_layer = n_layer if mix_n_layer is None else int(mix_n_layer)
        final_n_layer = n_layer if final_n_layer is None else int(final_n_layer)
        self.hparams.prior_n_layer = prior_n_layer
        self.hparams.mix_n_layer = mix_n_layer
        self.hparams.final_n_layer = final_n_layer

        identity_dim = int(use_seg_ids) + int(use_task_ids)
        input_dim = x_dim + y_dim + identity_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.input_noise = input_noise
        self.local_pos_encoder = None
        if not rope:
            self.local_pos_encoder = LocalSinusoidalPositionalEncoding(
                d_model=d_model,
                max_len=max_local_len,
            )

        config = GPTConfig(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=d_model,
            dropout=dropout,
            bias=True,
        )
        self.prior_blocks = nn.ModuleList(
            [MaskedTransformerBlock(config, rope=rope) for _ in range(prior_n_layer)]
        )
        self.mix_blocks = nn.ModuleList(
            [MaskedTransformerBlock(config, rope=False) for _ in range(mix_n_layer)]
        )
        self.final_blocks = nn.ModuleList(
            [MaskedTransformerBlock(config, rope=rope) for _ in range(final_n_layer)]
        )
        self.ln_f = GPTLayerNorm(d_model, bias=True)
        self.output_proj = nn.Linear(d_model, 2 * y_dim)

    def _prepare_input_sequence(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        seg_ids: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher forcing with y shifted only within each dataset. This avoids the
        first token of a prior dataset seeing the previous prior dataset's last y.
        """
        y_shifted = torch.empty_like(y)
        y_shifted[:, 0, :] = float(Y_PADDING)
        y_shifted[:, 1:, :] = y[:, :-1, :]

        task_changed = task_ids[:, 1:] != task_ids[:, :-1]
        y_shifted[:, 1:, :] = torch.where(
            task_changed.unsqueeze(-1),
            torch.full_like(y_shifted[:, 1:, :], float(Y_PADDING)),
            y_shifted[:, 1:, :],
        )

        # Add noise only to the continuous inputs, not the identity features.
        if self.training and self.hparams.input_noise > 0.0:
            x = x + torch.randn_like(x) * self.hparams.input_noise
            y_shifted = y_shifted + torch.randn_like(y_shifted) * self.hparams.input_noise

        pieces = [x, y_shifted]

        if self.hparams.use_seg_ids:
            pieces.append(seg_ids.unsqueeze(-1).to(dtype=x.dtype) * self.hparams.seg_id_scale)

        if self.hparams.use_task_ids:
            pieces.append(task_ids.unsqueeze(-1).to(dtype=x.dtype) * self.hparams.task_id_scale)

        return torch.cat(pieces, dim=-1)

    @staticmethod
    def _causal_allow_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    @staticmethod
    def _final_allow_mask(
        num_prior_tokens: int,
        num_target_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        seq_len = num_prior_tokens + num_target_tokens
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        if num_prior_tokens > 0:
            mask[:num_prior_tokens, :num_prior_tokens] = True
        tgt = torch.tril(torch.ones(num_target_tokens, num_target_tokens, device=device, dtype=torch.bool))
        mask[num_prior_tokens:, :num_prior_tokens] = True
        mask[num_prior_tokens:, num_prior_tokens:] = tgt
        return mask

    def _embed(self, src: torch.Tensor) -> torch.Tensor:
        return self.input_proj(src) * math.sqrt(self.hparams.d_model)

    def _embed_with_local_positions(self, src: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = src.shape
        x = self._embed(src)
        if self.local_pos_encoder is not None:
            local_pos = torch.arange(seq_len, device=src.device).unsqueeze(0).expand(bsz, -1)
            x = self.local_pos_encoder(x, local_pos)
        return self.input_dropout(x)

    def _run_blocks(
        self,
        x: torch.Tensor,
        blocks: nn.ModuleList,
        attn_allow_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in blocks:
            x = block(
                x,
                attn_allow_mask=attn_allow_mask,
                position_ids=position_ids,
            )
        return x

    def _split_sequence(
        self,
        src: torch.Tensor,
        seg_ids: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        bsz = src.size(0)
        prior_task_values = torch.unique(task_ids[seg_ids == SEG_PRIOR], sorted=True)
        num_prior_datasets = int(prior_task_values.numel())

        target_mask = seg_ids == SEG_TARGET
        num_target_tokens = int(target_mask[0].sum().item())
        if not torch.all(target_mask.sum(dim=1) == num_target_tokens):
            raise ValueError("All samples in a batch must have the same number of target tokens.")
        target_src = src[target_mask].view(bsz, num_target_tokens, -1)

        if num_prior_datasets == 0:
            empty_prior = src.new_empty(bsz, 0, src.size(-1))
            return empty_prior, target_src, prior_task_values, 0, 0, num_target_tokens

        prior_chunks = []
        prior_len = None
        for task_value in prior_task_values:
            ds_mask = (seg_ids == SEG_PRIOR) & (task_ids == task_value)
            ds_len = int(ds_mask[0].sum().item())
            if not torch.all(ds_mask.sum(dim=1) == ds_len):
                raise ValueError("All samples in a batch must have matching prior dataset lengths.")
            if prior_len is None:
                prior_len = ds_len
            elif prior_len != ds_len:
                raise ValueError("Prior datasets must have a common length for batched encoding.")
            prior_chunks.append(src[ds_mask].view(bsz, ds_len, -1))

        prior_src = torch.stack(prior_chunks, dim=1)
        return (
            prior_src,
            target_src,
            prior_task_values,
            num_prior_datasets,
            int(prior_len),
            num_target_tokens,
        )

    def _encode(
        self,
        src: torch.Tensor,
        seg_ids: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        (
            prior_src,
            target_src,
            _prior_task_values,
            num_prior_datasets,
            prior_len,
            target_len,
        ) = self._split_sequence(src, seg_ids=seg_ids, task_ids=task_ids)
        bsz = src.size(0)

        if num_prior_datasets > 0:
            prior_flat = prior_src.view(bsz * num_prior_datasets, prior_len, -1)
            prior_hidden = self._embed_with_local_positions(prior_flat)
            prior_position_ids = torch.arange(
                prior_len,
                device=src.device,
            ).unsqueeze(0).expand(bsz * num_prior_datasets, -1)
            prior_causal_mask = self._causal_allow_mask(prior_len, device=src.device)
            prior_hidden = self._run_blocks(
                prior_hidden,
                self.prior_blocks,
                attn_allow_mask=prior_causal_mask,
                position_ids=prior_position_ids if self.hparams.rope else None,
            )
            prior_hidden = prior_hidden.view(bsz, num_prior_datasets * prior_len, -1)
            prior_hidden = self.input_dropout(prior_hidden)
            prior_hidden = self._run_blocks(prior_hidden, self.mix_blocks, attn_allow_mask=None)
        else:
            prior_hidden = src.new_empty(bsz, 0, self.hparams.d_model)

        target_hidden = self._embed_with_local_positions(target_src)
        final_hidden = torch.cat([prior_hidden, target_hidden], dim=1)
        final_mask = self._final_allow_mask(
            num_prior_tokens=prior_hidden.size(1),
            num_target_tokens=target_len,
            device=src.device,
        )
        final_position_ids = None
        if self.hparams.rope:
            final_position_ids = torch.full(
                (bsz, final_hidden.size(1)),
                -1,
                device=src.device,
                dtype=torch.long,
            )
            final_position_ids[:, prior_hidden.size(1) :] = torch.arange(
                target_len,
                device=src.device,
            ).unsqueeze(0)
        final_hidden = self._run_blocks(
            final_hidden,
            self.final_blocks,
            attn_allow_mask=final_mask,
            position_ids=final_position_ids,
        )
        return self.ln_f(final_hidden)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        seg_ids: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seg_ids is None or task_ids is None:
            raise ValueError(
                "Permutation-invariant ablation requires seg_ids and task_ids for dataset boundaries."
            )

        src = self._prepare_input_sequence(x, y, seg_ids=seg_ids, task_ids=task_ids)
        hidden = self._encode(src, seg_ids=seg_ids, task_ids=task_ids)
        if hidden.shape[:2] != y.shape[:2]:
            raise ValueError(
                f"Encoded sequence shape {tuple(hidden.shape[:2])} does not match "
                f"target shape {tuple(y.shape[:2])}."
            )
        pred = self.output_proj(hidden)
        mu, logvar = pred.chunk(2, dim=-1)
        logvar = torch.clamp(
            logvar,
            min=self.hparams.min_logvar,
            max=self.hparams.max_logvar,
        )
        return mu, logvar

    @staticmethod
    def masked_gaussian_nll(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        var = torch.exp(logvar)
        nll = 0.5 * (math.log(2.0 * math.pi) + logvar + (y_true - mu) ** 2 / var)
        nll = nll.squeeze(-1)
        mask = mask.to(dtype=nll.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (nll * mask).sum() / denom

    @staticmethod
    def masked_mse(
        mu: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mse = ((mu - y_true) ** 2).squeeze(-1)
        mask = mask.to(dtype=mse.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (mse * mask).sum() / denom

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) != 5:
            raise ValueError(
                "Permutation-invariant ablation requires (x, y, seg_ids, task_ids, role_ids)."
            )
        return batch

    def _shared_step(self, batch, stage: str):
        x, y, seg_ids, task_ids, role_ids = self._unpack_batch(batch)
        mu, logvar = self(x, y, seg_ids=seg_ids, task_ids=task_ids)

        pred_mask = role_ids == ROLE_PREDICTION
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

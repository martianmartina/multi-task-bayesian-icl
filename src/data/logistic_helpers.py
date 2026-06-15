import torch
import torch.nn as nn
from typing import Optional, Tuple

PRIOR_TOKEN_X = -10.0
TARGET_TOKEN_X = -11.0
SEP_TOKEN_X = -12.0
THINKING_TOKEN_X = -13.0
# Only calculate loss for the prediction tokens
ROLE_PRIOR = 0      
ROLE_EVIDENCE = 1  
ROLE_THINKING = 2   
ROLE_PREDICTION = 3
ROLE_PADDING = -1

# y label space: {0, 1}, -1 is padding
Y_PADDING = -1

SEG_PRIOR  = 0
SEG_TARGET = 1

def make_transform(
    transform_name: str,
    x_dim: int,
    generator: Optional[torch.Generator] = None,
) -> Optional[nn.Module]:
    """
    Create a fresh transform module.

    NOTE:
      - This is intentionally a factory so we can resample flow params per batch/sample.
      - Returned module is CPU by default.
    """
    from data.transforms import RadialFlow, PlanarFlow, ChaoticFlow, ComposedFlow, SpiralFlow

    name = str(transform_name).lower()
    if name == "identity":
        return None
    if name == "radial_flow":
        return RadialFlow(x_dim, generator=generator)
    if name == "planar_flow":
        return PlanarFlow(x_dim, generator=generator)
    if name == "chaotic_flow":
        return ChaoticFlow(x_dim, generator=generator)
    if name == "spiral_flow":
        return SpiralFlow(x_dim, generator=generator, verbose=False)
    if name == "composed_flow":
        return ComposedFlow(x_dim, depth=12, generator=generator)
    raise ValueError(f"Unsupported transform: {transform_name}")


def as_token_vec(token, x_dim: int, dtype=torch.float32) -> torch.Tensor:
    """
    Ensure token becomes a 1D tensor of shape [x_dim].
    Supports scalar or vector tokens.
    """
    t = torch.as_tensor(token, dtype=dtype)
    if t.ndim == 0:
        return t.repeat(x_dim)
    if t.numel() == 1:
        return t.view(1).repeat(x_dim)
    assert t.shape[-1] == x_dim, f"Token dim mismatch: token has {t.shape}, expected last dim {x_dim}"
    return t.view(x_dim)


def build_templates(
    K: int,
    T: int,
    T_ctx: int,
    x_dim: int,
    num_evidence_tokens: int,
    num_thinking_tokens: int,
    add_task_ids: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-sequence templates that are identical for all samples.
    These include:
      - token rows (PRIOR/TARGET/THINKING)
      - seg/task/role ids
      - Y initialized to padding (X for data rows filled later)
    """
    # Total length
    L = K * (T + 1) + 1 + T_ctx + num_thinking_tokens

    X_tpl = torch.zeros(L, x_dim, dtype=torch.float32)
    Y_tpl = torch.full((L, 1), Y_PADDING, dtype=torch.float32)

    seg_tpl  = torch.zeros(L, dtype=torch.long)
    task_tpl = torch.zeros(L, dtype=torch.long)
    role_tpl = torch.zeros(L, dtype=torch.long)

    prior_token = as_token_vec(PRIOR_TOKEN_X, x_dim)
    target_token = as_token_vec(TARGET_TOKEN_X, x_dim)
    thinking_token = as_token_vec(THINKING_TOKEN_X, x_dim)

    pos = 0

    # Prior blocks: [PRIOR_TOKEN] + T data rows
    for t in range(K):
        X_tpl[pos] = prior_token
        task_tpl[pos] = t
        role_tpl[pos] = ROLE_PRIOR
        pos += 1

        task_tpl[pos:pos+T] = t
        role_tpl[pos:pos+T] = ROLE_PRIOR
        pos += T

    # Target separator
    X_tpl[pos] = target_token
    seg_tpl[pos] = SEG_TARGET
    task_tpl[pos] = K
    role_tpl[pos] = ROLE_EVIDENCE
    pos += 1

    # Evidence rows (subset of target context)
    if num_evidence_tokens > 0:
        seg_tpl[pos:pos+num_evidence_tokens] = SEG_TARGET
        task_tpl[pos:pos+num_evidence_tokens] = K
        role_tpl[pos:pos+num_evidence_tokens] = ROLE_EVIDENCE
        pos += num_evidence_tokens

    # Thinking rows
    if num_thinking_tokens > 0:
        X_tpl[pos:pos+num_thinking_tokens] = thinking_token
        seg_tpl[pos:pos+num_thinking_tokens] = SEG_TARGET
        task_tpl[pos:pos+num_thinking_tokens] = K
        role_tpl[pos:pos+num_thinking_tokens] = ROLE_THINKING
        pos += num_thinking_tokens

    # Prediction rows (remaining target context)
    num_pred = T_ctx - num_evidence_tokens
    if num_pred > 0:
        seg_tpl[pos:pos+num_pred] = SEG_TARGET
        task_tpl[pos:pos+num_pred] = K
        role_tpl[pos:pos+num_pred] = ROLE_PREDICTION

    return X_tpl.contiguous(), Y_tpl.contiguous(), seg_tpl.contiguous(), task_tpl.contiguous(), role_tpl.contiguous()

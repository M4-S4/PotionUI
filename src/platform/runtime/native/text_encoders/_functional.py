"""Small functional helpers shared by the vendored text-encoder architectures.

These replace the ComfyUI helpers the vendored modules originally called
(``comfy.ldm.modules.attention.optimized_attention_for_device``,
``comfy.rmsnorm.rms_norm``, ``comfy.ops.cast_to_input``) with dependency-free
equivalents so the native engine never imports ComfyUI at runtime.

Numerics are matched to ComfyUI's default PyTorch paths so a golden comparison
against ComfyUI stays valid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cast_to_input(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Cast ``weight`` to the dtype+device of ``x`` (ComfyUI ``cast_to_input``)."""
    return weight.to(dtype=x.dtype, device=x.device)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS norm matching ``comfy.rmsnorm.rms_norm`` (torch's fused path)."""
    return F.rms_norm(x, weight.shape, weight=weight.to(dtype=x.dtype, device=x.device), eps=eps)


def optimized_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
    skip_reshape: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Scaled-dot-product attention matching ComfyUI's ``attention_pytorch``.

    ``skip_reshape=False``: ``q/k/v`` are ``[B, S, heads*dim]`` and get folded to
    ``[B, heads, S, dim]``. ``skip_reshape=True``: ``q/k/v`` already
    ``[B, heads, S, dim]`` (the RoPE path in the Llama/Qwen attention). Returns
    ``[B, S, heads*dim]``.

    ``scale`` overrides the softmax temperature; ``None`` keeps torch's
    ``1/sqrt(dim)`` default. Architectures whose learned QK-norms carry the
    temperature (gemma3n/gemma4: ``scaling = 1.0``) must pass it explicitly.
    """
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_qk(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    org = xq.dtype
    q = (xq * cos) + (rotate_half(xq) * sin)
    k = (xk * cos) + (rotate_half(xk) * sin)
    return q.to(org), k.to(org)

def module_device(module: nn.Module) -> torch.device:
    for p in module.parameters():
        if p is not None:
            return p.device
    for b in module.buffers():
        if b is not None:
            return b.device
    raise RuntimeError(f"{module.__class__.__name__} has no tensors to read a device from")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)

class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, operations, merged: bool = False, device=None, dtype=None) -> None:
        super().__init__()
        self.merged = merged
        if merged:
            self.gate_up_proj = operations.Linear(hidden_size, 2 * intermediate_size, bias=False, device=device, dtype=dtype)
        else:
            self.gate_proj = operations.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
            self.up_proj = operations.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

# Derived from: github.com/capitan01R/ComfyUI-Krea2T-Enhancer (MIT, Copyright (c) 2026 capitan01R; re-upload github.com/K3nt3L/ComfyUI-Krea2-Enhancer v1.3.0)
from __future__ import annotations

import math

import torch
from torch import Tensor

TAPS = 12
CHUNKS_PER_TAP = 2

_PROFILE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0)
CHUNK_PROFILE = _PROFILE + _PROFILE
GLOBAL_GAIN = 15.0
TOKEN_REL_CAP = 0.75
_EPS = 1e-8


def fusion_key(fusion: dict | None) -> tuple[float, float] | None:
    if not fusion:
        return None
    strength = float(fusion.get("strength", 0.0))
    scale = float(fusion.get("scale", 1.0))
    if not (math.isfinite(strength) and math.isfinite(scale)):
        raise ValueError(f"text_fusion settings must be finite, got strength={strength} scale={scale}")
    if strength < 0.0:
        raise ValueError(f"text_fusion strength must be >= 0, got {strength}")
    if scale <= 0.0:
        raise ValueError(f"text_fusion scale must be > 0, got {scale}")
    if strength == 0.0 and scale == 1.0:
        return None
    return strength, scale


def chunk_gains(strength: float) -> Tensor:
    base = torch.tensor(CHUNK_PROFILE, dtype=torch.float32)
    return (1.0 + strength * (base - 1.0)) * (1.0 + strength * (GLOBAL_GAIN - 1.0))


def scaled_input(x: Tensor, strength: float) -> Tensor:
    b, s, taps, dim = x.shape
    if taps != TAPS or dim % CHUNKS_PER_TAP:
        raise ValueError(
            f"text fusion enhancer needs a ({TAPS}, even-width) tap axis, got {taps} taps of width {dim}"
        )
    gains = chunk_gains(strength).to(device=x.device)
    chunks = x.reshape(b, s, taps * CHUNKS_PER_TAP, dim // CHUNKS_PER_TAP).float()
    limit = torch.finfo(x.dtype).max
    scaled = (chunks * gains.view(1, 1, -1, 1)).clamp(-limit, limit)
    return scaled.to(x.dtype).reshape(b, s, taps, dim)


def enhanced_fusion(txtfusion, x: Tensor, reference: Tensor, mask: Tensor | None, strength: float) -> Tensor:
    candidate = txtfusion(scaled_input(x, strength), mask=mask).float()
    base = reference.float()
    shift = candidate - base
    finite = torch.isfinite(candidate).all(dim=-1, keepdim=True) & torch.isfinite(shift).all(dim=-1, keepdim=True)
    delta = torch.where(finite, shift, torch.zeros_like(base))
    base_rms = base.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(_EPS)
    delta_rms = delta.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(_EPS)
    keep = (TOKEN_REL_CAP / (delta_rms / base_rms)).clamp(max=1.0)
    return (base + delta * keep).to(reference.dtype)

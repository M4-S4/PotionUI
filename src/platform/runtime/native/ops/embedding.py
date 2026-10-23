from __future__ import annotations

import math

import torch


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000) -> torch.Tensor:
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    args = t[:, None].float() * torch.exp(exponent)[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

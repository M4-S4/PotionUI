from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ...lora.key_mapping import LoraDelta


def _slice_of(target_slice: tuple) -> tuple:
    _dim, start, length = target_slice
    return (slice(start, start + length),)


def _term(pool, name: str, device: torch.device, shape: torch.Size) -> torch.Tensor:
    if pool is None:
        return torch.empty(shape, dtype=torch.float32, device=device)
    return pool.view(name, device, torch.float32, shape)


def accumulate_delta(
    out: torch.Tensor,
    deltas: Sequence[LoraDelta],
    compute_device: torch.device,
    pool=None,
) -> None:
    for d in deltas:
        up = d.up.to(device=compute_device, dtype=torch.float32)
        down = d.down.to(device=compute_device, dtype=torch.float32)
        dest = out[_slice_of(d.target_slice)] if d.target_slice is not None else out
        if d.kron:
            natural_shape = torch.Size((up.shape[0] * down.shape[0], up.shape[1] * down.shape[1]))
            term = _term(pool, "term", compute_device, natural_shape)
            torch.kron(up, down, out=term)
            term = term.view(dest.shape)
            term.mul_(float(d.scale) * float(d.alpha))
        else:
            rank = d.down.shape[0]
            term = _term(pool, "term", compute_device, dest.shape)
            torch.matmul(up, down, out=term)
            term.mul_(float(d.scale) * float(d.alpha) / rank)
        dest += term


def compute_delta(
    deltas: Sequence[LoraDelta],
    shape: Sequence[int],
    device: torch.device | str | None = None,
    pool=None,
) -> torch.Tensor:
    compute_device = torch.device(device) if device is not None else torch.device("cpu")
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=compute_device)
    accumulate_delta(out, deltas, compute_device, pool)
    return out

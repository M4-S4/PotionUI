from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def pad_to_patch(x: torch.Tensor, patch: Sequence[int], mode: str = "circular") -> torch.Tensor:
    pad: tuple[int, ...] = ()
    for i in range(x.ndim - 2):
        pad = (0, (patch[i] - x.shape[i + 2] % patch[i]) % patch[i]) + pad
    if all(p == 0 for p in pad):
        return x
    return F.pad(x, pad, mode=mode)

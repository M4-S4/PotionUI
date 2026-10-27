from __future__ import annotations

import torch

from src.platform.runtime.native.kernels import pick_operations


def fp32_ops():
    return pick_operations(torch.float32, torch.float32)

from __future__ import annotations

import torch

from src.platform.runtime.native.ops.dtype import pick_dtypes


def test_is_mixed_precision():
    from src.platform.runtime.native.ops.dtype import is_mixed_precision
    # Krea-2 style: bf16 block Linears + f32 norm/peripheral weights.
    mixed = {
        "blocks.0.attn.wq.weight": torch.zeros(4, dtype=torch.bfloat16),
        "first.weight": torch.zeros(4, dtype=torch.float32),
    }
    assert is_mixed_precision(mixed) is True
    # single-dtype (Klein bf16) -> not mixed.
    single = {"a": torch.zeros(4, dtype=torch.bfloat16), "b": torch.zeros(4, dtype=torch.bfloat16)}
    assert is_mixed_precision(single) is False
    # fp8 weights + f32 scale sidecars must NOT read as mixed (scales excluded).
    fp8 = {
        "l.weight": torch.zeros(4, dtype=torch.float8_e4m3fn),
        "l.weight_scale": torch.zeros((), dtype=torch.float32),
        "l.input_scale": torch.zeros((), dtype=torch.float32),
    }
    assert is_mixed_precision(fp8) is False


def test_pick_dtypes_fp8_keeps_storage():
    storage, compute = pick_dtypes(torch.float8_e4m3fn, "cpu")
    assert storage == torch.float8_e4m3fn
    assert compute in (torch.bfloat16, torch.float16)


def test_pick_dtypes_fp32_downcasts():
    storage, compute = pick_dtypes(torch.float32, "cpu")
    assert storage == compute

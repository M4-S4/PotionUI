from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.platform.runtime.native.ops.dtype import pick_dtypes
from vendor.gpl.comfyui.ops import disable_weight_init, manual_cast
from src.platform.runtime.native.kernels import fp8_ops


def test_disable_weight_init_skips_init_and_runs():
    lin = disable_weight_init.Linear(4, 3)
    # reset_parameters is a no-op; weights are whatever load provides.
    lin.weight = nn.Parameter(torch.eye(3, 4))
    lin.bias = nn.Parameter(torch.zeros(3))
    out = lin(torch.ones(1, 4))
    assert out.shape == (1, 3)


def test_manual_cast_linear_fp16_storage_fp32_compute():
    lin = manual_cast.Linear(4, 3, bias=True)
    lin.weight = nn.Parameter(torch.randn(3, 4, dtype=torch.float16))
    lin.bias = nn.Parameter(torch.randn(3, dtype=torch.float16))
    x = torch.randn(2, 4, dtype=torch.float32)
    out = lin(x)
    assert out.dtype == torch.float32
    ref = F.linear(x, lin.weight.float(), lin.bias.float())
    assert torch.allclose(out, ref, atol=1e-3)


def test_manual_cast_stack_two_layers():
    net = nn.Sequential(
        manual_cast.Linear(4, 8, bias=False),
        manual_cast.Linear(8, 2, bias=False),
    )
    for m in net:
        m.weight = nn.Parameter(torch.randn(m.out_features, m.in_features, dtype=torch.float16))
    x = torch.randn(3, 4, dtype=torch.float32)
    out = net(x)
    assert out.shape == (3, 2)
    assert out.dtype == torch.float32






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












def test_ops_namespace_protocol_surface():
    for ns in (disable_weight_init, manual_cast, fp8_ops):
        for attr in ("Linear", "Conv2d", "Conv3d", "GroupNorm", "LayerNorm", "RMSNorm", "Embedding"):
            assert hasattr(ns, attr), f"{ns.__name__} missing {attr}"


def test_manual_cast_conv2d():
    conv = manual_cast.Conv2d(2, 2, 3, padding=1, bias=True)
    conv.weight = nn.Parameter(torch.randn(2, 2, 3, 3, dtype=torch.float16))
    conv.bias = nn.Parameter(torch.zeros(2, dtype=torch.float16))
    x = torch.randn(1, 2, 4, 4, dtype=torch.float32)
    out = conv(x)
    assert out.dtype == torch.float32
    assert out.shape == (1, 2, 4, 4)


def test_pick_dtypes_fp8_keeps_storage():
    storage, compute = pick_dtypes(torch.float8_e4m3fn, "cpu")
    assert storage == torch.float8_e4m3fn
    assert compute in (torch.bfloat16, torch.float16)


def test_pick_dtypes_fp32_downcasts():
    storage, compute = pick_dtypes(torch.float32, "cpu")
    assert storage == compute

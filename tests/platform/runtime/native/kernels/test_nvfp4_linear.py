from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import vendor.gpl.comfyui.ops as ops
from vendor.gpl.comfyui.ops import _E2M1_LUT, _NVFP4_BLOCK, _to_blocked, _unblock_scale, dequantize_nvfp4, disable_weight_init
from src.platform.runtime.native.kernels import detect_quant_format, fp8_ops, pick_operations

from .._nvfp4_ref import F4_MAX as _F4_MAX
from .._nvfp4_ref import F8_MAX as _F8_MAX
from .._nvfp4_ref import quantize_nvfp4


def _old_dequantize_nvfp4(
    packed: torch.Tensor, block_scale: torch.Tensor, tensor_scale: torch.Tensor,
    out_features: int, in_features: int,
) -> torch.Tensor:
    device = packed.device
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    codes = torch.empty(out_features, in_features, dtype=torch.long, device=device)
    codes[:, 0::2] = high.long()
    codes[:, 1::2] = low.long()
    values = _E2M1_LUT.to(device)[codes]
    num_blocks = in_features // _NVFP4_BLOCK
    block_nat = _unblock_scale(block_scale, out_features, num_blocks).to(device)
    scale = (block_nat * tensor_scale.to(torch.float32).to(device)).repeat_interleave(_NVFP4_BLOCK, dim=1)
    return values * scale


def test_to_blocked_inverse_roundtrips():
    m = torch.arange(4096 * 256).reshape(4096, 256).float()
    sw = _to_blocked(m)
    back = _unblock_scale(sw, 4096, 256)
    assert torch.equal(back, m)


def test_dequant_matches_independent_natural_reference():
    torch.manual_seed(0)
    out, inn = 256, 512
    w = torch.randn(out, inn) * 0.03
    pts = (w.abs().amax() / (_F4_MAX * _F8_MAX)).detach()
    packed, block_sw, codes_nat, block_nat = quantize_nvfp4(w, pts)

    expected = _E2M1_LUT[codes_nat] * (block_nat.to(torch.float32) * pts).repeat_interleave(_NVFP4_BLOCK, dim=1)
    got = dequantize_nvfp4(packed, block_sw, pts, out, inn)
    assert torch.allclose(got, expected, atol=1e-6)


def test_dequant_error_is_grid_level_not_garbage():
    torch.manual_seed(1)
    out, inn = 128, 256
    w = torch.randn(out, inn) * 0.02
    pts = (w.abs().amax() / (_F4_MAX * _F8_MAX)).detach()
    packed, block_sw, _, _ = quantize_nvfp4(w, pts)
    deq = dequantize_nvfp4(packed, block_sw, pts, out, inn)
    rel = (deq - w).abs().mean() / w.abs().mean()
    assert 0.0 < rel < 0.20


def test_nvfp4_linear_load_and_forward_parity():
    torch.manual_seed(2)
    out, inn = 256, 512
    w = torch.randn(out, inn) * 0.03
    pts = (w.abs().amax() / (_F4_MAX * _F8_MAX)).detach()
    packed, block_sw, _, _ = quantize_nvfp4(w, pts)
    bias = torch.randn(out)

    lin = fp8_ops.Linear(inn, out, bias=True)
    sd = {
        "weight": packed,
        "weight_scale": block_sw,
        "weight_scale_2": pts.clone(),
        "comfy_quant": torch.zeros(5, dtype=torch.uint8),
        "bias": bias,
    }
    missing, unexpected = [], []
    lin._load_from_state_dict(sd, "", {}, True, missing, unexpected, [])
    assert lin._is_nvfp4 and lin.weight is None
    assert missing == [] and unexpected == []
    assert set(sd) == {"bias"}

    x = torch.randn(3, inn)
    ref = torch.nn.functional.linear(x, dequantize_nvfp4(packed, block_sw, pts, out, inn), bias)
    assert torch.equal(lin(x), ref)


def _module_footprint_bytes(module: torch.nn.Module) -> int:
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        total += t.numel() * t.element_size()
    return total


def test_nvfp4_linear_forward_never_retains_dequantised_weight():
    torch.manual_seed(4)
    out, inn = 64, 256
    w = torch.randn(out, inn) * 0.03
    pts = (w.abs().amax() / (_F4_MAX * _F8_MAX)).detach()
    packed, block_sw, _, _ = quantize_nvfp4(w, pts)

    lin = fp8_ops.Linear(inn, out, bias=False)
    sd = {"weight": packed, "weight_scale": block_sw, "weight_scale_2": pts.clone()}
    missing, unexpected = [], []
    lin._load_from_state_dict(sd, "", {}, True, missing, unexpected, [])
    assert lin._is_nvfp4 and lin.weight is None

    baseline = _module_footprint_bytes(lin)
    assert baseline > 0

    lin.eval()
    x = torch.randn(3, inn)
    with torch.inference_mode():
        for _ in range(5):
            lin(x)
            assert _module_footprint_bytes(lin) == baseline
    assert lin.weight is None


def test_mixed_module_fp8_and_nvfp4_and_plain():
    torch.manual_seed(3)
    dim = 128

    mod = torch.nn.Module()
    mod.nv = fp8_ops.Linear(dim, dim, bias=False)
    mod.fp8 = fp8_ops.Linear(dim, dim, bias=False)
    mod.plain = fp8_ops.Linear(dim, dim, bias=False)

    w_nv = torch.randn(dim, dim) * 0.03
    pts = (w_nv.abs().amax() / (_F4_MAX * _F8_MAX)).detach()
    packed, block_sw, _, _ = quantize_nvfp4(w_nv, pts)

    w_fp8 = (torch.randn(dim, dim) * 0.02).to(torch.float8_e4m3fn)
    sd = {
        "nv.weight": packed, "nv.weight_scale": block_sw, "nv.weight_scale_2": pts.clone(),
        "nv.comfy_quant": torch.zeros(3, dtype=torch.uint8),
        "fp8.weight": w_fp8, "fp8.weight_scale": torch.tensor(0.5),
        "plain.weight": torch.randn(dim, dim) * 0.02,
    }
    missing, unexpected = [], []
    mod.load_state_dict
    torch.nn.Module.load_state_dict(mod, sd, strict=False, assign=True)

    assert mod.nv._is_nvfp4 is True
    assert mod.fp8._is_nvfp4 is False and mod.fp8.weight_scale is not None
    assert mod.plain._is_nvfp4 is False and mod.plain.weight_scale is None

    x = torch.randn(2, dim)
    for lin in (mod.nv, mod.fp8, mod.plain):
        lin.comfy_cast_weights = True
        y = lin(x)
        assert y.shape == (2, dim)
        assert torch.isfinite(y).all()


def test_detect_and_pick_route_nvfp4_to_fp8_ops():
    fmt = detect_quant_format({}, {"layer.weight_scale_2": torch.zeros(())})
    assert fmt == "fp8_scaled"
    assert pick_operations(torch.uint8, torch.float32, fmt) is fp8_ops
    assert pick_operations(torch.float32, torch.float32, None) is disable_weight_init


@pytest.mark.parametrize("out_features,in_features,seed", [
    (256, 512, 10),
    (200, 208, 11),
    (1, 16, 12),
    (129, 32, 13),
])
def test_new_dequant_bit_exact_vs_old_random_shapes(out_features, in_features, seed):
    torch.manual_seed(seed)
    w = torch.randn(out_features, in_features) * 0.05
    pts = (w.abs().amax() / (_F4_MAX * _F8_MAX)).clamp(min=1e-8).detach()
    packed, block_sw, _, _ = quantize_nvfp4(w, pts)

    old = _old_dequantize_nvfp4(packed, block_sw, pts, out_features, in_features)
    new = dequantize_nvfp4(packed, block_sw, pts, out_features, in_features)
    assert torch.equal(new, old)


def test_new_dequant_bit_exact_with_zero_tensor_scale():
    out_features, in_features = 32, 64
    torch.manual_seed(14)
    packed = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
    num_blocks = in_features // _NVFP4_BLOCK
    block_sw = _to_blocked(torch.rand(out_features, num_blocks)).to(torch.float8_e4m3fn)
    tensor_scale = torch.tensor(0.0)

    old = _old_dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    new = dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    assert torch.equal(new, old)
    assert torch.count_nonzero(new) == 0


def test_new_dequant_bit_exact_with_some_zero_block_scales():
    out_features, in_features = 16, 32
    torch.manual_seed(15)
    packed = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
    num_blocks = in_features // _NVFP4_BLOCK
    block = torch.rand(out_features, num_blocks)
    block[:, 0] = 0.0
    block_sw = _to_blocked(block).to(torch.float8_e4m3fn)
    tensor_scale = torch.tensor(0.02)

    old = _old_dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    new = dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    assert torch.equal(new, old)


def test_new_dequant_bit_exact_mixed_sign_codes():
    out_features, in_features = 1, 16
    packed = torch.tensor([[0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]], dtype=torch.uint8)
    block_sw = _to_blocked(torch.ones(out_features, 1)).to(torch.float8_e4m3fn)
    tensor_scale = torch.tensor(0.1)

    old = _old_dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    new = dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    assert torch.equal(new, old)
    assert (old > 0).any() and (old < 0).any()


def test_class_forward_bit_exact_vs_old_reference():
    torch.manual_seed(16)
    out_features, in_features = 200, 208
    w = torch.randn(out_features, in_features) * 0.04
    pts = (w.abs().amax() / (_F4_MAX * _F8_MAX)).clamp(min=1e-8).detach()
    packed, block_sw, _, _ = quantize_nvfp4(w, pts)
    bias = torch.randn(out_features)

    lin = fp8_ops.Linear(in_features, out_features, bias=True)
    sd = {
        "weight": packed, "weight_scale": block_sw, "weight_scale_2": pts.clone(),
        "comfy_quant": torch.zeros(3, dtype=torch.uint8), "bias": bias,
    }
    lin._load_from_state_dict(sd, "", {}, True, [], [], [])
    assert lin._is_nvfp4

    x = torch.randn(4, in_features)
    old_weight = _old_dequantize_nvfp4(packed, block_sw, pts, out_features, in_features)
    ref = torch.nn.functional.linear(x, old_weight, bias)
    assert torch.equal(lin(x), ref)


def test_forward_never_calls_inv_blocked_index_after_load():
    torch.manual_seed(17)
    out_features, in_features = 128, 256
    w = torch.randn(out_features, in_features) * 0.03
    pts = (w.abs().amax() / (_F4_MAX * _F8_MAX)).clamp(min=1e-8).detach()
    packed, block_sw, _, _ = quantize_nvfp4(w, pts)

    lin = fp8_ops.Linear(in_features, out_features, bias=False)
    sd = {"weight": packed, "weight_scale": block_sw, "weight_scale_2": pts.clone()}
    lin._load_from_state_dict(sd, "", {}, True, [], [], [])
    assert lin._is_nvfp4

    x = torch.randn(3, in_features)
    old_weight = _old_dequantize_nvfp4(packed, block_sw, pts, out_features, in_features)
    expected = torch.nn.functional.linear(x, old_weight)

    def _boom(*args, **kwargs):
        raise AssertionError("_inv_blocked_index must not run from the forward hot path")

    original = ops._inv_blocked_index
    ops._inv_blocked_index = _boom
    try:
        y = lin(x)
    finally:
        ops._inv_blocked_index = original

    assert torch.equal(y, expected)


def test_inv_blocked_index_is_reachable_at_load_time():
    torch.manual_seed(18)
    out_features, in_features = 32, 32
    w = torch.randn(out_features, in_features) * 0.03
    pts = (w.abs().amax() / (_F4_MAX * _F8_MAX)).clamp(min=1e-8).detach()
    packed, block_sw, _, _ = quantize_nvfp4(w, pts)

    def _boom(*args, **kwargs):
        raise AssertionError("boom")

    lin = fp8_ops.Linear(in_features, out_features, bias=False)
    sd = {"weight": packed, "weight_scale": block_sw, "weight_scale_2": pts.clone()}
    original = ops._inv_blocked_index
    ops._inv_blocked_index = _boom
    try:
        with pytest.raises(AssertionError, match="boom"):
            lin._load_from_state_dict(sd, "", {}, True, [], [], [])
    finally:
        ops._inv_blocked_index = original

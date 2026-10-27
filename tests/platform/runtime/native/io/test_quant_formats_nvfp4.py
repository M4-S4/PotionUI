from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.io.quant_formats import (
    _E2M1_LUT,
    NVFP4_BLOCK,
    _to_blocked,
    _unblock_scale,
    dequant_values,
    dequantize_nvfp4,
    precompute_nvfp4_scale,
    quantize_nvfp4_dynamic,
    scale_values,
)

from .._nvfp4_ref import F4_MAX as _F4_MAX
from .._nvfp4_ref import F8_MAX as _F8_MAX
from .._nvfp4_ref import old_dequantize_nvfp4, quantize_nvfp4


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

    expected = _E2M1_LUT[codes_nat] * (block_nat.to(torch.float32) * pts).repeat_interleave(NVFP4_BLOCK, dim=1)
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

    old = old_dequantize_nvfp4(packed, block_sw, pts, out_features, in_features)
    new = dequantize_nvfp4(packed, block_sw, pts, out_features, in_features)
    assert torch.equal(new, old)


def test_new_dequant_bit_exact_with_zero_tensor_scale():
    out_features, in_features = 32, 64
    torch.manual_seed(14)
    packed = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
    num_blocks = in_features // NVFP4_BLOCK
    block_sw = _to_blocked(torch.rand(out_features, num_blocks)).to(torch.float8_e4m3fn)
    tensor_scale = torch.tensor(0.0)

    old = old_dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    new = dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    assert torch.equal(new, old)
    assert torch.count_nonzero(new) == 0


def test_new_dequant_bit_exact_with_some_zero_block_scales():
    out_features, in_features = 16, 32
    torch.manual_seed(15)
    packed = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
    num_blocks = in_features // NVFP4_BLOCK
    block = torch.rand(out_features, num_blocks)
    block[:, 0] = 0.0
    block_sw = _to_blocked(block).to(torch.float8_e4m3fn)
    tensor_scale = torch.tensor(0.02)

    old = old_dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    new = dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    assert torch.equal(new, old)


def test_new_dequant_bit_exact_mixed_sign_codes():
    out_features, in_features = 1, 16
    packed = torch.tensor([[0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]], dtype=torch.uint8)
    block_sw = _to_blocked(torch.ones(out_features, 1)).to(torch.float8_e4m3fn)
    tensor_scale = torch.tensor(0.1)

    old = old_dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    new = dequantize_nvfp4(packed, block_sw, tensor_scale, out_features, in_features)
    assert torch.equal(new, old)
    assert (old > 0).any() and (old < 0).any()

def test_quantize_nvfp4_dynamic_matches_weight_quantizer_nibble_convention():
    torch.manual_seed(30)
    m, k = 8, 64
    x = torch.randn(m, k) * 0.3
    packed, block_sw, tensor_scale = quantize_nvfp4_dynamic(x)

    ref_packed, ref_block_sw, _, _ = quantize_nvfp4(x, tensor_scale)
    assert torch.equal(packed, ref_packed)
    assert torch.equal(block_sw, ref_block_sw)


def test_quantize_nvfp4_dynamic_output_dtypes_and_scale_shape():
    x = torch.randn(16, 64) * 0.2
    packed, block_sw, tensor_scale = quantize_nvfp4_dynamic(x)
    assert packed.dtype == torch.uint8 and packed.shape == (16, 32)
    assert block_sw.dtype == torch.float8_e4m3fn
    assert tensor_scale.dtype == torch.float32 and tensor_scale.ndim == 0


@pytest.mark.parametrize("m,k", [(200, 64), (17, 32), (129, 512)])
def test_quantize_nvfp4_dynamic_blocked_scale_matches_scaled_mm_v2_size_contract(m, k):
    x = torch.randn(m, k) * 0.1
    packed, block_sw, _ = quantize_nvfp4_dynamic(x)
    num_blocks = -(-k // 16)
    nrb = -(-m // 128)
    ncb = -(-num_blocks // 4)
    assert block_sw.numel() == nrb * 128 * ncb * 4


def test_quantize_nvfp4_dynamic_roundtrip_within_grid_tolerance():
    torch.manual_seed(31)
    m, k = 16, 128
    x = torch.randn(m, k) * 0.05
    packed, block_sw, tensor_scale = quantize_nvfp4_dynamic(x)
    deq = dequantize_nvfp4(packed, block_sw, tensor_scale, m, k)
    rel = (deq - x).abs().mean() / x.abs().mean()
    assert 0.0 <= rel < 0.25


def test_quantize_nvfp4_dynamic_handles_all_zero_block_without_nan():
    x = torch.zeros(16, 32)
    x[0, :16] = torch.randn(16) * 0.1
    packed, block_sw, tensor_scale = quantize_nvfp4_dynamic(x)
    deq = dequantize_nvfp4(packed, block_sw, tensor_scale, 16, 32)
    assert torch.isfinite(deq).all()
    assert torch.count_nonzero(deq[0, 16:]) == 0
    assert torch.count_nonzero(deq[1:, :]) == 0


def test_the_high_nibble_is_the_even_element_and_the_low_nibble_the_odd_one():
    packed = torch.tensor([[0x12, 0x9F]], dtype=torch.uint8)
    values = dequant_values(packed, 1, 4)
    assert values.tolist() == [[_E2M1_LUT[1].item(), _E2M1_LUT[2].item(), _E2M1_LUT[9].item(), _E2M1_LUT[15].item()]]
    assert values.tolist() == [[0.5, 1.0, -0.5, -6.0]]


def test_a_block_scale_covers_sixteen_consecutive_values():
    values = torch.ones(2, 2 * NVFP4_BLOCK)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scaled = scale_values(values, scale, 2, 2 * NVFP4_BLOCK, 2)
    assert scaled[0, :NVFP4_BLOCK].eq(1.0).all() and scaled[0, NVFP4_BLOCK:].eq(2.0).all()
    assert scaled[1, :NVFP4_BLOCK].eq(3.0).all() and scaled[1, NVFP4_BLOCK:].eq(4.0).all()


def test_the_precomputed_scale_is_the_unswizzled_block_scale_times_the_tensor_scale():
    block = torch.tensor([[1.0, 2.0, 0.5, 4.0], [0.25, 1.0, 2.0, 8.0], [1.0, 1.0, 1.0, 1.0]])
    stored = _to_blocked(block).to(torch.float8_e4m3fn)
    tensor_scale = torch.tensor(0.125)
    got = precompute_nvfp4_scale(stored, tensor_scale, 3, 4 * NVFP4_BLOCK)
    assert torch.equal(got, _unblock_scale(stored, 3, 4) * tensor_scale)
    assert torch.equal(got, block * tensor_scale)


def test_precomputing_the_scale_then_scaling_equals_dequantising_in_one_call():
    torch.manual_seed(42)
    out, inn = 64, 128
    w = torch.randn(out, inn) * 0.03
    tensor_scale = (w.abs().amax() / (_F4_MAX * _F8_MAX)).detach()
    packed, block_sw, _, _ = quantize_nvfp4(w, tensor_scale)
    scale = precompute_nvfp4_scale(block_sw, tensor_scale, out, inn)
    staged = scale_values(dequant_values(packed, out, inn), scale, out, inn, inn // NVFP4_BLOCK)
    assert torch.equal(staged, dequantize_nvfp4(packed, block_sw, tensor_scale, out, inn))

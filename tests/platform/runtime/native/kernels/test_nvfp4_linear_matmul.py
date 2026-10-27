from __future__ import annotations

from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F
import vendor.gpl.comfyui.ops as comfy_ops
from tests.fixtures.kernel_gemm import installed_gemm_verdicts
from vendor.gpl.comfyui.ops import _quantize_nvfp4_dynamic, dequantize_nvfp4
from src.platform.runtime.native.kernels import fp8_ops, nvfp4_linear
from src.platform.runtime.native.kernels.fp8_linear import _quantize_fp8_dynamic
from src.platform.runtime.native.kernels.nvfp4_linear import (
    _nvfp4_fast_path_reject_reason,
    _nvfp4_matmul_enabled,
    _nvfp4_scaled_mm_fast_path_ok,
    reset_nvfp4_fast_path_rejection_log,
)
from .._nvfp4_ref import F4_MAX as _F4_MAX_REF
from .._nvfp4_ref import F8_MAX as _F8_MAX_REF
from .._nvfp4_ref import quantize_nvfp4 as _ref_quantize_nvfp4


def test_nvfp4_matmul_disabled_by_default(monkeypatch):
    monkeypatch.setattr(comfy_ops, "RUNTIME_FLAGS", {})
    with installed_gemm_verdicts(nvfp4=True):
        assert _nvfp4_matmul_enabled() is False


def test_nvfp4_matmul_flag_off_never_uses_fast_path_even_if_probe_true(monkeypatch):
    monkeypatch.setitem(comfy_ops.RUNTIME_FLAGS, "native_nvfp4_matmul", False)
    with installed_gemm_verdicts(nvfp4=True):
        assert _nvfp4_matmul_enabled() is False


def test_nvfp4_matmul_flag_on_requires_probe(monkeypatch):
    monkeypatch.setitem(comfy_ops.RUNTIME_FLAGS, "native_nvfp4_matmul", True)
    with installed_gemm_verdicts(nvfp4=False):
        assert _nvfp4_matmul_enabled() is False
    with installed_gemm_verdicts(nvfp4=True):
        assert _nvfp4_matmul_enabled() is True


def test_nvfp4_scaled_mm_fast_path_ok_happy_case():
    assert _nvfp4_scaled_mm_fast_path_ok(
        lora_deltas=None, input_dtype=torch.bfloat16,
        input_is_cuda=True, weight_is_cuda=True, in_features=64, out_features=32,
    ) is True


def test_nvfp4_scaled_mm_fast_path_rejects_lora_deltas():
    assert _nvfp4_scaled_mm_fast_path_ok(
        lora_deltas=["fake-delta"], input_dtype=torch.bfloat16,
        input_is_cuda=True, weight_is_cuda=True, in_features=64, out_features=32,
    ) is False


def test_nvfp4_scaled_mm_fast_path_rejects_non_cuda_operands():
    assert _nvfp4_scaled_mm_fast_path_ok(
        lora_deltas=None, input_dtype=torch.bfloat16,
        input_is_cuda=False, weight_is_cuda=True, in_features=64, out_features=32,
    ) is False
    assert _nvfp4_scaled_mm_fast_path_ok(
        lora_deltas=None, input_dtype=torch.bfloat16,
        input_is_cuda=True, weight_is_cuda=False, in_features=64, out_features=32,
    ) is False


def test_nvfp4_scaled_mm_fast_path_rejects_fp32_input():
    assert _nvfp4_scaled_mm_fast_path_ok(
        lora_deltas=None, input_dtype=torch.float32,
        input_is_cuda=True, weight_is_cuda=True, in_features=64, out_features=32,
    ) is False


def test_nvfp4_scaled_mm_fast_path_requires_in_features_multiple_of_32():
    assert _nvfp4_scaled_mm_fast_path_ok(
        lora_deltas=None, input_dtype=torch.bfloat16,
        input_is_cuda=True, weight_is_cuda=True, in_features=48, out_features=32,
    ) is False
    assert _nvfp4_scaled_mm_fast_path_ok(
        lora_deltas=None, input_dtype=torch.bfloat16,
        input_is_cuda=True, weight_is_cuda=True, in_features=32, out_features=32,
    ) is True


def test_nvfp4_scaled_mm_fast_path_requires_out_features_multiple_of_16():
    assert _nvfp4_scaled_mm_fast_path_ok(
        lora_deltas=None, input_dtype=torch.bfloat16,
        input_is_cuda=True, weight_is_cuda=True, in_features=64, out_features=24,
    ) is False


def test_quantize_nvfp4_dynamic_matches_weight_quantizer_nibble_convention():
    torch.manual_seed(30)
    m, k = 8, 64
    x = torch.randn(m, k) * 0.3
    packed, block_sw, tensor_scale = _quantize_nvfp4_dynamic(x)

    ref_packed, ref_block_sw, _, _ = _ref_quantize_nvfp4(x, tensor_scale)
    assert torch.equal(packed, ref_packed)
    assert torch.equal(block_sw, ref_block_sw)


@pytest.mark.parametrize("quantize", [_quantize_fp8_dynamic, _quantize_nvfp4_dynamic], ids=["fp8", "nvfp4"])
def test_activation_quantisers_leave_an_fp32_input_untouched(quantize):
    x = torch.randn(8, 64) * 0.3
    before = x.clone()
    quantize(x)
    assert torch.equal(x, before)


def test_quantize_nvfp4_dynamic_output_dtypes_and_scale_shape():
    x = torch.randn(16, 64) * 0.2
    packed, block_sw, tensor_scale = _quantize_nvfp4_dynamic(x)
    assert packed.dtype == torch.uint8 and packed.shape == (16, 32)
    assert block_sw.dtype == torch.float8_e4m3fn
    assert tensor_scale.dtype == torch.float32 and tensor_scale.ndim == 0


@pytest.mark.parametrize("m,k", [(200, 64), (17, 32), (129, 512)])
def test_quantize_nvfp4_dynamic_blocked_scale_matches_scaled_mm_v2_size_contract(m, k):
    x = torch.randn(m, k) * 0.1
    packed, block_sw, _ = _quantize_nvfp4_dynamic(x)
    num_blocks = -(-k // 16)
    nrb = -(-m // 128)
    ncb = -(-num_blocks // 4)
    assert block_sw.numel() == nrb * 128 * ncb * 4


def test_quantize_nvfp4_dynamic_roundtrip_within_grid_tolerance():
    torch.manual_seed(31)
    m, k = 16, 128
    x = torch.randn(m, k) * 0.05
    packed, block_sw, tensor_scale = _quantize_nvfp4_dynamic(x)
    deq = dequantize_nvfp4(packed, block_sw, tensor_scale, m, k)
    rel = (deq - x).abs().mean() / x.abs().mean()
    assert 0.0 <= rel < 0.25


def test_quantize_nvfp4_dynamic_handles_all_zero_block_without_nan():
    x = torch.zeros(16, 32)
    x[0, :16] = torch.randn(16) * 0.1
    packed, block_sw, tensor_scale = _quantize_nvfp4_dynamic(x)
    deq = dequantize_nvfp4(packed, block_sw, tensor_scale, 16, 32)
    assert torch.isfinite(deq).all()
    assert torch.count_nonzero(deq[0, 16:]) == 0
    assert torch.count_nonzero(deq[1:, :]) == 0


def _nvfp4_layer(in_f=32, out_f=16, bias=False, seed=40):
    lin = fp8_ops.Linear(in_f, out_f, bias=bias)
    torch.manual_seed(seed)
    w = torch.randn(out_f, in_f) * 0.05
    pts = (w.abs().amax() / (_F4_MAX_REF * _F8_MAX_REF)).clamp(min=1e-8).detach()
    packed, block_sw, _, _ = _ref_quantize_nvfp4(w, pts)
    sd = {"weight": packed, "weight_scale": block_sw, "weight_scale_2": pts.clone()}
    if bias:
        sd["bias"] = torch.randn(out_f)
    lin._load_from_state_dict(sd, "", {}, True, [], [], [])
    return lin


def test_nvfp4_load_keeps_raw_scales_alongside_precomputed_dequant_scale():
    lin = _nvfp4_layer()
    assert lin.nvfp4_scale is not None
    assert lin.nvfp4_weight_block_scale is not None
    assert lin.nvfp4_weight_block_scale.dtype == torch.float8_e4m3fn
    assert lin.nvfp4_weight_global_scale is not None
    assert lin.nvfp4_weight_global_scale.dtype == torch.float32
    assert lin.nvfp4_weight_global_scale.numel() == 1


def test_forward_nvfp4_scaled_mm_calls_f_scaled_mm_with_expected_layout_and_dtypes():
    lin = _nvfp4_layer(in_f=32, out_f=16, bias=True)
    x = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    fake_out = torch.zeros(6, 16, dtype=torch.bfloat16)
    with patch("torch.nn.functional.scaled_mm", return_value=fake_out) as mock_scaled_mm:
        out = lin._forward_nvfp4_scaled_mm(x)

    assert out.shape == (2, 3, 16)
    mock_scaled_mm.assert_called_once()
    args, kwargs = mock_scaled_mm.call_args
    mat_a, mat_b, scale_a, recipe_a, scale_b, recipe_b = args[:6]

    assert mat_a.dtype == torch.float4_e2m1fn_x2 and mat_a.shape == (6, 16)
    assert mat_b.dtype == torch.float4_e2m1fn_x2 and mat_b.shape == (16, 16)
    assert mat_b.t().is_contiguous()

    assert scale_a[0].dtype == torch.float8_e4m3fn and scale_a[1].dtype == torch.float32
    assert scale_b[0].dtype == torch.float8_e4m3fn and scale_b[1].dtype == torch.float32
    assert recipe_a == [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
    assert recipe_b == [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
    assert kwargs["swizzle_a"] == [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
    assert kwargs["swizzle_b"] == [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]
    assert kwargs["output_dtype"] == torch.bfloat16


def test_forward_nvfp4_scaled_mm_adds_bias_on_host_after_gemm():
    lin = _nvfp4_layer(in_f=32, out_f=16, bias=True)
    x = torch.randn(1, 32, dtype=torch.bfloat16)
    fake_out = torch.ones(1, 16, dtype=torch.bfloat16)
    with patch("torch.nn.functional.scaled_mm", return_value=fake_out) as mock_scaled_mm:
        out = lin._forward_nvfp4_scaled_mm(x)
    _, kwargs = mock_scaled_mm.call_args
    assert "bias" not in kwargs or kwargs["bias"] is None
    assert torch.allclose(out, fake_out + lin.bias.to(torch.bfloat16))


def test_forward_nvfp4_scaled_mm_bails_on_kernel_runtime_error():
    lin = _nvfp4_layer()
    x = torch.randn(1, 32, dtype=torch.bfloat16)
    with patch("torch.nn.functional.scaled_mm", side_effect=RuntimeError("no compatible kernel image")):
        assert lin._forward_nvfp4_scaled_mm(x) is None


def test_forward_nvfp4_scaled_mm_bails_on_missing_weight_scales():
    lin = fp8_ops.Linear(32, 16, bias=False)
    x = torch.randn(1, 32, dtype=torch.bfloat16)
    with patch("torch.nn.functional.scaled_mm", side_effect=AssertionError("must not run")):
        assert lin._forward_nvfp4_scaled_mm(x) is None


def test_forward_nvfp4_scaled_mm_bails_on_mixed_devices():
    lin = _nvfp4_layer()
    lin.nvfp4_weight_block_scale = lin.nvfp4_weight_block_scale.to("meta")

    x = torch.randn(1, 32, dtype=torch.bfloat16)
    with patch("torch.nn.functional.scaled_mm", side_effect=AssertionError("must not run")):
        assert lin._forward_nvfp4_scaled_mm(x) is None


def test_forward_falls_back_to_dequant_when_nvfp4_fast_path_bails():
    lin = _nvfp4_layer(in_f=32, out_f=16)
    x = torch.randn(1, 32, dtype=torch.bfloat16)
    expected = lin(x)

    with patch.object(nvfp4_linear, "_nvfp4_matmul_enabled", return_value=True), \
         patch.object(nvfp4_linear, "_nvfp4_fast_path_reject_reason", return_value=None), \
         patch.object(lin, "_forward_nvfp4_scaled_mm", return_value=None) as attempted:
        out = lin.forward_comfy_cast_weights(x)
    attempted.assert_called_once()
    assert torch.equal(out, expected)


@pytest.fixture(autouse=True)
def _reset_nvfp4_fast_path_rejection_log():
    reset_nvfp4_fast_path_rejection_log()
    yield
    reset_nvfp4_fast_path_rejection_log()


def test_nvfp4_fast_path_reject_reason_agrees_with_ok_predicate():
    ok_kwargs = dict(
        lora_deltas=None, input_dtype=torch.bfloat16,
        input_is_cuda=True, weight_is_cuda=True, in_features=64, out_features=32,
    )
    assert _nvfp4_fast_path_reject_reason(**ok_kwargs) is None
    assert _nvfp4_scaled_mm_fast_path_ok(**ok_kwargs) is True

    bad_kwargs = dict(ok_kwargs, input_dtype=torch.float32)
    reason = _nvfp4_fast_path_reject_reason(**bad_kwargs)
    assert reason is not None and "float32" in reason
    assert _nvfp4_scaled_mm_fast_path_ok(**bad_kwargs) is False


def test_nvfp4_fast_path_reject_reason_distinguishes_weight_device_from_input_dtype():
    weight_not_cuda = _nvfp4_fast_path_reject_reason(
        lora_deltas=None, input_dtype=torch.bfloat16,
        input_is_cuda=True, weight_is_cuda=False, in_features=64, out_features=32,
    )
    wrong_dtype = _nvfp4_fast_path_reject_reason(
        lora_deltas=None, input_dtype=torch.float32,
        input_is_cuda=True, weight_is_cuda=True, in_features=64, out_features=32,
    )
    assert weight_not_cuda == "weight_not_cuda"
    assert wrong_dtype == "input_dtype=torch.float32"
    assert weight_not_cuda != wrong_dtype


def test_forward_comfy_cast_weights_logs_nvfp4_fast_path_rejection_reason(monkeypatch, caplog):
    lin = _nvfp4_layer(in_f=32, out_f=16)
    x_wrong_dtype = torch.randn(1, 32, dtype=torch.float32)

    with patch.object(nvfp4_linear, "_nvfp4_matmul_enabled", return_value=True), \
         patch("torch.nn.functional.scaled_mm", side_effect=AssertionError("must not run")), \
         caplog.at_level("WARNING"):
        lin.forward_comfy_cast_weights(x_wrong_dtype)

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("float32" in w for w in warnings)


def test_forward_comfy_cast_weights_logs_nvfp4_fast_path_rejection_only_once_per_reason(monkeypatch, caplog):
    lin = _nvfp4_layer(in_f=32, out_f=16)
    x_wrong_dtype = torch.randn(1, 32, dtype=torch.float32)

    with patch.object(nvfp4_linear, "_nvfp4_matmul_enabled", return_value=True), \
         patch("torch.nn.functional.scaled_mm", side_effect=AssertionError("must not run")), \
         caplog.at_level("WARNING"):
        for _ in range(5):
            lin.forward_comfy_cast_weights(x_wrong_dtype)

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1


def test_forward_comfy_cast_weights_does_not_log_when_fast_path_eligible(monkeypatch, caplog):
    lin = _nvfp4_layer(in_f=32, out_f=16)
    x = torch.randn(1, 32, dtype=torch.bfloat16)
    fake_out = torch.zeros(1, 16, dtype=torch.bfloat16)

    with patch.object(nvfp4_linear, "_nvfp4_matmul_enabled", return_value=True), \
         patch.object(nvfp4_linear, "_nvfp4_fast_path_reject_reason", return_value=None), \
         patch("torch.nn.functional.scaled_mm", return_value=fake_out), \
         caplog.at_level("WARNING"):
        lin.forward_comfy_cast_weights(x)

    assert not [r for r in caplog.records if r.levelname == "WARNING"]

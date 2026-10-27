from __future__ import annotations

from unittest.mock import patch
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import vendor.gpl.comfyui.ops as comfy_ops
from tests.fixtures.kernel_gemm import installed_gemm_verdicts
from tests.fixtures.kernel_ops import patch_ops_name
from src.platform.runtime.native.kernels import fp8_linear, fp8_ops
from src.platform.runtime.native.kernels.fp8_linear import (
    _fp8_matmul_enabled,
    _quantize_fp8_dynamic,
    _scaled_mm_fast_path_ok,
    _scaled_mm_fast_path_reject_reason,
    reset_scaled_mm_fast_path_rejection_log,
)


def test_fp8_scaled_linear_load_and_forward():
    lin = fp8_ops.Linear(4, 3, bias=False)
    real_w = torch.randn(3, 4)
    scale = torch.tensor(0.5)
    w_fp8 = (real_w / scale).to(torch.float8_e4m3fn)
    sd = {"weight": w_fp8, "weight_scale": scale, "input_scale": torch.tensor(1.0)}

    missing, unexpected = lin.load_state_dict(sd, strict=False, assign=True)
    assert missing == []
    assert unexpected == []
    assert lin.weight_scale is not None
    assert lin.input_scale is not None

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    out = lin(x)
    ref = F.linear(x, w_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16))
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out, ref)


def test_fp8_scaled_linear_legacy_scale_weight_spelling():
    lin = fp8_ops.Linear(4, 3, bias=False)
    real_w = torch.randn(3, 4)
    scale = torch.tensor(0.25)
    w_fp8 = (real_w / scale).to(torch.float8_e4m3fn)
    missing, unexpected = lin.load_state_dict(
        {"weight": w_fp8, "scale_weight": scale}, strict=False, assign=True,
    )
    assert missing == [] and unexpected == []
    assert lin.weight_scale is not None
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    ref = F.linear(x, w_fp8.to(torch.bfloat16) * scale.to(torch.bfloat16))
    assert torch.allclose(lin(x), ref)


def test_fp8_linear_falls_back_when_no_scale():
    lin = fp8_ops.Linear(4, 3, bias=False)
    lin.weight = nn.Parameter(torch.randn(3, 4, dtype=torch.bfloat16))
    x = torch.randn(1, 4, dtype=torch.bfloat16)
    out = lin(x)
    ref = F.linear(x, lin.weight)
    assert torch.allclose(out, ref)


def test_fp8_matmul_disabled_by_default(monkeypatch):
    monkeypatch.setattr(comfy_ops, "RUNTIME_FLAGS", {})
    with installed_gemm_verdicts(fp8=True):
        assert _fp8_matmul_enabled() is False


def test_fp8_matmul_flag_off_never_uses_fast_path_even_if_probe_true(monkeypatch):
    monkeypatch.setitem(comfy_ops.RUNTIME_FLAGS, "native_fp8_matmul", False)
    with installed_gemm_verdicts(fp8=True):
        assert _fp8_matmul_enabled() is False


def test_fp8_matmul_flag_on_requires_probe(monkeypatch):
    monkeypatch.setitem(comfy_ops.RUNTIME_FLAGS, "native_fp8_matmul", True)
    with installed_gemm_verdicts(fp8=False):
        assert _fp8_matmul_enabled() is False
    with installed_gemm_verdicts(fp8=True):
        assert _fp8_matmul_enabled() is True


def test_scaled_mm_fast_path_ok_happy_case():
    assert _scaled_mm_fast_path_ok(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=True, lora_deltas=None,
        input_dtype=torch.bfloat16, input_is_cuda=True, weight_is_cuda=True,
        in_features=32, out_features=32,
    ) is True


def test_scaled_mm_fast_path_rejects_lora_deltas():
    assert _scaled_mm_fast_path_ok(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=True, lora_deltas=["fake-delta"],
        input_dtype=torch.bfloat16, input_is_cuda=True, weight_is_cuda=True,
        in_features=32, out_features=32,
    ) is False


def test_scaled_mm_fast_path_rejects_cpu_input():
    assert _scaled_mm_fast_path_ok(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=True, lora_deltas=None,
        input_dtype=torch.bfloat16, input_is_cuda=False, weight_is_cuda=True,
        in_features=32, out_features=32,
    ) is False


def test_scaled_mm_fast_path_rejects_unaligned_shapes():
    assert _scaled_mm_fast_path_ok(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=True, lora_deltas=None,
        input_dtype=torch.bfloat16, input_is_cuda=True, weight_is_cuda=True,
        in_features=30, out_features=32,
    ) is False


def test_scaled_mm_fast_path_rejects_fp32_input():
    assert _scaled_mm_fast_path_ok(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=True, lora_deltas=None,
        input_dtype=torch.float32, input_is_cuda=True, weight_is_cuda=True,
        in_features=32, out_features=32,
    ) is False


def test_scaled_mm_fast_path_reject_reason_agrees_with_ok_predicate():
    ok_kwargs = dict(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=True, lora_deltas=None,
        input_dtype=torch.bfloat16, input_is_cuda=True, weight_is_cuda=True,
        in_features=32, out_features=32,
    )
    assert _scaled_mm_fast_path_reject_reason(**ok_kwargs) is None
    assert _scaled_mm_fast_path_ok(**ok_kwargs) is True

    bad_kwargs = dict(ok_kwargs, input_dtype=torch.float32)
    reason = _scaled_mm_fast_path_reject_reason(**bad_kwargs)
    assert reason is not None and "float32" in reason
    assert _scaled_mm_fast_path_ok(**bad_kwargs) is False


def test_scaled_mm_fast_path_reject_reason_distinguishes_weight_device_dtype_and_scale():
    base = dict(
        lora_deltas=None, input_dtype=torch.bfloat16,
        input_is_cuda=True, weight_is_cuda=True, in_features=32, out_features=32,
    )
    weight_not_cuda = _scaled_mm_fast_path_reject_reason(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=True,
        **dict(base, weight_is_cuda=False),
    )
    wrong_weight_dtype = _scaled_mm_fast_path_reject_reason(
        weight_dtype=torch.float16, has_weight_scale=True, **base,
    )
    no_scale = _scaled_mm_fast_path_reject_reason(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=False, **base,
    )
    wrong_input_dtype = _scaled_mm_fast_path_reject_reason(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=True,
        **dict(base, input_dtype=torch.float32),
    )
    assert weight_not_cuda == "weight_not_cuda"
    assert wrong_weight_dtype == "weight_dtype=torch.float16"
    assert no_scale == "no_weight_scale"
    assert wrong_input_dtype == "input_dtype=torch.float32"
    assert len({weight_not_cuda, wrong_weight_dtype, no_scale, wrong_input_dtype}) == 4


def test_scaled_mm_fast_path_reject_reason_allow_weight_staging_admits_cpu_weight():
    base = dict(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=True, lora_deltas=None,
        input_dtype=torch.bfloat16, input_is_cuda=True, weight_is_cuda=False,
        in_features=32, out_features=32,
    )
    assert _scaled_mm_fast_path_reject_reason(**base) == "weight_not_cuda"
    assert _scaled_mm_fast_path_reject_reason(**base, allow_weight_staging=True) is None
    assert _scaled_mm_fast_path_ok(**base) is False
    assert _scaled_mm_fast_path_ok(**base, allow_weight_staging=True) is True

    cpu_input = dict(base, input_is_cuda=False)
    assert _scaled_mm_fast_path_reject_reason(**cpu_input, allow_weight_staging=True) == "input_not_cuda"

    wrong_dtype = dict(base, weight_dtype=torch.float16, weight_is_cuda=True)
    assert _scaled_mm_fast_path_reject_reason(**wrong_dtype, allow_weight_staging=True) == "weight_dtype=torch.float16"


def test_quantize_fp8_dynamic_round_trips_within_e4m3_precision():
    x = torch.randn(4, 16) * 0.3
    x_fp8, scale = _quantize_fp8_dynamic(x)
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32 and scale.ndim == 0
    x_dq = x_fp8.to(torch.float32) * scale
    torch.testing.assert_close(x_dq, x, rtol=0.15, atol=0.05)


def test_forward_scaled_mm_calls_torch_scaled_mm_with_expected_layout_and_dtypes():
    lin = fp8_ops.Linear(32, 16, bias=True)
    real_w = torch.randn(16, 32) * 0.05
    w_scale = torch.tensor(0.01)
    w_fp8 = (real_w / w_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    bias = torch.randn(16)
    lin.load_state_dict(
        {"weight": w_fp8, "weight_scale": w_scale, "bias": bias}, strict=False, assign=True,
    )

    x = torch.randn(2, 3, 32, dtype=torch.bfloat16)
    fake_out = torch.zeros(6, 16, dtype=torch.bfloat16)
    with patch("torch._scaled_mm", return_value=fake_out) as mock_scaled_mm:
        out = lin._forward_scaled_mm(x)

    assert out.shape == (2, 3, 16)
    mock_scaled_mm.assert_called_once()
    args, kwargs = mock_scaled_mm.call_args
    a, b = args
    assert a.dtype == torch.float8_e4m3fn and a.shape == (6, 32)
    assert b.dtype == torch.float8_e4m3fn and b.shape == (32, 16)
    assert b.t().is_contiguous()
    assert kwargs["scale_a"].dtype == torch.float32
    assert kwargs["scale_b"].dtype == torch.float32
    assert kwargs["out_dtype"] == torch.bfloat16
    assert kwargs["bias"] is not None and kwargs["bias"].dtype == torch.bfloat16


def test_forward_scaled_mm_uses_static_input_scale_when_present():
    lin = fp8_ops.Linear(16, 16, bias=False)
    real_w = torch.randn(16, 16) * 0.05
    w_scale = torch.tensor(0.01)
    w_fp8 = (real_w / w_scale).to(torch.float8_e4m3fn)
    lin.load_state_dict(
        {"weight": w_fp8, "weight_scale": w_scale, "input_scale": torch.tensor(0.02)},
        strict=False, assign=True,
    )
    x = torch.randn(1, 16, dtype=torch.bfloat16)
    fake_out = torch.zeros(1, 16, dtype=torch.bfloat16)
    with patch("torch._scaled_mm", return_value=fake_out) as mock_scaled_mm:
        lin._forward_scaled_mm(x)
    _, kwargs = mock_scaled_mm.call_args
    assert torch.equal(kwargs["scale_a"], torch.tensor(0.02))


def test_scaled_mm_math_matches_existing_dequant_forward():
    lin = fp8_ops.Linear(16, 16, bias=True)
    real_w = torch.randn(16, 16) * 0.1
    w_scale = torch.tensor(0.02)
    w_fp8 = (real_w / w_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    bias = torch.randn(16)
    lin.load_state_dict(
        {"weight": w_fp8, "weight_scale": w_scale, "bias": bias}, strict=False, assign=True,
    )

    x = torch.randn(3, 16, dtype=torch.bfloat16)
    expected = lin(x)

    x2d = x.reshape(-1, 16)
    x_fp8, x_scale = _quantize_fp8_dynamic(x2d)
    x_dq = x_fp8.to(torch.float32) * x_scale
    w_dq = w_fp8.to(torch.float32) * w_scale
    simulated = F.linear(x_dq, w_dq, bias.to(torch.float32)).to(torch.bfloat16)

    torch.testing.assert_close(simulated, expected.to(torch.float32), rtol=0.1, atol=0.1, check_dtype=False)


def test_forward_comfy_cast_weights_falls_back_when_non_branchable_lora_deltas_present(monkeypatch):
    from src.platform.runtime.native.lora.key_mapping import LoraDelta

    monkeypatch.setitem(comfy_ops.RUNTIME_FLAGS, "native_fp8_matmul", True)
    with installed_gemm_verdicts(fp8=True):
        lin = fp8_ops.Linear(16, 16, bias=False)
        real_w = torch.randn(16, 16) * 0.05
        w_scale = torch.tensor(0.01)
        w_fp8 = (real_w / w_scale).to(torch.float8_e4m3fn)
        lin.load_state_dict({"weight": w_fp8, "weight_scale": w_scale}, strict=False, assign=True)
        lin.lora_deltas = [LoraDelta(down=torch.randn(4, 4), up=torch.randn(4, 4), alpha=4.0, scale=1.0, kron=True)]

        x = torch.randn(1, 16, dtype=torch.bfloat16)
        with patch("torch._scaled_mm", side_effect=AssertionError("must not be called")):
            out = lin(x)
        assert out.shape == (1, 16)


def test_forward_comfy_cast_weights_falls_back_on_cpu_even_when_gate_on(monkeypatch):
    monkeypatch.setitem(comfy_ops.RUNTIME_FLAGS, "native_fp8_matmul", True)
    with installed_gemm_verdicts(fp8=True):
        lin = fp8_ops.Linear(16, 16, bias=False)
        real_w = torch.randn(16, 16) * 0.05
        w_scale = torch.tensor(0.01)
        w_fp8 = (real_w / w_scale).to(torch.float8_e4m3fn)
        lin.load_state_dict({"weight": w_fp8, "weight_scale": w_scale}, strict=False, assign=True)

        x = torch.randn(1, 16, dtype=torch.bfloat16)
        with patch("torch._scaled_mm", side_effect=AssertionError("must not be called")):
            out = lin(x)
        ref = F.linear(x, w_fp8.to(torch.bfloat16) * w_scale.to(torch.bfloat16))
        assert torch.allclose(out, ref)


def _fp8_layer(w_scale, *, input_scale=None, bias=False, in_f=16, out_f=16):
    lin = fp8_ops.Linear(in_f, out_f, bias=bias)
    sd = {"weight": torch.zeros(out_f, in_f, dtype=torch.float8_e4m3fn), "weight_scale": w_scale}
    if input_scale is not None:
        sd["input_scale"] = input_scale
    if bias:
        sd["bias"] = torch.zeros(out_f)
    lin.load_state_dict(sd, strict=False, assign=True)
    return lin


def test_forward_scaled_mm_bails_on_nonscalar_weight_scale():
    lin = _fp8_layer(torch.full((16, 1), 0.01))
    x = torch.randn(1, 16, dtype=torch.bfloat16)
    with patch("torch._scaled_mm", side_effect=AssertionError("must not be called")):
        assert lin._forward_scaled_mm(x) is None


def test_forward_scaled_mm_bails_on_nonscalar_input_scale():
    lin = _fp8_layer(torch.tensor(0.01), input_scale=torch.tensor([0.02, 0.03]))
    x = torch.randn(1, 16, dtype=torch.bfloat16)
    with patch("torch._scaled_mm", side_effect=AssertionError("must not be called")):
        assert lin._forward_scaled_mm(x) is None


def test_forward_scaled_mm_bails_on_zero_or_nonfinite_scales():
    x = torch.randn(1, 16, dtype=torch.bfloat16)
    with patch("torch._scaled_mm", side_effect=AssertionError("must not be called")):
        assert _fp8_layer(torch.tensor(0.0))._forward_scaled_mm(x) is None
        assert _fp8_layer(torch.tensor(float("inf")))._forward_scaled_mm(x) is None
        assert _fp8_layer(torch.tensor(float("nan")))._forward_scaled_mm(x) is None
        assert _fp8_layer(torch.tensor(0.01), input_scale=torch.tensor(0.0))._forward_scaled_mm(x) is None
        assert _fp8_layer(torch.tensor(0.01), input_scale=torch.tensor(float("inf")))._forward_scaled_mm(x) is None


def test_forward_scaled_mm_bails_on_kernel_runtime_error():
    lin = _fp8_layer(torch.tensor(0.01))
    x = torch.randn(1, 16, dtype=torch.bfloat16)
    with patch("torch._scaled_mm", side_effect=RuntimeError("no compatible kernel image")):
        assert lin._forward_scaled_mm(x) is None


def test_forward_scaled_mm_moves_scales_and_bias_to_input_device():
    lin = _fp8_layer(torch.tensor(0.01), input_scale=torch.tensor(0.02), bias=True)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    seen = {}

    def _fake(a, b, *, scale_a, scale_b, out_dtype, bias=None):
        seen["dev"] = (a.device, scale_a.device, scale_b.device, None if bias is None else bias.device)
        return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype, device=a.device)

    with patch("torch._scaled_mm", _fake):
        out = lin._forward_scaled_mm(x)
    assert out is not None and out.shape == (2, 16)
    a_dev, sa_dev, sb_dev, bias_dev = seen["dev"]
    assert sa_dev == a_dev and sb_dev == a_dev and bias_dev == a_dev


def test_forward_falls_back_to_dequant_when_fast_path_bails():
    real_w = torch.randn(16, 16) * 0.05
    w_scale = torch.tensor(0.01)
    w_fp8 = (real_w / w_scale).to(torch.float8_e4m3fn)
    lin = fp8_ops.Linear(16, 16, bias=False)
    lin.load_state_dict({"weight": w_fp8, "weight_scale": w_scale}, strict=False, assign=True)
    x = torch.randn(1, 16, dtype=torch.bfloat16)

    with patch.object(fp8_linear, "_fp8_matmul_enabled", return_value=True), \
         patch.object(fp8_linear, "_scaled_mm_fast_path_reject_reason", return_value=None), \
         patch.object(lin, "_forward_scaled_mm", return_value=None) as attempted:
        out = lin.forward_comfy_cast_weights(x)
    attempted.assert_called_once()
    ref = F.linear(x, w_fp8.to(torch.bfloat16) * w_scale.to(torch.bfloat16))
    assert torch.allclose(out, ref)


def _streamed_fp8_layer(in_f: int = 16, out_f: int = 16) -> "fp8_ops.Linear":
    lin = fp8_ops.Linear(in_f, out_f, bias=False)
    real_w = torch.randn(out_f, in_f) * 0.05
    w_scale = torch.tensor(0.01)
    w_fp8 = (real_w / w_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    lin.load_state_dict({"weight": w_fp8, "weight_scale": w_scale}, strict=False, assign=True)
    lin.stream_non_blocking = True
    return lin


def _install_fake_cuda_residency(monkeypatch):
    resident_ids: set[int] = set()
    calls: list[tuple[int, torch.dtype, torch.device, bool]] = []

    def _fake_cast_to(tensor, dtype, device, *, non_blocking=False):
        if tensor is None:
            return None
        calls.append((id(tensor), dtype, device, non_blocking))
        clone = tensor.detach().to(dtype=dtype).clone()
        resident_ids.add(id(clone))
        return clone

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: id(self) in resident_ids))
    patch_ops_name(monkeypatch, "cast_to", _fake_cast_to)

    def stage_calls_for(tensor):
        return [c for c in calls if c[0] == id(tensor)]

    return resident_ids, stage_calls_for


def test_forward_stages_streamed_cpu_weight_on_demand_and_reaches_kernel(monkeypatch):
    lin = _streamed_fp8_layer()
    resident_ids, stage_calls_for = _install_fake_cuda_residency(monkeypatch)
    monkeypatch.setitem(comfy_ops.RUNTIME_FLAGS, "native_fp8_matmul", True)

    x = torch.randn(1, 16, dtype=torch.bfloat16)
    resident_ids.add(id(x))
    fake_out = torch.zeros(1, 16, dtype=torch.bfloat16)

    assert lin.weight.is_cuda is False

    with installed_gemm_verdicts(fp8=True), \
         patch("torch._scaled_mm", return_value=fake_out) as mock_scaled_mm:
        out = lin(x)

    mock_scaled_mm.assert_called_once()
    staged = stage_calls_for(lin.weight)
    assert len(staged) == 1
    _, dtype, device, non_blocking = staged[0]
    assert dtype == lin.weight.dtype and device == x.device and non_blocking is True
    assert torch.equal(out, fake_out)
    assert lin.weight.is_cuda is False


def test_forward_reuses_already_resident_weight_without_extra_copy(monkeypatch):
    lin = _streamed_fp8_layer()
    resident_ids, stage_calls_for = _install_fake_cuda_residency(monkeypatch)
    monkeypatch.setitem(comfy_ops.RUNTIME_FLAGS, "native_fp8_matmul", True)
    resident_ids.add(id(lin.weight))

    x = torch.randn(1, 16, dtype=torch.bfloat16)
    resident_ids.add(id(x))
    fake_out = torch.zeros(1, 16, dtype=torch.bfloat16)

    with installed_gemm_verdicts(fp8=True), \
         patch("torch._scaled_mm", return_value=fake_out) as mock_scaled_mm:
        out = lin(x)

    mock_scaled_mm.assert_called_once()
    assert stage_calls_for(lin.weight) == []
    assert torch.equal(out, fake_out)


def test_forward_restages_every_call_with_no_cross_call_retention(monkeypatch):
    lin = _streamed_fp8_layer()
    resident_ids, stage_calls_for = _install_fake_cuda_residency(monkeypatch)
    monkeypatch.setitem(comfy_ops.RUNTIME_FLAGS, "native_fp8_matmul", True)

    x = torch.randn(1, 16, dtype=torch.bfloat16)
    resident_ids.add(id(x))
    fake_out = torch.zeros(1, 16, dtype=torch.bfloat16)
    weight_id_before = id(lin.weight)

    with installed_gemm_verdicts(fp8=True), \
         patch("torch._scaled_mm", return_value=fake_out) as mock_scaled_mm:
        for _ in range(3):
            lin(x)

    assert mock_scaled_mm.call_count == 3
    assert len(stage_calls_for(lin.weight)) == 3
    assert id(lin.weight) == weight_id_before
    assert lin.weight.is_cuda is False


def test_forward_falls_back_to_dequant_when_on_demand_staging_oom(monkeypatch):
    from src.platform.runtime.native.kernels import cast_to as real_cast_to

    lin = _streamed_fp8_layer()
    resident_ids, _ = _install_fake_cuda_residency(monkeypatch)
    monkeypatch.setitem(comfy_ops.RUNTIME_FLAGS, "native_fp8_matmul", True)

    x = torch.randn(1, 16, dtype=torch.bfloat16)
    resident_ids.add(id(x))

    oom_raised = {"done": False}

    def _oom_once_for_weight(tensor, dtype, device, *, non_blocking=False):
        if tensor is lin.weight and not oom_raised["done"]:
            oom_raised["done"] = True
            raise torch.cuda.OutOfMemoryError("simulated staging OOM")
        return real_cast_to(tensor, dtype, device, non_blocking=non_blocking)

    patch_ops_name(monkeypatch, "cast_to", _oom_once_for_weight)

    with installed_gemm_verdicts(fp8=True), \
         patch("torch._scaled_mm", side_effect=AssertionError("must not run after staging OOM")):
        out = lin(x)

    assert oom_raised["done"] is True
    ref = F.linear(x, lin.weight.to(torch.bfloat16) * lin.weight_scale.to(torch.bfloat16))
    assert torch.allclose(out, ref)


def test_forward_on_cpu_gate_off_is_byte_identical_to_dequant(monkeypatch):
    lin = _streamed_fp8_layer()
    _install_fake_cuda_residency(monkeypatch)
    monkeypatch.setitem(comfy_ops.RUNTIME_FLAGS, "native_fp8_matmul", False)

    x = torch.randn(1, 16, dtype=torch.bfloat16)
    with patch("torch._scaled_mm", side_effect=AssertionError("must not run when gate is off")):
        out = lin(x)

    ref = F.linear(x, lin.weight.to(torch.bfloat16) * lin.weight_scale.to(torch.bfloat16))
    assert torch.allclose(out, ref)


@pytest.fixture(autouse=True)
def _reset_scaled_mm_fast_path_rejection_log():
    reset_scaled_mm_fast_path_rejection_log()
    yield
    reset_scaled_mm_fast_path_rejection_log()


def _loaded_fp8_layer(in_f=16, out_f=16):
    real_w = torch.randn(out_f, in_f) * 0.05
    w_scale = torch.tensor(0.01)
    w_fp8 = (real_w / w_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    lin = fp8_ops.Linear(in_f, out_f, bias=False)
    lin.load_state_dict({"weight": w_fp8, "weight_scale": w_scale}, strict=False, assign=True)
    return lin


def test_forward_comfy_cast_weights_logs_scaled_mm_fast_path_rejection_reason(caplog):
    lin = _loaded_fp8_layer()
    x_wrong_dtype = torch.randn(1, 16, dtype=torch.float32)

    with patch.object(fp8_linear, "_fp8_matmul_enabled", return_value=True), \
         patch("torch._scaled_mm", side_effect=AssertionError("must not run")), \
         caplog.at_level("WARNING"):
        lin.forward_comfy_cast_weights(x_wrong_dtype)

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("float32" in w for w in warnings)


def test_forward_comfy_cast_weights_logs_scaled_mm_fast_path_rejection_only_once_per_reason(caplog):
    lin = _loaded_fp8_layer()
    x_wrong_dtype = torch.randn(1, 16, dtype=torch.float32)

    with patch.object(fp8_linear, "_fp8_matmul_enabled", return_value=True), \
         patch("torch._scaled_mm", side_effect=AssertionError("must not run")), \
         caplog.at_level("WARNING"):
        for _ in range(5):
            lin.forward_comfy_cast_weights(x_wrong_dtype)

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1


def test_forward_comfy_cast_weights_does_not_log_when_fp8_fast_path_eligible(caplog):
    lin = _loaded_fp8_layer()
    x = torch.randn(1, 16, dtype=torch.bfloat16)
    fake_out = torch.zeros(1, 16, dtype=torch.bfloat16)

    with patch.object(fp8_linear, "_fp8_matmul_enabled", return_value=True), \
         patch.object(fp8_linear, "_scaled_mm_fast_path_reject_reason", return_value=None), \
         patch("torch._scaled_mm", return_value=(fake_out, torch.zeros(1, 16))), \
         patch.object(lin, "_forward_scaled_mm", return_value=fake_out), \
         caplog.at_level("WARNING"):
        lin.forward_comfy_cast_weights(x)

    assert not [r for r in caplog.records if r.levelname == "WARNING"]

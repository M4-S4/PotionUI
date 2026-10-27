from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.fixtures.kernel_gemm import install_gemm_verdicts
from tests.fixtures.kernel_ops import patch_ops_name
from vendor.gpl.comfyui.ops import dequantize_nvfp4
from src.platform.runtime.native.kernels import fp8_ops

from .._nvfp4_ref import F4_MAX as _F4_MAX
from .._nvfp4_ref import F8_MAX as _F8_MAX
from .._nvfp4_ref import quantize_nvfp4


def _nvfp4_sd(out_f: int, in_f: int, w: torch.Tensor, pre_quant_scale: torch.Tensor | None):
    pts = (w.abs().amax() / (_F4_MAX * _F8_MAX)).detach()
    packed, block_sw, _, _ = quantize_nvfp4(w, pts)
    sd = {
        "weight": packed,
        "weight_scale": block_sw,
        "weight_scale_2": pts.clone(),
        "comfy_quant": torch.zeros(5, dtype=torch.uint8),
    }
    if pre_quant_scale is not None:
        sd["pre_quant_scale"] = pre_quant_scale
    return sd, pts


def test_nvfp4_linear_consumes_pre_quant_scale_key():
    torch.manual_seed(0)
    out_f, in_f = 32, 64
    w = torch.randn(out_f, in_f) * 0.03
    pqs = torch.rand(in_f, dtype=torch.bfloat16) * 0.5 + 0.75
    sd, _ = _nvfp4_sd(out_f, in_f, w, pqs)

    lin = fp8_ops.Linear(in_f, out_f, bias=False)
    missing, unexpected = [], []
    lin._load_from_state_dict(dict(sd), "", {}, True, missing, unexpected, [])
    assert missing == [] and unexpected == []
    assert lin.pre_quant_scale is not None
    assert torch.equal(lin.pre_quant_scale, pqs)


def test_nvfp4_linear_applies_pre_quant_scale_to_the_activation():
    torch.manual_seed(1)
    out_f, in_f = 32, 64
    w = torch.randn(out_f, in_f) * 0.03
    pqs = torch.rand(in_f, dtype=torch.bfloat16) * 0.5 + 0.75
    sd, pts = _nvfp4_sd(out_f, in_f, w, pqs)
    packed, block_sw = sd["weight"], sd["weight_scale"]

    lin = fp8_ops.Linear(in_f, out_f, bias=False)
    lin._load_from_state_dict(dict(sd), "", {}, True, [], [], [])

    x = torch.randn(3, in_f, dtype=torch.bfloat16)
    dequant_w = dequantize_nvfp4(packed, block_sw, pts, out_f, in_f).to(torch.bfloat16)
    expected = torch.nn.functional.linear(x * pqs.to(x.dtype), dequant_w, None)
    got = lin(x)
    assert torch.allclose(got.float(), expected.float(), atol=1e-3, rtol=1e-3)


def test_nvfp4_linear_pre_quant_scale_is_load_bearing():
    torch.manual_seed(2)
    out_f, in_f = 32, 64
    w = torch.randn(out_f, in_f) * 0.03
    pqs = torch.rand(in_f, dtype=torch.bfloat16) * 0.5 + 0.75
    sd, pts = _nvfp4_sd(out_f, in_f, w, pqs)
    packed, block_sw = sd["weight"], sd["weight_scale"]

    lin = fp8_ops.Linear(in_f, out_f, bias=False)
    lin._load_from_state_dict(dict(sd), "", {}, True, [], [], [])

    x = torch.randn(3, in_f, dtype=torch.bfloat16)
    dequant_w = dequantize_nvfp4(packed, block_sw, pts, out_f, in_f).to(torch.bfloat16)
    unscaled = torch.nn.functional.linear(x, dequant_w, None)
    got = lin(x)
    assert not torch.allclose(got.float(), unscaled.float(), atol=1e-3, rtol=1e-3)


def test_nvfp4_linear_without_pre_quant_scale_key_is_unaffected():
    torch.manual_seed(3)
    out_f, in_f = 16, 32
    w = torch.randn(out_f, in_f) * 0.03
    sd, pts = _nvfp4_sd(out_f, in_f, w, None)
    packed, block_sw = sd["weight"], sd["weight_scale"]

    lin = fp8_ops.Linear(in_f, out_f, bias=False)
    lin._load_from_state_dict(dict(sd), "", {}, True, [], [], [])
    assert lin.pre_quant_scale is None

    x = torch.randn(3, in_f)
    dequant_w = dequantize_nvfp4(packed, block_sw, pts, out_f, in_f)
    expected = torch.nn.functional.linear(x, dequant_w, None)
    assert torch.allclose(lin(x), expected)


def test_fp8_scaled_linear_also_applies_pre_quant_scale():
    torch.manual_seed(4)
    out_f, in_f = 16, 32
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.03
    q = w.to(torch.float8_e4m3fn)
    scale = torch.tensor(1.0)
    pqs = torch.rand(in_f, dtype=torch.bfloat16) * 0.5 + 0.75

    lin = fp8_ops.Linear(in_f, out_f, bias=False)
    lin.comfy_cast_weights = True
    lin._load_from_state_dict(
        {"weight": q, "weight_scale": scale, "pre_quant_scale": pqs}, "", {}, True, [], [], [],
    )
    assert not lin._is_nvfp4
    assert lin.pre_quant_scale is not None

    x = torch.randn(2, in_f, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x * pqs.to(x.dtype), q.to(torch.bfloat16) * scale.to(torch.bfloat16), None)
    got = lin(x)
    assert torch.allclose(got.float(), expected.float(), atol=1e-2, rtol=1e-2)


def _fp8_scaled_leaf(out_f: int, in_f: int, pqs: torch.Tensor, *, seed: int):
    torch.manual_seed(seed)
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.03
    q = w.to(torch.float8_e4m3fn)
    scale = torch.tensor(0.02)
    lin = fp8_ops.Linear(in_f, out_f, bias=False)
    lin.comfy_cast_weights = True
    lin.load_state_dict(
        {"weight": q, "weight_scale": scale, "pre_quant_scale": pqs}, strict=False, assign=True,
    )
    assert not lin._is_nvfp4
    assert lin.weight.dtype == torch.float8_e4m3fn
    return lin, q, scale


def test_prepared_linear_dequant_branch_applies_pre_quant_scale_per_chunk():
    out_f, in_f = 16, 32
    pqs = torch.rand(in_f, dtype=torch.bfloat16) * 0.5 + 0.75
    lin, q, scale = _fp8_scaled_leaf(out_f, in_f, pqs, seed=10)

    x = torch.randn(5, in_f, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x * pqs.to(x.dtype), q.to(torch.bfloat16) * scale.to(torch.bfloat16), None)

    with lin.prepared_linear(x) as proj:
        got = torch.cat([proj(c) for c in x.split(2, dim=0)], dim=0)

    assert torch.allclose(got.float(), expected.float(), atol=1e-2, rtol=1e-2)

    unscaled = torch.nn.functional.linear(x, q.to(torch.bfloat16) * scale.to(torch.bfloat16), None)
    assert not torch.allclose(got.float(), unscaled.float(), atol=1e-6, rtol=0.05)


def _install_fake_cuda_residency(monkeypatch, *, streamed: list[torch.Tensor]):
    streamed_ids = {id(t) for t in streamed}

    def _fake_cast_to(tensor, dtype, device, *, non_blocking=False):
        if tensor is None:
            return None
        if id(tensor) not in streamed_ids:
            return tensor if tensor.device == device and tensor.dtype == dtype else tensor.to(device=device, dtype=dtype)
        return tensor.detach().to(dtype=dtype).clone()

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: id(self) not in streamed_ids))
    patch_ops_name(monkeypatch, "cast_to", _fake_cast_to)


def test_prepared_linear_fast_branch_applies_pre_quant_scale_per_chunk(monkeypatch):
    import vendor.gpl.comfyui.ops as wo

    out_f, in_f = 16, 32
    pqs = torch.rand(in_f, dtype=torch.bfloat16) * 0.5 + 0.75
    lin, q, scale = _fp8_scaled_leaf(out_f, in_f, pqs, seed=11)
    _install_fake_cuda_residency(monkeypatch, streamed=[lin.weight])
    monkeypatch.setitem(wo.RUNTIME_FLAGS, "native_fp8_matmul", True)

    x = torch.randn(5, in_f, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x * pqs.to(x.dtype), q.to(torch.bfloat16) * scale.to(torch.bfloat16), None)

    call_n = {"n": 0}

    def _computing_scaled_mm(a, b, *, scale_a, scale_b, out_dtype, bias=None):
        call_n["n"] += 1
        out = (a.to(torch.float32) * scale_a.to(torch.float32)) @ (b.to(torch.float32) * scale_b.to(torch.float32))
        return out.to(out_dtype)

    install_gemm_verdicts(monkeypatch, fp8=True)
    monkeypatch.setattr(torch, "_scaled_mm", _computing_scaled_mm)

    with lin.prepared_linear(x) as proj:
        got = torch.cat([proj(c) for c in x.split(2, dim=0)], dim=0)

    assert call_n["n"] == 3
    torch.testing.assert_close(got.float(), expected.float(), rtol=0.15, atol=0.1)

    lin.pre_quant_scale = None
    with lin.prepared_linear(x) as proj_unscaled:
        unscaled = torch.cat([proj_unscaled(c) for c in x.split(2, dim=0)], dim=0)
    assert not torch.allclose(got.float(), unscaled.float(), atol=1e-6, rtol=0.05)

    unscaled = torch.nn.functional.linear(x, q.to(torch.bfloat16) * scale.to(torch.bfloat16), None)
    assert not torch.allclose(got.float(), unscaled.float(), atol=1e-6, rtol=0.05)


def test_prepared_linear_pre_quant_scale_not_doubled_after_kernel_rejection(monkeypatch):
    import vendor.gpl.comfyui.ops as wo

    out_f, in_f = 16, 32
    pqs = torch.rand(in_f, dtype=torch.bfloat16) * 0.5 + 0.75
    lin, q, scale = _fp8_scaled_leaf(out_f, in_f, pqs, seed=12)
    _install_fake_cuda_residency(monkeypatch, streamed=[lin.weight])
    monkeypatch.setitem(wo.RUNTIME_FLAGS, "native_fp8_matmul", True)

    x = torch.randn(5, in_f, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(x * pqs.to(x.dtype), q.to(torch.bfloat16) * scale.to(torch.bfloat16), None)

    call_n = {"n": 0}

    def _flaky_scaled_mm(a, b, *, scale_a, scale_b, out_dtype, bias=None):
        call_n["n"] += 1
        if call_n["n"] == 1:
            raise RuntimeError("simulated kernel-level rejection")
        return (a.to(torch.float32) * scale_a.to(torch.float32)) @ (b.to(torch.float32) * scale_b.to(torch.float32))

    install_gemm_verdicts(monkeypatch, fp8=True)
    monkeypatch.setattr(torch, "_scaled_mm", _flaky_scaled_mm)

    with lin.prepared_linear(x) as proj:
        got = torch.cat([proj(c) for c in x.split(2, dim=0)], dim=0)

    assert call_n["n"] == 1
    torch.testing.assert_close(got.float(), expected.float(), rtol=1e-2, atol=1e-4)

    unscaled = torch.nn.functional.linear(x, q.to(torch.bfloat16) * scale.to(torch.bfloat16), None)
    assert not torch.allclose(got.float(), unscaled.float(), atol=1e-6, rtol=0.05)

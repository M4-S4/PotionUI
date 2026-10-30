from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.platform.runtime.native.io import quant_formats
from src.platform.runtime.native.io.quant_formats import dequantize_nvfp4
from src.platform.runtime.native.kernels import detect_quant_format, disable_weight_init, fp8_ops, pick_operations

from .._nvfp4_ref import F4_MAX as _F4_MAX
from .._nvfp4_ref import F8_MAX as _F8_MAX
from .._nvfp4_ref import old_dequantize_nvfp4
from .._nvfp4_ref import quantize_nvfp4


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

    loaded = _module_footprint_bytes(lin)
    assert loaded > 0

    lin.eval()
    x = torch.randn(3, inn)
    with torch.inference_mode():
        for _ in range(5):
            lin(x)
            assert _module_footprint_bytes(lin) == loaded + lin.dequant_table_bytes()
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
    old_weight = old_dequantize_nvfp4(packed, block_sw, pts, out_features, in_features)
    ref = torch.nn.functional.linear(x, old_weight, bias)
    assert torch.equal(lin(x), ref)


def test_only_the_first_dequant_forward_builds_the_scale_table(monkeypatch):
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
    old_weight = old_dequantize_nvfp4(packed, block_sw, pts, out_features, in_features)
    expected = torch.nn.functional.linear(x, old_weight)

    def _boom(*args, **kwargs):
        raise AssertionError("_inv_blocked_index must not run from the forward hot path")

    assert torch.equal(lin(x), expected)
    table = lin.nvfp4_scale
    assert table is not None and table.shape == (out_features, in_features // 16)
    monkeypatch.setattr(quant_formats, "_inv_blocked_index", _boom)
    y = lin(x)

    assert torch.equal(y, expected)
    assert lin.nvfp4_scale is table


def test_load_leaves_the_scale_table_unbuilt(monkeypatch):
    torch.manual_seed(18)
    out_features, in_features = 32, 32
    w = torch.randn(out_features, in_features) * 0.03
    pts = (w.abs().amax() / (_F4_MAX * _F8_MAX)).clamp(min=1e-8).detach()
    packed, block_sw, _, _ = quantize_nvfp4(w, pts)

    def _boom(*args, **kwargs):
        raise AssertionError("boom")

    lin = fp8_ops.Linear(in_features, out_features, bias=False)
    sd = {"weight": packed, "weight_scale": block_sw, "weight_scale_2": pts.clone()}
    monkeypatch.setattr(quant_formats, "_inv_blocked_index", _boom)
    lin._load_from_state_dict(sd, "", {}, True, [], [], [])
    assert lin._is_nvfp4 and lin.nvfp4_scale is None

from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.kernels import FastPathKind, QuantFormat, fast_path, gemm
from src.platform.runtime.native.lora import apply as apply_mod
from src.platform.runtime.native.lora.apply import _needs_runtime_deltas, apply_loras_with_report
from tests.fixtures.kernel_capabilities import blackwell, cpu_only, fake_capabilities
from tests.platform.runtime.native._nvfp4_ref import F4_MAX, F8_MAX, quantize_nvfp4
from tests.platform.runtime.native.lora.test_lora import _build, _kohya_lora
from vendor.gpl.comfyui import ops

NVFP4_KEY = "native_nvfp4_matmul"
IN, OUT = 64, 192

MACHINES = {
    "cpu-flag-off": (cpu_only, False, FastPathKind.DEQUANT),
    "cpu-flag-on": (cpu_only, True, FastPathKind.DEQUANT),
    "blackwell-flag-off": (lambda: fake_capabilities(blackwell()), False, FastPathKind.DEQUANT),
    "blackwell-flag-on": (lambda: fake_capabilities(blackwell()), True, FastPathKind.NVFP4_GEMM),
}


def _nvfp4_layer() -> torch.nn.Module:
    lin = ops.fp8_ops.Linear(IN, OUT, bias=False)
    weight = torch.randn(OUT, IN, generator=torch.Generator().manual_seed(5)) * 0.05
    scale_2 = (weight.abs().amax() / (F4_MAX * F8_MAX)).clamp(min=1e-8).detach()
    packed, block, _, _ = quantize_nvfp4(weight, scale_2)
    lin._load_from_state_dict(
        {"weight": packed, "weight_scale": block, "weight_scale_2": scale_2.clone()}, "", {}, True, [], [], []
    )
    return lin


@pytest.fixture
def machine(install_capabilities, monkeypatch):
    monkeypatch.setattr(gemm, "_nvfp4_trial_gemm", lambda device: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def configure(snapshot, nvfp4: bool) -> None:
        gemm._reset_verdicts()
        install_capabilities(snapshot())
        monkeypatch.setattr(ops, "RUNTIME_FLAGS", {NVFP4_KEY: nvfp4})

    yield configure
    gemm._reset_verdicts()


@pytest.mark.parametrize("snapshot,nvfp4,kind", MACHINES.values(), ids=MACHINES.keys())
def test_an_nvfp4_layer_takes_runtime_deltas_whatever_gemm_kind_the_machine_gives_it(
    machine, snapshot, nvfp4, kind
):
    machine(snapshot, nvfp4)
    model = _build(ops.pick_operations(torch.float8_e4m3fn, torch.bfloat16))
    block = model.double_blocks[0].img_attn
    block.qkv = _nvfp4_layer()

    _patched, _unmatched, [report] = apply_loras_with_report(model, [(_kohya_lora(), 1.0)])

    assert (report.runtime_params, report.inplace_params) == (1, 0)
    assert block.qkv.weight is None
    assert len(block.qkv.lora_deltas) == 1
    path = fast_path(block.qkv)
    assert path.quant_format is QuantFormat.NVFP4
    assert path.kind is kind


@pytest.mark.parametrize(
    "answer,marker,expected",
    [
        (QuantFormat.NVFP4, False, True),
        (None, True, False),
        (QuantFormat.FP8, True, False),
        (QuantFormat.INT8, True, False),
    ],
    ids=["nvfp4", "marker-without-format", "fp8-format", "int8-format"],
)
def test_the_nvfp4_routing_follows_the_kernels_format_answer_not_the_layer_marker(
    monkeypatch, answer, marker, expected
):
    layer = ops.manual_cast.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    layer._is_nvfp4 = marker
    asked = []

    def stub(candidate):
        asked.append(candidate)
        return answer

    monkeypatch.setattr(apply_mod, "quant_format", stub)

    assert _needs_runtime_deltas(layer) is expected
    assert asked == [layer]

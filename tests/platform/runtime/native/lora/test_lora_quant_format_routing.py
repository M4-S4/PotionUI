from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.kernels import QuantFormat, manual_cast, pick_operations
from src.platform.runtime.native.lora import apply as apply_mod
from src.platform.runtime.native.lora.apply import _needs_runtime_deltas, apply_loras_with_report
from tests.platform.runtime.native.kernels.test_fast_path import _nvfp4, machine
from tests.platform.runtime.native.lora.test_lora import _build, _kohya_lora


def test_an_nvfp4_layer_takes_runtime_deltas_on_a_machine_that_runs_the_nvfp4_gemm(machine, monkeypatch):
    machine("blackwell", fp8=True, nvfp4=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = _build(pick_operations(torch.float8_e4m3fn, torch.bfloat16))
    block = model.double_blocks[0].img_attn
    block.qkv = _nvfp4()

    _patched, _unmatched, [report] = apply_loras_with_report(model, [(_kohya_lora(out=16, inf=32), 1.0)])

    assert (report.runtime_params, report.inplace_params) == (1, 0)
    assert block.qkv.weight is None
    assert len(block.qkv.lora_deltas) == 1


@pytest.mark.parametrize(
    "answer,marker,expected",
    [(QuantFormat.NVFP4, False, True), (None, True, False), (QuantFormat.FP8, True, False)],
    ids=["nvfp4", "marker-without-format", "other-format"],
)
def test_the_nvfp4_routing_follows_the_kernels_format_answer_not_the_layer_marker(
    monkeypatch, answer, marker, expected
):
    layer = manual_cast.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    layer._is_nvfp4 = marker
    asked = []

    def stub(candidate):
        asked.append(candidate)
        return answer

    monkeypatch.setattr(apply_mod, "quant_format", stub)

    assert _needs_runtime_deltas(layer) is expected
    assert asked == [layer]

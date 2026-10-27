from __future__ import annotations

import torch

from src.platform.runtime.native.kernels import FastPathKind, fast_path, fp8_linear, linear, nvfp4_linear

from .test_fast_path import OUT_FEATURES, _fp8, _int8, _nvfp4, machine


def _record_every_hardware_ask(monkeypatch) -> list:
    asked = []
    for module in (linear, fp8_linear, nvfp4_linear):
        monkeypatch.setattr(module, "gemm_support", lambda device=None: asked.append(device))
    return asked


def test_no_kernels_module_asks_hardware_while_the_flag_is_off(machine, monkeypatch):
    asked = _record_every_hardware_ask(monkeypatch)
    machine("blackwell", fp8=False, nvfp4=False)

    fast_path(_fp8())
    fast_path(_nvfp4())

    assert asked == []


def test_no_kernels_module_asks_hardware_for_a_layer_that_can_never_take_the_kernel(machine, monkeypatch):
    asked = _record_every_hardware_ask(monkeypatch)
    machine("blackwell", fp8=True, nvfp4=True)

    fast_path(_int8(True))
    fast_path(_fp8(in_f=24))
    fast_path(_fp8(scale=torch.full((OUT_FEATURES, 1), 0.02)))
    fast_path(_nvfp4(in_f=48))

    assert asked == []


def test_the_reason_names_the_flag_or_the_hardware_gap_that_keeps_the_gemm_off(machine):
    machine("blackwell", fp8=False, nvfp4=False)
    assert fast_path(_fp8()).reason == "native_fp8_matmul is off"
    assert fast_path(_nvfp4()).reason == "native_nvfp4_matmul is off"

    machine("ampere", fp8=True, nvfp4=True)
    fp8, nvfp4 = fast_path(_fp8()), fast_path(_nvfp4())
    assert fp8.kind is nvfp4.kind is FastPathKind.DEQUANT
    assert "torch._scaled_mm" in fp8.reason
    assert "the nvfp4 GEMM" in nvfp4.reason

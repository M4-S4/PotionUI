from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.platform.runtime.native import kernels
from src.platform.runtime.native.kernels import QuantFormat, fast_path, gemm, linear, quant_format
from vendor.gpl.comfyui import ops

from .test_fast_path import PLAIN, QUANTIZED, machine

EXPECTED = {
    "fp8-scaled-linear": QuantFormat.FP8,
    "fp8-nvfp4-linear": QuantFormat.FP8,
    "fp8-e5m2": QuantFormat.FP8,
    "fp8-misaligned": QuantFormat.FP8,
    "fp8-per-channel-scale": QuantFormat.FP8,
    "fp8-per-channel-input-scale": QuantFormat.FP8,
    "fp8-static-input-scale": QuantFormat.FP8,
    "int8": QuantFormat.INT8,
    "int8-convrot": QuantFormat.INT8_CONVROT,
    "nvfp4": QuantFormat.NVFP4,
    "nvfp4-misaligned": QuantFormat.NVFP4,
    **dict.fromkeys(PLAIN),
}
BUILDERS = {**QUANTIZED, **PLAIN}
MACHINES = {"cpu-flags-off": ("cpu", False), "blackwell-flags-on": ("blackwell", True)}


def _forbidden(*args, **kwargs):
    raise AssertionError("quant_format reached for hardware or a runtime flag")


class _NoFlags(dict):
    get = __getitem__ = __contains__ = _forbidden


def test_every_layer_variant_of_the_fast_path_matrix_has_an_expected_format():
    assert set(EXPECTED) == set(BUILDERS)


@pytest.mark.parametrize("name", EXPECTED)
def test_the_format_comes_from_the_layer_state(name):
    assert quant_format(BUILDERS[name]()) is EXPECTED[name]


@pytest.mark.parametrize("name", EXPECTED)
def test_the_format_never_asks_for_hardware_or_runtime_flags(monkeypatch, name):
    layer = BUILDERS[name]()
    asked = []

    def spy(*args, **kwargs):
        asked.append(args)
        return _forbidden()

    monkeypatch.setattr(linear, "gemm_support", spy)
    monkeypatch.setattr(gemm, "capabilities", spy)
    monkeypatch.setattr(gemm, "_nvfp4_trial_gemm", spy)
    monkeypatch.setattr(ops, "GEMM_SUPPORT", spy)
    monkeypatch.setattr(ops, "RUNTIME_FLAGS", _NoFlags())

    assert quant_format(layer) is EXPECTED[name]
    assert asked == []


@pytest.mark.parametrize("hardware,flags", MACHINES.values(), ids=MACHINES.keys())
@pytest.mark.parametrize("name", EXPECTED)
def test_the_format_is_the_one_fast_path_reports_on_any_machine(machine, name, hardware, flags):
    machine(hardware, fp8=flags, nvfp4=flags)
    layer = BUILDERS[name]()

    path = fast_path(layer)

    assert quant_format(layer) is (None if path is None else path.quant_format)


def test_objects_that_are_not_ops_linears_have_no_format():
    conv = ops.manual_cast.Conv2d(3, 4, 1, dtype=torch.bfloat16)
    impostor = SimpleNamespace(weight_scale=torch.tensor(1.0), _is_nvfp4=True)

    assert quant_format(conv) is None
    assert quant_format(impostor) is None
    assert quant_format(None) is None


def test_the_format_is_a_public_kernel_export():
    assert "quant_format" in kernels.__all__

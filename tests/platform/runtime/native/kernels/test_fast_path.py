from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vendor.gpl.comfyui.ops as ops
from src.platform.runtime.native.kernels import (
    FastPath,
    FastPathKind,
    QuantFormat,
    fast_path,
    gemm,
    gemm_support,
    linear,
)
from src.platform.runtime.native.lora.key_mapping import LoraDelta
from tests.fixtures.kernel_capabilities import ada, ampere, blackwell, cpu_only, fake_capabilities

from .._nvfp4_ref import F4_MAX, F8_MAX, quantize_nvfp4

IN_FEATURES = 32
OUT_FEATURES = 16
FUSED_KEY = "native_lora_fused"
FP8_KEY = "native_fp8_matmul"
NVFP4_KEY = "native_nvfp4_matmul"

HARDWARE = {
    "cpu": cpu_only,
    "ampere": lambda: fake_capabilities(ampere()),
    "ada": lambda: fake_capabilities(ada()),
    "blackwell": lambda: fake_capabilities(blackwell()),
}


def _blob(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _loaded(layer: torch.nn.Module, state: dict) -> torch.nn.Module:
    layer.requires_grad_(False)
    layer.load_state_dict(state, strict=False, assign=True)
    return layer


def _fp8(cls=ops.Fp8ScaledLinear, dtype=torch.float8_e4m3fn, in_f=IN_FEATURES):
    torch.manual_seed(1)
    return _loaded(cls(in_f, OUT_FEATURES, bias=True), {
        "weight": torch.randn(OUT_FEATURES, in_f).to(dtype),
        "weight_scale": torch.tensor(0.02),
        "bias": torch.zeros(OUT_FEATURES),
        "comfy_quant": _blob({"format": "float8_e4m3fn"}),
    })


def _int8(convrot: bool):
    torch.manual_seed(2)
    conf = {"format": "int8_tensorwise"}
    scale = torch.tensor(0.05)
    if convrot:
        conf = {**conf, "convrot": True, "convrot_groupsize": 16}
        scale = torch.full((OUT_FEATURES, 1), 0.05)
    return _loaded(ops.fp8_ops.Linear(IN_FEATURES, OUT_FEATURES, bias=True), {
        "weight": torch.randint(-8, 8, (OUT_FEATURES, IN_FEATURES), dtype=torch.int8),
        "weight_scale": scale,
        "bias": torch.zeros(OUT_FEATURES),
        "comfy_quant": _blob(conf),
    })


def _nvfp4(in_f=IN_FEATURES):
    torch.manual_seed(3)
    weight = torch.randn(OUT_FEATURES, in_f) * 0.05
    tensor_scale = (weight.abs().amax() / (F4_MAX * F8_MAX)).clamp(min=1e-8).detach()
    packed, block, _, _ = quantize_nvfp4(weight, tensor_scale)
    return _loaded(ops.fp8_ops.Linear(in_f, OUT_FEATURES, bias=True), {
        "weight": packed,
        "weight_scale": block,
        "weight_scale_2": tensor_scale.clone(),
        "bias": torch.zeros(OUT_FEATURES),
    })


QUANTIZED = {
    "fp8-scaled-linear": _fp8,
    "fp8-nvfp4-linear": lambda: _fp8(ops.Nvfp4Linear),
    "fp8-e5m2": lambda: _fp8(dtype=torch.float8_e5m2),
    "fp8-misaligned": lambda: _fp8(in_f=24),
    "int8": lambda: _int8(False),
    "int8-convrot": lambda: _int8(True),
    "nvfp4": _nvfp4,
    "nvfp4-misaligned": lambda: _nvfp4(in_f=48),
}

PLAIN = {
    "torch-linear": lambda: torch.nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16),
    "disable-weight-init": lambda: ops.disable_weight_init.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16),
    "manual-cast": lambda: ops.manual_cast.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16),
    "fp8-without-scale": lambda: ops.Fp8ScaledLinear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16),
    "nvfp4-class-without-scale": lambda: ops.Nvfp4Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16),
    "conv": lambda: ops.manual_cast.Conv2d(3, 4, 1),
}


def _lora(kind: str, in_f: int):
    torch.manual_seed(4)
    if kind == "none":
        return None
    if kind == "plain":
        return [LoraDelta(down=torch.randn(2, in_f) * 0.1, up=torch.randn(OUT_FEATURES, 2) * 0.1, alpha=2.0)]
    return [
        LoraDelta(
            down=torch.randn(4, in_f // 4) * 0.1, up=torch.randn(4, 4) * 0.1, alpha=1.0, kron=True,
        )
    ]


class Recorder:
    def __init__(self) -> None:
        self.kinds: list[FastPathKind] = []
        self.fused = 0


@pytest.fixture
def recorder(monkeypatch):
    seen = Recorder()

    def scaled_mm(a, b, *, scale_a, scale_b, out_dtype, bias=None):
        seen.kinds.append(FastPathKind.FP8_SCALED_MM)
        return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype)

    def nvfp4_gemm(a, b, scale_a, recipe_a, scale_b, recipe_b, **kwargs):
        seen.kinds.append(FastPathKind.NVFP4_GEMM)
        return torch.zeros(a.shape[0], b.shape[1], dtype=kwargs["output_dtype"])

    real_prepare = ops.Fp8ScaledLinear._prepare_dequant_operand
    real_values = ops._dequant_values
    real_fused = ops._add_fused_lora_branch

    def prepare(self, *args, **kwargs):
        seen.kinds.append(FastPathKind.DEQUANT)
        return real_prepare(self, *args, **kwargs)

    def values(*args, **kwargs):
        seen.kinds.append(FastPathKind.DEQUANT)
        return real_values(*args, **kwargs)

    def fused(*args, **kwargs):
        seen.fused += 1
        return real_fused(*args, **kwargs)

    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)
    monkeypatch.setattr(F, "scaled_mm", nvfp4_gemm)
    monkeypatch.setattr(ops.Fp8ScaledLinear, "_prepare_dequant_operand", prepare)
    monkeypatch.setattr(ops, "_dequant_values", values)
    monkeypatch.setattr(ops, "_add_fused_lora_branch", fused)
    return seen


@pytest.fixture
def per_call_conditions_met(monkeypatch):
    for name in ("_scaled_mm_fast_path_reject_reason", "_nvfp4_fast_path_reject_reason"):
        real = getattr(ops, name)
        monkeypatch.setattr(
            ops, name,
            lambda real=real, **kwargs: real(**{**kwargs, "input_is_cuda": True, "weight_is_cuda": True}),
        )


@pytest.fixture(autouse=True)
def _fresh_verdicts():
    gemm._reset_verdicts()
    yield
    gemm._reset_verdicts()


@pytest.fixture
def machine(install_capabilities, monkeypatch):
    monkeypatch.setattr(gemm, "_nvfp4_trial_gemm", lambda device: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(ops, "GEMM_SUPPORT", gemm_support)

    def configure(hardware: str, *, fp8: bool, nvfp4: bool, fused: bool = True) -> None:
        gemm._reset_verdicts()
        install_capabilities(HARDWARE[hardware]())
        monkeypatch.setattr(ops, "RUNTIME_FLAGS", {FP8_KEY: fp8, NVFP4_KEY: nvfp4, FUSED_KEY: fused})

    return configure


def _forward(layer: torch.nn.Module, entry: str = "forward") -> torch.Tensor:
    x = torch.randn(2, 3, layer.in_features, dtype=torch.bfloat16)
    with torch.no_grad():
        if entry == "prepared_linear":
            with layer.prepared_linear(x) as projection:
                return projection(x)
        return layer(x)


def _assert_forward_took(layer: torch.nn.Module, expected: FastPath, seen: Recorder, entry: str) -> None:
    out = _forward(layer, entry)
    assert out.shape == (2, 3, OUT_FEATURES)
    assert set(seen.kinds) == {expected.kind}, expected
    assert (seen.fused > 0) is expected.lora_fused, expected
    assert (expected.reason is None) is (expected.kind is not FastPathKind.DEQUANT)


@pytest.mark.parametrize("lora", ["none", "plain", "lokr"])
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
@pytest.mark.parametrize("nvfp4", [True, False], ids=["nvfp4-on", "nvfp4-off"])
@pytest.mark.parametrize("fp8", [True, False], ids=["fp8-on", "fp8-off"])
@pytest.mark.parametrize("hardware", HARDWARE)
@pytest.mark.parametrize("name", QUANTIZED)
def test_fast_path_names_the_branch_the_real_forward_takes(
    machine, recorder, per_call_conditions_met, name, hardware, fp8, nvfp4, fused, lora
):
    machine(hardware, fp8=fp8, nvfp4=nvfp4, fused=fused)
    layer = QUANTIZED[name]()
    layer.lora_deltas = _lora(lora, layer.in_features)

    expected = fast_path(layer)

    assert expected is not None
    _assert_forward_took(layer, expected, recorder, "forward")


@pytest.mark.parametrize("lora", ["none", "plain", "lokr"])
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
@pytest.mark.parametrize("fp8", [True, False], ids=["fp8-on", "fp8-off"])
@pytest.mark.parametrize("hardware", HARDWARE)
@pytest.mark.parametrize("name", ["fp8-scaled-linear", "fp8-e5m2", "int8-convrot", "nvfp4"])
def test_fast_path_names_the_branch_prepared_linear_takes(
    machine, recorder, per_call_conditions_met, name, hardware, fp8, fused, lora
):
    machine(hardware, fp8=fp8, nvfp4=True, fused=fused)
    layer = QUANTIZED[name]()
    layer.lora_deltas = _lora(lora, layer.in_features)

    expected = fast_path(layer)

    assert expected is not None
    _assert_forward_took(layer, expected, recorder, "prepared_linear")


@pytest.mark.parametrize("name", PLAIN)
def test_layers_without_quantized_storage_have_no_fast_path_and_take_no_quantized_branch(
    machine, recorder, per_call_conditions_met, name
):
    machine("blackwell", fp8=True, nvfp4=True)
    layer = PLAIN[name]()

    assert fast_path(layer) is None
    if hasattr(layer, "in_features"):
        _forward(layer)
    assert recorder.kinds == []


def test_an_object_that_is_not_a_module_has_no_fast_path():
    assert fast_path(SimpleNamespace(weight_scale=torch.tensor(1.0), _is_nvfp4=True)) is None


@pytest.mark.parametrize(
    "name,quant_format",
    [
        ("fp8-scaled-linear", QuantFormat.FP8),
        ("fp8-nvfp4-linear", QuantFormat.FP8),
        ("fp8-e5m2", QuantFormat.FP8),
        ("int8", QuantFormat.INT8),
        ("int8-convrot", QuantFormat.INT8_CONVROT),
        ("nvfp4", QuantFormat.NVFP4),
    ],
)
def test_the_quant_format_comes_from_the_layer_state(machine, name, quant_format):
    machine("blackwell", fp8=True, nvfp4=True)
    assert fast_path(QUANTIZED[name]()).quant_format is quant_format


@pytest.mark.parametrize(
    "name,reason",
    [
        ("fp8-e5m2", "weight_dtype=torch.float8_e5m2"),
        ("int8", "weight_dtype=torch.int8"),
        ("int8-convrot", "weight_dtype=torch.int8"),
        ("fp8-misaligned", "in_features_not_multiple_of_16"),
        ("nvfp4-misaligned", "in_features_not_multiple_of_32"),
    ],
)
def test_a_layer_the_kernel_can_never_take_says_why_whatever_the_flags_and_hardware(machine, name, reason):
    machine("blackwell", fp8=True, nvfp4=True)
    result = fast_path(QUANTIZED[name]())
    assert result.kind is FastPathKind.DEQUANT
    assert result.reason == reason
    machine("cpu", fp8=False, nvfp4=False)
    assert fast_path(QUANTIZED[name]()).reason == reason


def test_a_lora_delta_with_no_output_side_form_sends_the_layer_to_dequant(machine):
    machine("blackwell", fp8=True, nvfp4=True)
    layer = _fp8()
    layer.lora_deltas = [
        LoraDelta(down=torch.randn(2, IN_FEATURES), up=torch.randn(8, 2), alpha=1.0, target_slice=(0, 12, 8)),
    ]
    result = fast_path(layer)
    assert result.kind is FastPathKind.DEQUANT
    assert result.reason == "lora_deltas"
    assert result.lora_fused is False


@pytest.mark.parametrize("name,flag", [("fp8-scaled-linear", FP8_KEY), ("nvfp4", NVFP4_KEY)])
def test_a_flag_that_is_off_is_the_reason_even_on_capable_hardware(machine, name, flag):
    machine("blackwell", fp8=flag != FP8_KEY, nvfp4=flag != NVFP4_KEY)
    result = fast_path(QUANTIZED[name]())
    assert result.kind is FastPathKind.DEQUANT
    assert result.reason == f"{flag} is off"


@pytest.mark.parametrize(
    "name,hardware,reason",
    [
        ("fp8-scaled-linear", "ampere", "Fake GPU sm86 is sm86, torch._scaled_mm needs sm89 or newer"),
        ("fp8-scaled-linear", "cpu", "no CUDA device is available"),
        ("nvfp4", "ada", "Fake GPU sm89 is sm89, the nvfp4 GEMM needs sm120 or newer"),
        ("nvfp4", "cpu", "no CUDA device is available"),
    ],
)
def test_unsupported_hardware_reports_the_gemm_support_reason(machine, name, hardware, reason):
    machine(hardware, fp8=True, nvfp4=True)
    result = fast_path(QUANTIZED[name]())
    assert result.kind is FastPathKind.DEQUANT
    assert result.reason == reason


@pytest.mark.parametrize(
    "name,hardware,kind",
    [
        ("fp8-scaled-linear", "ada", FastPathKind.FP8_SCALED_MM),
        ("fp8-scaled-linear", "blackwell", FastPathKind.FP8_SCALED_MM),
        ("nvfp4", "blackwell", FastPathKind.NVFP4_GEMM),
    ],
)
def test_a_capable_layer_with_its_flag_on_takes_the_gemm_kind_with_no_reason(machine, name, hardware, kind):
    machine(hardware, fp8=True, nvfp4=True)
    result = fast_path(QUANTIZED[name]())
    assert result == FastPath(kind=kind, quant_format=result.quant_format, lora_fused=False, reason=None)


def test_hardware_is_never_asked_while_the_flag_is_off_or_the_layer_can_never_take_the_kernel(machine, monkeypatch):
    asked = []
    monkeypatch.setattr(linear, "gemm_support", lambda device=None: asked.append(device))
    machine("blackwell", fp8=False, nvfp4=False)
    fast_path(_fp8())
    fast_path(_nvfp4())
    machine("blackwell", fp8=True, nvfp4=True)
    fast_path(_int8(True))
    fast_path(_fp8(in_f=24))
    fast_path(_nvfp4(in_f=48))
    assert asked == []


def test_the_live_flags_are_read_on_every_call(machine):
    layer = _fp8()
    machine("ada", fp8=False, nvfp4=False)
    assert fast_path(layer).kind is FastPathKind.DEQUANT
    machine("ada", fp8=True, nvfp4=False)
    assert fast_path(layer).kind is FastPathKind.FP8_SCALED_MM


class _OnSecondGpu(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda", 1)


def test_a_cuda_weight_asks_for_its_own_device_and_a_cpu_weight_for_the_current_one(install_capabilities, monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(ops, "RUNTIME_FLAGS", {FP8_KEY: True, NVFP4_KEY: True})
    install_capabilities(fake_capabilities(ampere(), ada()))

    streamed = _fp8()
    assert fast_path(streamed).kind is FastPathKind.DEQUANT

    resident = _fp8()
    resident._parameters["weight"] = resident.weight.detach().as_subclass(_OnSecondGpu)
    assert fast_path(resident).kind is FastPathKind.FP8_SCALED_MM


@pytest.mark.parametrize("name", ["fp8-scaled-linear", "nvfp4"])
def test_per_call_input_conditions_are_deliberately_not_part_of_the_answer(machine, recorder, name):
    machine("blackwell", fp8=True, nvfp4=True)
    layer = QUANTIZED[name]()

    expected = fast_path(layer)
    _forward(layer)

    assert expected.kind is not FastPathKind.DEQUANT
    assert set(recorder.kinds) == {FastPathKind.DEQUANT}

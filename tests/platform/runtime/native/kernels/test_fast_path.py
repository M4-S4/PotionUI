from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.platform.runtime.native.io import quant_formats
from src.platform.runtime.native.kernels import lora_math
from src.platform.runtime.native.kernels import (
    FastPath,
    FastPathKind,
    Fp8ScaledLinear,
    QuantFormat,
    disable_weight_init,
    fast_path,
    fp8_ops,
    nvfp4_path_summary,
    fp8_linear,
    gemm,
    linear,
    manual_cast,
    nvfp4_linear,
)
from src.platform.runtime.native.kernels.nvfp4_linear import Nvfp4Linear
from src.platform.runtime.native.lora.key_mapping import LoraDelta
from src.platform.settings.runtime_flags import runtime_flag_values
from tests.fixtures.kernel_capabilities import ada, ampere, blackwell, cpu_only, fake_capabilities
from tests.fixtures.kernel_ops import patch_ops_name

from .._nvfp4_ref import F4_MAX, F8_MAX, quantize_nvfp4
from .._quant_layouts import descriptor_blob

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
TWO_GPUS = {
    "ampere+ada": lambda: fake_capabilities(ampere(), ada()),
    "blackwell+ada": lambda: fake_capabilities(blackwell(), ada()),
}


def _loaded(layer: torch.nn.Module, state: dict) -> torch.nn.Module:
    layer.requires_grad_(False)
    layer.load_state_dict(state, strict=False, assign=True)
    return layer


def _fp8(cls=Fp8ScaledLinear, dtype=torch.float8_e4m3fn, in_f=IN_FEATURES, scale=None, input_scale=None):
    torch.manual_seed(1)
    state = {
        "weight": torch.randn(OUT_FEATURES, in_f).to(dtype),
        "weight_scale": torch.tensor(0.02) if scale is None else scale,
        "bias": torch.zeros(OUT_FEATURES),
        "comfy_quant": descriptor_blob({"format": "float8_e4m3fn"}),
    }
    if input_scale is not None:
        state["input_scale"] = input_scale
    return _loaded(cls(in_f, OUT_FEATURES, bias=True), state)


def _int8(convrot: bool):
    torch.manual_seed(2)
    conf = {"format": "int8_tensorwise"}
    scale = torch.tensor(0.05)
    if convrot:
        conf = {**conf, "convrot": True, "convrot_groupsize": 16}
        scale = torch.full((OUT_FEATURES, 1), 0.05)
    return _loaded(fp8_ops.Linear(IN_FEATURES, OUT_FEATURES, bias=True), {
        "weight": torch.randint(-8, 8, (OUT_FEATURES, IN_FEATURES), dtype=torch.int8),
        "weight_scale": scale,
        "bias": torch.zeros(OUT_FEATURES),
        "comfy_quant": descriptor_blob(conf),
    })


def _nvfp4(in_f=IN_FEATURES):
    torch.manual_seed(3)
    weight = torch.randn(OUT_FEATURES, in_f) * 0.05
    tensor_scale = (weight.abs().amax() / (F4_MAX * F8_MAX)).clamp(min=1e-8).detach()
    packed, block, _, _ = quantize_nvfp4(weight, tensor_scale)
    return _loaded(fp8_ops.Linear(in_f, OUT_FEATURES, bias=True), {
        "weight": packed,
        "weight_scale": block,
        "weight_scale_2": tensor_scale.clone(),
        "bias": torch.zeros(OUT_FEATURES),
    })


QUANTIZED = {
    "fp8-scaled-linear": _fp8,
    "fp8-nvfp4-linear": lambda: _fp8(Nvfp4Linear),
    "fp8-e5m2": lambda: _fp8(dtype=torch.float8_e5m2),
    "fp8-misaligned": lambda: _fp8(in_f=24),
    "fp8-per-channel-scale": lambda: _fp8(scale=torch.full((OUT_FEATURES, 1), 0.02)),
    "fp8-per-channel-input-scale": lambda: _fp8(input_scale=torch.full((2,), 0.5)),
    "fp8-static-input-scale": lambda: _fp8(input_scale=torch.tensor(0.5)),
    "int8": lambda: _int8(False),
    "int8-convrot": lambda: _int8(True),
    "nvfp4": _nvfp4,
    "nvfp4-misaligned": lambda: _nvfp4(in_f=48),
}

PLAIN = {
    "torch-linear": lambda: torch.nn.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16),
    "disable-weight-init": lambda: disable_weight_init.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16),
    "manual-cast": lambda: manual_cast.Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16),
    "fp8-without-scale": lambda: Fp8ScaledLinear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16),
    "nvfp4-class-without-scale": lambda: Nvfp4Linear(IN_FEATURES, OUT_FEATURES, dtype=torch.bfloat16),
}

LORA_SCENARIOS = [("none", True), ("plain", True), ("plain", False), ("lokr", True), ("lokr", False)]
FP8_GEMM_LAYERS = ["fp8-scaled-linear", "fp8-nvfp4-linear", "fp8-static-input-scale"]
NEVER_GEMM_LAYERS = [
    "fp8-e5m2",
    "fp8-misaligned",
    "fp8-per-channel-scale",
    "fp8-per-channel-input-scale",
    "int8",
    "int8-convrot",
    "nvfp4-misaligned",
]


def _case(name, hardware, fp8, nvfp4, resident, lora, fused):
    label = "-".join([
        name,
        hardware,
        "fp8" if fp8 else "nofp8",
        "nvfp4" if nvfp4 else "nonvfp4",
        "resident" if resident else "offgpu",
        lora,
        "fused" if fused else "unfused",
    ])
    return pytest.param(name, hardware, fp8, nvfp4, resident, lora, fused, id=label)


def _matrix(fp8_layers, nvfp4_layers, never_layers, residency):
    cases = []
    for lora, fused in LORA_SCENARIOS:
        for name in fp8_layers:
            for hardware in HARDWARE:
                for fp8 in (True, False):
                    cases.append(_case(name, hardware, fp8, False, True, lora, fused))
        for name in nvfp4_layers:
            for hardware in HARDWARE:
                for nvfp4 in (True, False):
                    for resident in residency:
                        cases.append(_case(name, hardware, True, nvfp4, resident, lora, fused))
        for name in never_layers:
            cases.append(_case(name, "blackwell", True, True, True, lora, fused))
    return cases


FORWARD_MATRIX = _matrix(FP8_GEMM_LAYERS, ["nvfp4"], NEVER_GEMM_LAYERS, (True, False))
PREPARED_MATRIX = _matrix(["fp8-scaled-linear"], ["nvfp4"], ["fp8-e5m2", "fp8-per-channel-scale", "int8-convrot"], (True,))


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

    real_prepare = Fp8ScaledLinear._prepare_dequant_operand
    real_values = quant_formats.dequant_values
    real_fused = lora_math._add_fused_lora_branch

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
    monkeypatch.setattr(Fp8ScaledLinear, "_prepare_dequant_operand", prepare)
    patch_ops_name(monkeypatch, "dequant_values", values)
    monkeypatch.setattr(lora_math, "_add_fused_lora_branch", fused)
    return seen


@pytest.fixture
def activation_on_cuda(monkeypatch):
    for module, name in (
        (fp8_linear, "_scaled_mm_fast_path_reject_reason"),
        (nvfp4_linear, "_nvfp4_fast_path_reject_reason"),
    ):
        real = getattr(module, name)
        monkeypatch.setattr(module, name, lambda real=real, **kwargs: real(**{**kwargs, "input_is_cuda": True}))


@pytest.fixture(autouse=True)
def _fresh_verdicts():
    gemm._reset_verdicts()
    yield
    gemm._reset_verdicts()


@pytest.fixture
def machine(install_capabilities, monkeypatch):
    monkeypatch.setattr(gemm, "_nvfp4_trial_gemm", lambda device: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def configure(hardware: str, *, fp8: bool, nvfp4: bool, fused: bool = True) -> None:
        gemm._reset_verdicts()
        install_capabilities({**HARDWARE, **TWO_GPUS}[hardware]())
        flags = runtime_flag_values()
        monkeypatch.setitem(flags, FP8_KEY, fp8)
        monkeypatch.setitem(flags, NVFP4_KEY, nvfp4)
        monkeypatch.setitem(flags, FUSED_KEY, fused)

    return configure


def _forward(layer: torch.nn.Module, entry: str = "forward") -> torch.Tensor:
    x = torch.randn(2, 3, layer.in_features, dtype=torch.bfloat16)
    with torch.no_grad():
        if entry == "prepared_linear":
            with layer.prepared_linear(x) as projection:
                return projection(x)
        return layer(x)


class _PackedOnGpu(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def _resident(layer: torch.nn.Module) -> torch.nn.Module:
    layer._buffers["nvfp4_packed"] = layer.nvfp4_packed.detach().as_subclass(_PackedOnGpu)
    return layer


def _assert_forward_took(
    layer: torch.nn.Module, expected: FastPath, seen: Recorder, entry: str, resident: bool = True
) -> None:
    out = _forward(layer, entry)
    took = FastPathKind.DEQUANT if expected.requires_resident_weight and not resident else expected.kind
    assert out.shape == (2, 3, OUT_FEATURES)
    assert set(seen.kinds) == {took}, expected
    assert (seen.fused > 0) is expected.lora_fused, expected


@pytest.mark.parametrize("name,hardware,fp8,nvfp4,resident,lora,fused", FORWARD_MATRIX)
def test_fast_path_names_the_branch_the_real_forward_takes(
    machine, recorder, activation_on_cuda, name, hardware, fp8, nvfp4, resident, lora, fused
):
    machine(hardware, fp8=fp8, nvfp4=nvfp4, fused=fused)
    layer = QUANTIZED[name]()
    if resident and getattr(layer, "_is_nvfp4", False):
        _resident(layer)
    layer.lora_deltas = _lora(lora, layer.in_features)

    expected = fast_path(layer)

    assert expected is not None
    _assert_forward_took(layer, expected, recorder, "forward", resident)


@pytest.mark.parametrize("name,hardware,fp8,nvfp4,resident,lora,fused", PREPARED_MATRIX)
def test_fast_path_names_the_branch_prepared_linear_takes(
    machine, recorder, activation_on_cuda, name, hardware, fp8, nvfp4, resident, lora, fused
):
    machine(hardware, fp8=fp8, nvfp4=nvfp4, fused=fused)
    layer = QUANTIZED[name]()
    if getattr(layer, "_is_nvfp4", False):
        _resident(layer)
    layer.lora_deltas = _lora(lora, layer.in_features)

    expected = fast_path(layer)

    assert expected is not None
    _assert_forward_took(layer, expected, recorder, "prepared_linear")


@pytest.mark.parametrize("name", PLAIN)
def test_layers_without_quantized_storage_have_no_fast_path_and_take_no_quantized_branch(
    machine, recorder, activation_on_cuda, name
):
    machine("blackwell", fp8=True, nvfp4=True)
    layer = PLAIN[name]()

    assert fast_path(layer) is None
    _forward(layer)
    assert recorder.kinds == []


def test_a_conv_layer_has_no_fast_path_and_runs_no_quantized_branch(machine, recorder, activation_on_cuda):
    machine("blackwell", fp8=True, nvfp4=True)
    conv = manual_cast.Conv2d(3, 4, 1, dtype=torch.bfloat16)

    assert fast_path(conv) is None
    out = conv(torch.randn(1, 3, 2, 2, dtype=torch.bfloat16))

    assert out.shape == (1, 4, 2, 2)
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
        ("fp8-per-channel-scale", "weight_scale_not_scalar"),
        ("fp8-per-channel-input-scale", "input_scale_not_scalar"),
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
    "name,hardware,kind,quant_format,resident",
    [
        ("fp8-scaled-linear", "ada", FastPathKind.FP8_SCALED_MM, QuantFormat.FP8, False),
        ("fp8-nvfp4-linear", "blackwell", FastPathKind.FP8_SCALED_MM, QuantFormat.FP8, False),
        ("nvfp4", "blackwell", FastPathKind.NVFP4_GEMM, QuantFormat.NVFP4, True),
    ],
)
def test_a_capable_layer_with_its_flag_on_takes_the_gemm_kind_with_no_reason(
    machine, name, hardware, kind, quant_format, resident
):
    machine(hardware, fp8=True, nvfp4=True)
    assert fast_path(QUANTIZED[name]()) == FastPath(
        kind=kind, quant_format=quant_format, requires_resident_weight=resident, lora_fused=False, reason=None
    )


@pytest.mark.parametrize("hardware,fp8,nvfp4", [("blackwell", True, True), ("ampere", True, True), ("blackwell", False, False)])
@pytest.mark.parametrize("name", QUANTIZED)
def test_a_reason_is_given_exactly_when_the_layer_dequantizes_and_only_the_nvfp4_gemm_needs_residency(
    machine, name, hardware, fp8, nvfp4
):
    machine(hardware, fp8=fp8, nvfp4=nvfp4)
    result = fast_path(QUANTIZED[name]())
    assert (result.reason is None) is (result.kind is not FastPathKind.DEQUANT)
    assert result.requires_resident_weight is (result.kind is FastPathKind.NVFP4_GEMM)


def test_hardware_is_never_asked_while_the_flag_is_off_or_the_layer_can_never_take_the_kernel(machine, monkeypatch):
    asked = []
    monkeypatch.setattr(linear, "gemm_support", lambda device=None: asked.append(device))
    machine("blackwell", fp8=False, nvfp4=False)
    fast_path(_fp8())
    fast_path(_nvfp4())
    machine("blackwell", fp8=True, nvfp4=True)
    fast_path(_int8(True))
    fast_path(_fp8(in_f=24))
    fast_path(_fp8(scale=torch.full((OUT_FEATURES, 1), 0.02)))
    fast_path(_nvfp4(in_f=48))
    assert asked == []


def test_the_live_flags_are_read_on_every_call(machine):
    layer = _fp8()
    machine("ada", fp8=False, nvfp4=False)
    assert fast_path(layer).kind is FastPathKind.DEQUANT
    machine("ada", fp8=True, nvfp4=False)
    assert fast_path(layer).kind is FastPathKind.FP8_SCALED_MM


def _on_gpu(index: int):
    class OnGpu(torch.Tensor):
        @property
        def is_cuda(self):
            return True

        @property
        def device(self):
            return torch.device("cuda", index)

    return OnGpu


def _fp8_weight_on(layer: torch.nn.Module, where: str) -> torch.nn.Module:
    if where != "cpu":
        layer._parameters["weight"] = layer.weight.detach().as_subclass(_on_gpu(int(where)))
    return layer


@pytest.mark.parametrize("lora", ["none", "plain"])
@pytest.mark.parametrize("current", [0, 1])
@pytest.mark.parametrize("where", ["cpu", "0", "1"])
def test_the_gemm_support_asked_is_the_current_devices_whatever_device_the_weight_is_on(
    machine, recorder, activation_on_cuda, monkeypatch, where, current, lora
):
    machine("ampere+ada", fp8=True, nvfp4=True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    layer = _fp8_weight_on(_fp8(), where)
    layer.lora_deltas = _lora(lora, layer.in_features)

    expected = fast_path(layer)

    assert expected.kind is (FastPathKind.FP8_SCALED_MM if current == 1 else FastPathKind.DEQUANT)
    _assert_forward_took(layer, expected, recorder, "forward")


@pytest.mark.parametrize("resident", [True, False], ids=["resident", "offgpu"])
@pytest.mark.parametrize("lora", ["none", "plain"])
@pytest.mark.parametrize("current", [0, 1])
def test_nvfp4_asks_the_current_devices_gemm_support_in_the_real_forward(
    machine, recorder, activation_on_cuda, monkeypatch, current, lora, resident
):
    machine("blackwell+ada", fp8=True, nvfp4=True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    layer = _nvfp4()
    if resident:
        _resident(layer)
    layer.lora_deltas = _lora(lora, layer.in_features)

    expected = fast_path(layer)

    assert expected.kind is (FastPathKind.NVFP4_GEMM if current == 0 else FastPathKind.DEQUANT)
    _assert_forward_took(layer, expected, recorder, "forward", resident)


def test_a_zero_weight_scale_is_a_value_condition_fast_path_documents_as_unseen(machine, recorder, activation_on_cuda):
    machine("ada", fp8=True, nvfp4=False)
    layer = _fp8(scale=torch.tensor(0.0))

    expected = fast_path(layer)
    _forward(layer)

    assert expected.kind is FastPathKind.FP8_SCALED_MM
    assert set(recorder.kinds) == {FastPathKind.DEQUANT}


@pytest.mark.parametrize("name", ["fp8-scaled-linear", "nvfp4"])
def test_per_call_input_conditions_are_deliberately_not_part_of_the_answer(machine, recorder, name):
    machine("blackwell", fp8=True, nvfp4=True)
    layer = QUANTIZED[name]()

    expected = fast_path(layer)
    _forward(layer)

    assert expected.kind is not FastPathKind.DEQUANT
    assert set(recorder.kinds) == {FastPathKind.DEQUANT}


def test_the_nvfp4_summary_counts_fast_path_dequant_reasons_and_other_linears(machine):
    model = torch.nn.Sequential(_nvfp4(), _nvfp4(), _nvfp4(in_f=48), _fp8(), torch.nn.Linear(4, 4))
    machine("blackwell", fp8=False, nvfp4=True)
    assert nvfp4_path_summary(model) == (
        "nvfp4 linears: 2 fast-path eligible, 1 dequant (in_features_not_multiple_of_32: 1), 2 non-nvfp4")
    machine("blackwell", fp8=False, nvfp4=False)
    assert nvfp4_path_summary(model) == (
        "nvfp4 linears: 0 fast-path eligible, 3 dequant "
        "(native_nvfp4_matmul is off: 2, in_features_not_multiple_of_32: 1), 2 non-nvfp4")


def test_a_model_without_nvfp4_linears_has_no_summary():
    assert nvfp4_path_summary(torch.nn.Sequential(_fp8(), torch.nn.Linear(4, 4))) is None
    assert nvfp4_path_summary(SimpleNamespace()) is None

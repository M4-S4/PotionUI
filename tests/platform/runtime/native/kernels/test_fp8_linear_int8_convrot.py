from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from src.platform.runtime.native.base import load_into_module
from src.platform.runtime.native.detect.registry import ModelSpec
from src.platform.runtime.native.io.state_dict_utils import weight_dtype
from src.platform.runtime.native.lora.apply import _needs_runtime_deltas
from src.platform.runtime.native.lora.key_mapping import LoraDelta
from src.platform.runtime.native.ops.dtype import is_mixed_precision
from src.platform.runtime.native.ops.fp8_quant import quantize_state_dict_to_fp8, should_quantize_fp8
from vendor.gpl.comfyui.ops import _build_convrot_hadamard, _extract_convrot_config, _parse_comfy_quant
from src.platform.runtime.native.kernels import QUANT_FP8_SCALED, detect_quant_format, fp8_ops, pick_operations
from src.platform.runtime.native.kernels.fp8_linear import _scaled_mm_fast_path_ok

from .._quant_layouts import (
    CONFIRMED_DESCRIPTORS,
    CONVROT_DEFAULT_GROUPSIZE,
    CONVROT_WILD_FORMS,
    HADAMARD_SIZES,
    INT8_CONVROT_LAYER,
    INT8_FORBIDDEN_KEYS,
    INT8_PLAIN_DESCRIPTOR,
    INT8_PLAIN_LAYER,
    convrot_descriptor,
    convrot_wild_blob,
    descriptor_blob,
    int8_state_dict,
    quantization_metadata_header,
    reference_hadamard,
    resolve_shape,
)

_STATED_GROUPSIZE = 64


def _linear(in_f: int, out_f: int, sd: dict, prefix: str = "proj.") -> nn.Module:
    lin = fp8_ops.Linear(in_f, out_f, bias=False)
    lin.comfy_cast_weights = True
    lin._load_from_state_dict(dict(sd), prefix, {}, True, [], [], [])
    return lin


class _OneLinear(nn.Module):

    def __init__(self, in_f: int, out_f: int) -> None:
        super().__init__()
        self.proj = fp8_ops.Linear(in_f, out_f, bias=False)
        self.proj.comfy_cast_weights = True

    def post_load(self) -> None:
        pass


def _spec() -> ModelSpec:
    return ModelSpec(family="t", variant="v", signature={}, model_class="x:Y")


def _rel_error(got: torch.Tensor, ref: torch.Tensor) -> float:
    return float((got.float() - ref.float()).abs().mean() / ref.float().abs().mean())


@pytest.mark.parametrize("conf", CONFIRMED_DESCRIPTORS)
def test_published_descriptors_round_trip(conf):
    blob = descriptor_blob(conf)
    assert blob.dtype == torch.uint8
    assert blob.numel() == len(json.dumps(conf).encode("utf-8"))
    assert _parse_comfy_quant(blob) == conf


@pytest.mark.parametrize("nested", [False, True])
def test_stated_convrot_groupsize_is_read_not_defaulted(nested):
    conf = convrot_descriptor(_STATED_GROUPSIZE, nested=nested)
    convrot, groupsize = _extract_convrot_config(conf, "proj")
    assert convrot is True
    assert groupsize == _STATED_GROUPSIZE


def test_convrot_without_stated_groupsize_uses_the_format_default(caplog):
    conf = convrot_descriptor(_STATED_GROUPSIZE, drop_groupsize=True)
    with caplog.at_level("INFO"):
        convrot, groupsize = _extract_convrot_config(conf, "proj")
    assert (convrot, groupsize) == (True, CONVROT_DEFAULT_GROUPSIZE)
    assert "no group size" in caplog.text or "states no group size" in caplog.text


@pytest.mark.parametrize("form", CONVROT_WILD_FORMS, ids=lambda f: f"{f[0]}b")
def test_wild_descriptor_variants_parse_at_their_published_byte_length(form):
    expected_bytes, _, _, extras = form
    blob = convrot_wild_blob(CONVROT_DEFAULT_GROUPSIZE, form)
    assert blob.numel() == expected_bytes
    conf = _parse_comfy_quant(blob)
    assert conf is not None
    convrot, groupsize = _extract_convrot_config(conf, "proj")
    assert convrot is True
    assert groupsize == CONVROT_DEFAULT_GROUPSIZE
    if extras.get("per_row"):
        assert conf["per_row"] is True


def test_plain_int8_descriptor_reports_no_convrot():
    assert _extract_convrot_config(dict(INT8_PLAIN_DESCRIPTOR), "proj")[0] is False


@pytest.mark.parametrize(
    "convrot,table",
    [(False, INT8_PLAIN_LAYER), (True, INT8_CONVROT_LAYER)],
)
def test_layer_layout_matches_declared_table(convrot, table):
    out_f, in_f = 64, 256
    sd, _, _, _ = int8_state_dict(out_f, in_f, convrot=convrot)
    assert set(sd) == {"proj." + name for name in table}
    for name, (dtype, shape) in table.items():
        t = sd["proj." + name]
        assert t.dtype == dtype, name
        expected = resolve_shape(shape, out_f=out_f, in_f=in_f, json_bytes=t.numel())
        assert tuple(t.shape) == expected, name


@pytest.mark.parametrize("convrot", [False, True])
def test_int8_layout_carries_none_of_the_other_formats_sidecars(convrot):
    sd, _, _, _ = int8_state_dict(64, 256, convrot=convrot)
    for forbidden in INT8_FORBIDDEN_KEYS:
        assert "proj." + forbidden not in sd


def test_scale_rank_is_the_on_the_wire_per_channel_signal():
    _, _, _, plain_scale = int8_state_dict(32, 64, convrot=False)
    _, _, _, convrot_scale = int8_state_dict(32, 64, convrot=True, groupsize=64)
    assert plain_scale.dim() == 0
    assert convrot_scale.dim() > 0
    assert convrot_scale.shape == (32, 1)


@pytest.mark.parametrize("size", HADAMARD_SIZES)
def test_hadamard_is_symmetric_and_involutory(size):
    h = _build_convrot_hadamard(size, device="cpu", dtype=torch.float32)
    assert torch.allclose(h, h.T)
    assert torch.allclose(h @ h, torch.eye(size), atol=1e-5)
    assert torch.allclose(h, reference_hadamard(size), atol=1e-6)


@pytest.mark.parametrize("convrot", [False, True])
def test_int8_weight_stays_int8_through_the_real_load(convrot):
    out_f, in_f = 16, 64
    sd, _, codes, _ = int8_state_dict(out_f, in_f, convrot=convrot, groupsize=64)

    m = _OneLinear(in_f, out_f)
    load_into_module(m, sd, _spec())

    assert m.proj.weight.dtype == torch.int8
    assert m.proj.weight.element_size() == 1
    assert m.proj.weight.numel() * m.proj.weight.element_size() == codes.numel()
    assert torch.equal(m.proj.weight.to(torch.int8), codes)


def test_int8_forward_after_real_load_recovers_original_weight():
    out_f, in_f = 16, 64
    sd, w, _, _ = int8_state_dict(out_f, in_f, convrot=True, groupsize=64)
    m = _OneLinear(in_f, out_f)
    load_into_module(m, sd, _spec())

    x = torch.randn(4, in_f)
    assert _rel_error(m.proj(x), torch.nn.functional.linear(x, w)) < 0.05


def test_plain_int8_dequantizes_to_original_weight():
    out_f, in_f = 64, 128
    sd, w, codes, scale = int8_state_dict(out_f, in_f)
    lin = _linear(in_f, out_f, sd)

    x = torch.randn(4, in_f)
    assert _rel_error(lin(x), torch.nn.functional.linear(x, w)) < 0.05
    assert torch.allclose(
        lin(x).float(),
        torch.nn.functional.linear(x, codes.to(torch.float32) * scale).float(),
        atol=1e-3, rtol=1e-3,
    )


def test_signed_codes_survive_the_dequant_cast():
    codes = torch.tensor([[-127, 127, -1, 1]], dtype=torch.int8)
    scale = torch.tensor(0.01, dtype=torch.float32)
    lin = _linear(4, 1, {"proj.weight": codes, "proj.weight_scale": scale,
                         "proj.comfy_quant": descriptor_blob(dict(INT8_PLAIN_DESCRIPTOR))})
    got = lin(torch.eye(4))
    assert torch.allclose(got.flatten().float(), torch.tensor([-1.27, 1.27, -0.01, 0.01]), atol=1e-5)


def test_convrot_dequantizes_back_to_the_unrotated_basis():
    out_f, in_f = 64, 256
    sd, w, codes, scale = int8_state_dict(out_f, in_f, convrot=True, groupsize=256)
    lin = _linear(in_f, out_f, sd)

    assert lin.convrot_groupsize == 256
    x = torch.randn(4, in_f)
    assert _rel_error(lin(x), torch.nn.functional.linear(x, w)) < 0.05
    naive = torch.nn.functional.linear(x, codes.to(torch.float32) * scale)
    assert _rel_error(naive, torch.nn.functional.linear(x, w)) > 0.5


@pytest.mark.parametrize("groupsize", [4, 16, 64, 256])
def test_convrot_at_every_supported_groupsize(groupsize):
    out_f, in_f = 8, groupsize * 2
    sd, w, _, _ = int8_state_dict(out_f, in_f, convrot=True, groupsize=groupsize)
    lin = _linear(in_f, out_f, sd)

    x = torch.randn(4, in_f)
    assert _rel_error(lin(x), torch.nn.functional.linear(x, w)) < 0.05


@pytest.mark.parametrize("nested", [False, True])
def test_convrot_dequantizes_with_the_groupsize_nested_or_top_level(nested):
    out_f, in_f = 16, 256
    sd, w, _, _ = int8_state_dict(
        out_f, in_f, convrot=True, groupsize=_STATED_GROUPSIZE,
        descriptor=convrot_descriptor(_STATED_GROUPSIZE, nested=nested),
    )
    lin = _linear(in_f, out_f, sd)

    assert lin.convrot_groupsize == _STATED_GROUPSIZE
    x = torch.randn(4, in_f)
    assert _rel_error(lin(x), torch.nn.functional.linear(x, w)) < 0.05


@pytest.mark.parametrize("form", CONVROT_WILD_FORMS, ids=lambda f: f"{f[0]}b")
def test_convrot_dequantizes_from_every_wild_descriptor_variant(form):
    out_f, in_f = 16, 512
    sd, w, _, _ = int8_state_dict(
        out_f, in_f, convrot=True, groupsize=CONVROT_DEFAULT_GROUPSIZE,
        blob=convrot_wild_blob(CONVROT_DEFAULT_GROUPSIZE, form),
    )
    lin = _linear(in_f, out_f, sd)

    assert lin.convrot_groupsize == CONVROT_DEFAULT_GROUPSIZE
    x = torch.randn(4, in_f)
    assert _rel_error(lin(x), torch.nn.functional.linear(x, w)) < 0.05


def test_bias_stays_unquantized_and_is_applied():
    out_f, in_f = 16, 64
    sd, w, _, _ = int8_state_dict(
        out_f, in_f, convrot=True, groupsize=_STATED_GROUPSIZE, bias_dtype=torch.float32,
    )
    bias = sd["proj.bias"]
    assert bias.dtype == torch.float32

    lin = fp8_ops.Linear(in_f, out_f, bias=True)
    lin.comfy_cast_weights = True
    lin._load_from_state_dict(dict(sd), "proj.", {}, True, [], [], [])

    x = torch.randn(4, in_f)
    assert _rel_error(lin(x), torch.nn.functional.linear(x, w, bias)) < 0.05


def test_convrot_dequantizes_in_bf16_compute():
    out_f, in_f = 32, 256
    sd, w, _, _ = int8_state_dict(out_f, in_f, convrot=True, groupsize=256)
    lin = _linear(in_f, out_f, sd)

    x = torch.randn(4, in_f, dtype=torch.bfloat16)
    got = lin(x)
    assert got.dtype == torch.bfloat16
    assert _rel_error(got, torch.nn.functional.linear(x.float(), w)) < 0.05


def test_hadamard_rejects_non_power_of_4_groupsize():
    _build_convrot_hadamard(64, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="power of 4"):
        _build_convrot_hadamard(128, device="cpu", dtype=torch.float32)


def test_groupsize_that_does_not_divide_in_features_raises():
    out_f, in_f = 8, 48
    sd = {
        "proj.weight": torch.zeros(out_f, in_f, dtype=torch.int8),
        "proj.weight_scale": torch.ones(out_f, 1),
        "proj.comfy_quant": descriptor_blob(convrot_descriptor(CONVROT_DEFAULT_GROUPSIZE)),
    }
    lin = _linear(in_f, out_f, sd)
    with pytest.raises(ValueError, match="not divisible"):
        lin(torch.randn(2, in_f))


def test_detect_quant_format_recognises_an_int8_checkpoint_with_no_file_header():
    sd, _, _, _ = int8_state_dict(16, 64, convrot=True, groupsize=_STATED_GROUPSIZE)
    assert detect_quant_format({}, sd) == QUANT_FP8_SCALED
    assert detect_quant_format(
        {}, {"proj.comfy_quant": descriptor_blob(INT8_PLAIN_DESCRIPTOR)},
    ) == QUANT_FP8_SCALED


def test_detect_quant_format_recognises_an_int8_checkpoint_with_a_file_header():
    sd, _, _, _ = int8_state_dict(16, 64, convrot=True, groupsize=_STATED_GROUPSIZE)
    metadata = quantization_metadata_header(
        ["proj"], convrot_descriptor(_STATED_GROUPSIZE),
    )
    assert "_quantization_metadata" in metadata
    assert json.loads(metadata["_quantization_metadata"])["format_version"] == "1.0"
    assert detect_quant_format(metadata, sd) == QUANT_FP8_SCALED


def test_int8_codes_without_a_descriptor_are_refused_when_rotation_is_ambiguous():
    out_f, in_f = 16, 64
    sd = {
        "proj.weight": torch.zeros(out_f, in_f, dtype=torch.int8),
        "proj.weight_scale": torch.ones(out_f, 1),
    }
    lin = fp8_ops.Linear(in_f, out_f, bias=False)
    lin.comfy_cast_weights = True
    with pytest.raises(ValueError, match="indistinguishable") as excinfo:
        lin._load_from_state_dict(dict(sd), "proj.", {}, True, [], [], [])
    assert "proj" in str(excinfo.value)


def test_plain_int8_codes_without_a_descriptor_still_load():
    out_f, in_f = 16, 64
    sd, w, _, _ = int8_state_dict(out_f, in_f)
    del sd["proj.comfy_quant"]
    lin = _linear(in_f, out_f, sd)

    x = torch.randn(4, in_f)
    assert _rel_error(lin(x), torch.nn.functional.linear(x, w)) < 0.05


def test_int8_checkpoint_selects_the_dequantising_ops_namespace():
    sd, _, _, _ = int8_state_dict(16, 64, convrot=True, groupsize=64)
    quant = detect_quant_format({}, sd)
    assert weight_dtype(sd) is torch.float32
    assert is_mixed_precision(sd) is False
    assert pick_operations(torch.bfloat16, torch.bfloat16, quant) is fp8_ops


def test_int8_checkpoint_is_not_requantized_to_fp8():
    sd, _, _, _ = int8_state_dict(16, 64, convrot=True, groupsize=64)
    quant = detect_quant_format({}, sd)
    assert should_quantize_fp8(
        "force", quant_format=quant, sd_dtype=torch.float32,
        bf16_gb=1.0, fp8_gb=0.5, vram_gb=8.0,
    ) is False
    out, n = quantize_state_dict_to_fp8(sd)
    assert n == 0
    assert out["proj.weight"].dtype == torch.int8


def test_int8_codes_are_never_patched_in_place_by_lora():
    out_f, in_f = 16, 64
    sd, _, _, _ = int8_state_dict(out_f, in_f)
    m = _OneLinear(in_f, out_f)
    load_into_module(m, sd, _spec())
    assert _needs_runtime_deltas(m.proj) is True


def test_convrot_layer_is_never_patched_in_place_whatever_its_storage_width():
    out_f, in_f = 16, 64
    sd, _, _, _ = int8_state_dict(out_f, in_f, convrot=True, groupsize=64)
    lin = _linear(in_f, out_f, sd)
    assert lin.weight.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert _needs_runtime_deltas(lin) is True


def test_lora_delta_composes_in_the_original_weight_basis():
    out_f, in_f, rank = 16, 64, 4
    sd, w, _, _ = int8_state_dict(out_f, in_f, convrot=True, groupsize=64)
    lin = _linear(in_f, out_f, sd)

    torch.manual_seed(8)
    up = torch.randn(out_f, rank) * 0.1
    down = torch.randn(rank, in_f) * 0.1
    lin.lora_deltas = [LoraDelta(down=down, up=up, alpha=1.0, scale=1.0)]

    x = torch.randn(4, in_f)
    delta = (up.float() @ down.float()) * (1.0 / rank)
    assert _rel_error(lin(x), torch.nn.functional.linear(x, w + delta)) < 0.05


def test_fp8_gemm_fast_path_refuses_int8_codes():
    assert _scaled_mm_fast_path_ok(
        weight_dtype=torch.int8, has_weight_scale=True, lora_deltas=None,
        input_dtype=torch.bfloat16, input_is_cuda=True, weight_is_cuda=True,
        in_features=64, out_features=64,
    ) is False


def test_releasing_one_convrot_layer_leaves_the_shared_hadamard_intact():
    from src.platform.runtime.native.base import release_module_storage

    sd, _, _, _ = int8_state_dict(64, 256, convrot=True)
    fresh = lambda: {k: v.clone() for k, v in sd.items()}
    first = _linear(256, 64, fresh())
    second = _linear(256, 64, fresh())
    x = torch.randn(2, 256)
    expected = second(x)

    release_module_storage(first)

    assert second.convrot_hadamard.numel() == 256 * 256
    assert _build_convrot_hadamard(256, device="cpu", dtype=torch.float32).numel() == 256 * 256
    assert torch.equal(second(x), expected)
    third = _linear(256, 64, fresh())
    assert torch.equal(third(x), expected)

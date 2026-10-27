from __future__ import annotations

import json

import pytest
import torch

from src.platform.runtime.native.errors import NativeEngineUnsupportedError
from src.platform.runtime.native.io.safetensors_loader import expand_int4_layers, unpack_int4
from src.platform.runtime.native.lora.apply import _needs_runtime_deltas
from vendor.gpl.comfyui.ops import _parse_comfy_quant
from src.platform.runtime.native.kernels import detect_quant_format, fp8_ops

from ._quant_layouts import reference_hadamard, reference_rotate

STATED_GROUPSIZE = 64
INT4_CODE_MAX = 7
PREFIX = "proj."


def blob(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def reference_pack(codes: torch.Tensor) -> torch.Tensor:
    low = codes[:, 0::2].to(torch.int32) & 0x0F
    high = codes[:, 1::2].to(torch.int32) & 0x0F
    return (low | (high << 4)).to(torch.uint8).view(torch.int8)


def int4_layer(out_f: int, in_f: int, groupsize: int, *, state_groupsize: bool = True, seed: int = 0):
    weight = torch.randn(out_f, in_f, generator=torch.Generator().manual_seed(seed)) * 0.5
    rotated = reference_rotate(weight, reference_hadamard(groupsize), groupsize)
    scale = rotated.abs().amax(dim=1) / INT4_CODE_MAX
    codes = (rotated / scale[:, None]).round().clamp(-INT4_CODE_MAX, INT4_CODE_MAX).to(torch.int8)
    conf = {"format": "convrot_w4a4", "convrot_groupsize": groupsize} if state_groupsize else {"format": "convrot_w4a4"}
    sd = {PREFIX + "weight": reference_pack(codes), PREFIX + "weight_scale": scale, PREFIX + "comfy_quant": blob(conf)}
    return sd, weight, codes


def unprefixed(sd: dict) -> dict:
    return {k[len(PREFIX):]: v for k, v in sd.items()}


def loaded(sd: dict, out_f: int, in_f: int):
    linear = fp8_ops.Linear(in_f, out_f, bias=False, dtype=torch.float32)
    linear.requires_grad_(False)
    linear.load_state_dict(unprefixed(expand_int4_layers(dict(sd))), strict=True, assign=True)
    return linear


def test_unpack_reads_twos_complement_nibbles_even_column_low():
    packed = torch.tensor([[-31, -105], [48, 79]], dtype=torch.int8)
    assert unpack_int4(packed).tolist() == [[1, -2, 7, -7], [0, 3, -1, 4]]


def test_unpack_accepts_uint8_storage():
    packed = torch.tensor([[-31, -105]], dtype=torch.int8)
    assert torch.equal(unpack_int4(packed.view(torch.uint8)), unpack_int4(packed))


def test_an_int4_layer_is_read_as_an_int8_convrot_layer():
    sd, _, codes = int4_layer(16, 128, STATED_GROUPSIZE)
    expanded = unprefixed(expand_int4_layers(dict(sd)))
    assert torch.equal(expanded["weight"], codes)
    assert expanded["weight_scale"].shape == (16, 1)
    assert _parse_comfy_quant(expanded["comfy_quant"]) == {
        "format": "int8_tensorwise", "convrot": True, "convrot_groupsize": STATED_GROUPSIZE,
    }


def test_the_loaded_layer_keeps_int8_codes_and_the_stated_group_size():
    sd, _, codes = int4_layer(16, 128, STATED_GROUPSIZE)
    linear = loaded(sd, 16, 128)
    assert linear.weight.dtype == torch.int8
    assert torch.equal(linear.weight, codes)
    assert linear.convrot_groupsize == STATED_GROUPSIZE


def test_the_dequantized_weight_is_back_in_the_original_basis():
    sd, weight, _ = int4_layer(32, 256, STATED_GROUPSIZE)
    with torch.no_grad():
        restored = loaded(sd, 32, 256)(torch.eye(256)).T
    error = ((restored - weight).norm() / weight.norm()).item()
    assert 0.0 < error < 0.15


def test_the_forward_tracks_the_float_layer():
    sd, weight, _ = int4_layer(32, 256, STATED_GROUPSIZE, seed=3)
    x = torch.randn(5, 256, generator=torch.Generator().manual_seed(4))
    reference = x @ weight.T
    with torch.no_grad():
        error = ((loaded(sd, 32, 256)(x) - reference).norm() / reference.norm()).item()
    assert 0.0 < error < 0.15


def test_a_descriptor_without_a_group_size_uses_the_format_default():
    sd, weight, _ = int4_layer(8, 256, 256, state_groupsize=False)
    linear = loaded(sd, 8, 256)
    assert linear.convrot_groupsize == 256
    with torch.no_grad():
        restored = linear(torch.eye(256)).T
    assert ((restored - weight).norm() / weight.norm()).item() < 0.15


def test_a_missing_scale_is_refused():
    sd, _, _ = int4_layer(8, 64, STATED_GROUPSIZE)
    del sd[PREFIX + "weight_scale"]
    with pytest.raises(NativeEngineUnsupportedError, match="int4 layer"):
        expand_int4_layers(sd)


def test_a_raw_int4_layer_that_skipped_the_loader_is_refused_not_misread():
    sd, _, _ = int4_layer(8, 64, STATED_GROUPSIZE)
    with pytest.raises(ValueError, match="quant format 'convrot_w4a4' is not supported"):
        fp8_ops.Linear(64, 8, bias=False).load_state_dict(unprefixed(sd), strict=False)


def test_an_int4_checkpoint_is_detected_as_quantized():
    sd, _, _ = int4_layer(8, 64, STATED_GROUPSIZE)
    assert detect_quant_format({}, expand_int4_layers(dict(sd))) is not None


def test_lora_rides_the_runtime_delta_path_on_an_int4_layer():
    sd, _, _ = int4_layer(8, 64, STATED_GROUPSIZE)
    assert _needs_runtime_deltas(loaded(sd, 8, 64))

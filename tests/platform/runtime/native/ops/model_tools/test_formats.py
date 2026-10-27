from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.platform.runtime.native.arch.moss_image.model import MossImage, convert_moss_image_state_dict
from src.platform.runtime.native.base import load_into_module
from src.platform.runtime.native.detect.registry import match_model_spec
from src.platform.runtime.native.io import quant_formats
from src.platform.runtime.native.io.safetensors_loader import load_torch_file, unpack_int4
from src.platform.runtime.native.kernels import fp8_ops, pick_operations
from src.platform.runtime.native.ops.model_tools import formats
from src.platform.runtime.native.ops.model_tools.errors import UnsupportedFormatError
from src.platform.runtime.native.ops.model_tools.formats import QUANT_FORMATS, get_format, pack_int4
from src.platform.runtime.native.ops.model_tools.quantize import quantize_file, quantize_plan, shape_kept_layers
from src.platform.runtime.native.ops.model_tools.spec import TensorSpec
from tests.platform.runtime.native._quant_layouts import (
    CONVROT_WILD_FORMS,
    reference_hadamard,
    reference_quantize_convrot,
    reference_rotate,
)

COMFY_QUANT_ALGOS = {
    "float8_e4m3fn": (torch.float8_e4m3fn, {"weight_scale", "input_scale"}),
    "float8_e5m2": (torch.float8_e5m2, {"weight_scale", "input_scale"}),
    "nvfp4": (torch.uint8, {"weight_scale", "weight_scale_2", "input_scale", "pre_quant_scale"}),
    "int8_tensorwise": (torch.int8, {"weight_scale"}),
    "convrot_w4a4": (torch.int8, {"weight_scale"}),
}

LINEAR_ERROR_BOUND = {
    "fp8_e4m3": 0.05,
    "fp8_e5m2": 0.1,
    "int8": 0.015,
    "int8_convrot": 0.015,
    "nvfp4": 0.13,
    "int4_convrot": 0.16,
}

MOSS_COSINE_BOUND = {
    "fp8_e4m3": 0.999,
    "fp8_e5m2": 0.998,
    "int8": 0.9999,
    "int8_convrot": 0.9999,
    "nvfp4": 0.99,
    "int4_convrot": 0.99,
}

LAYER = "blocks.0.attn.q"
OUT_F, IN_F = 32, 512

MOSS_DIM, MOSS_HEADS, MOSS_FFN, MOSS_TEXT, MOSS_FREQ, MOSS_CHANNELS = 256, 2, 512, 48, 32, 16
MOSS_CONFIG = {
    "image_model": "moss_image", "hidden_size": MOSS_DIM, "head_dim": MOSS_DIM // MOSS_HEADS,
    "num_heads": MOSS_HEADS, "ffn_dim": MOSS_FFN, "depth": 1, "context_in_dim": MOSS_TEXT,
    "in_channels": MOSS_CHANNELS, "freq_dim": MOSS_FREQ, "rope_base": 10000.0,
}


def raw_layer(path, prefix=LAYER):
    with safe_open(str(path), framework="pt") as f:
        return {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix + ".")}


def write_source(path, seed=0):
    gen = torch.Generator().manual_seed(seed)
    weight = (torch.randn(OUT_F, IN_F, generator=gen) * 0.05).to(torch.bfloat16)
    save_file(
        {
            f"{LAYER}.weight": weight,
            "blocks.0.mlp.fc1.weight": (torch.randn(48, 40, generator=gen) * 0.05).to(torch.bfloat16),
            "img_in.weight": (torch.randn(8, 16, generator=gen) * 0.05).to(torch.bfloat16),
        },
        str(path),
    )
    return weight.float()


def loaded_linear(path):
    sd, _ = load_torch_file(path)
    layer = {k[len(LAYER) + 1:]: v for k, v in sd.items() if k.startswith(LAYER + ".")}
    linear = fp8_ops.Linear(IN_F, OUT_F, bias=False, dtype=torch.float32)
    linear.load_state_dict(layer, strict=True)
    return linear


@pytest.mark.parametrize("key", list(QUANT_FORMATS))
def test_every_format_round_trips_through_the_native_loader(tmp_path, key):
    weight = write_source(tmp_path / "src.safetensors")
    quantize_file(tmp_path / "src.safetensors", tmp_path / "dst.safetensors", family="krea2", quant_format=key)
    linear = loaded_linear(tmp_path / "dst.safetensors")
    x = torch.randn(4, IN_F, generator=torch.Generator().manual_seed(1))
    reference = x @ weight.T
    with torch.no_grad():
        error = ((linear(x) - reference).norm() / reference.norm()).item()
    assert 0.0 < error < LINEAR_ERROR_BOUND[key]


@pytest.mark.parametrize("key", list(QUANT_FORMATS))
def test_the_writer_restore_is_what_the_loader_computes(tmp_path, key):
    write_source(tmp_path / "src.safetensors")
    quantize_file(tmp_path / "src.safetensors", tmp_path / "dst.safetensors", family="krea2", quant_format=key)
    fmt = get_format(key)
    restored = fmt.restore(raw_layer(tmp_path / "dst.safetensors"), (OUT_F, IN_F))
    with torch.no_grad():
        loader = loaded_linear(tmp_path / "dst.safetensors")(torch.eye(IN_F)).T
    assert torch.allclose(restored, loader, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("key", list(QUANT_FORMATS))
def test_written_layers_match_the_comfyui_quant_algos_entry(tmp_path, key):
    write_source(tmp_path / "src.safetensors")
    quantize_file(tmp_path / "src.safetensors", tmp_path / "dst.safetensors", family="krea2", quant_format=key)
    layer = raw_layer(tmp_path / "dst.safetensors")
    descriptor = json.loads(bytes(layer.pop(".comfy_quant").tolist()))
    assert descriptor == get_format(key).descriptor
    storage, parameters = COMFY_QUANT_ALGOS[descriptor["format"]]
    assert layer.pop(".weight").dtype == storage
    assert {name[1:] for name in layer} <= parameters
    assert "quant_group_size" not in descriptor


def test_the_default_format_keeps_todays_fp8_output():
    fmt = get_format(None)
    assert fmt.key == "fp8_e4m3"
    assert fmt.name_suffix == "-fp8"
    assert fmt.descriptor_bytes == json.dumps({"format": "float8_e4m3fn", "full_precision_matrix_mult": True}).encode()


def test_the_convrot_descriptor_is_the_canonical_published_form():
    canonical_length = CONVROT_WILD_FORMS[0][0]
    assert len(get_format("int8_convrot").descriptor_bytes) == canonical_length


def test_the_convrot_writer_uses_the_loader_unrotation_not_a_copy():
    assert formats.convrot_unrotate_weight is quant_formats.convrot_unrotate_weight


def test_int8_convrot_codes_match_the_independent_reference():
    weight = torch.randn(16, 512, generator=torch.Generator().manual_seed(3))
    parts = dict(get_format("int8_convrot").quantize(weight))
    codes, scale = reference_quantize_convrot(weight, 256)
    assert torch.allclose(parts[".weight_scale"], scale, rtol=1e-5)
    differ = parts[".weight"].to(torch.int16) - codes.to(torch.int16)
    assert differ.abs().max() <= 1
    assert (differ != 0).float().mean() < 0.001


def test_int4_codes_match_the_independent_reference():
    weight = torch.randn(16, 512, generator=torch.Generator().manual_seed(4))
    parts = dict(get_format("int4_convrot").quantize(weight))
    rotated = reference_rotate(weight, reference_hadamard(256), 256)
    scale = rotated.abs().amax(dim=1) / 7
    codes = (rotated / scale[:, None]).round().clamp(-7, 7)
    assert torch.allclose(parts[".weight_scale"], scale, rtol=1e-5)
    differ = unpack_int4(parts[".weight"]).to(torch.int16) - codes.to(torch.int16)
    assert differ.abs().max() <= 1
    assert (differ != 0).float().mean() < 0.001


def test_int4_nibbles_follow_the_published_order():
    codes = torch.tensor([[1, -2, 7, -7]], dtype=torch.int8)
    packed = pack_int4(codes)
    assert packed.dtype == torch.int8
    assert packed.tolist() == [[-31, -105]]
    assert torch.equal(unpack_int4(packed), codes)


@pytest.mark.parametrize(
    "key, shape, eligible",
    [
        ("nvfp4", (32, 40), False),
        ("nvfp4", (24, 32), False),
        ("nvfp4", (32, 48), True),
        ("int8_convrot", (32, 384), False),
        ("int8_convrot", (32, 512), True),
        ("int4_convrot", (32, 128), False),
        ("int8", (3, 5), True),
    ],
)
def test_layers_a_format_cannot_hold_stay_in_the_source_dtype(key, shape, eligible):
    fmt = get_format(key)
    plan = [TensorSpec("blocks.0.attn.q.weight", "BF16", shape)]
    new_plan, quantized = quantize_plan(plan, "krea2", (), fmt)
    assert bool(quantized) is eligible
    assert (new_plan == plan) is not eligible
    assert shape_kept_layers(plan, "krea2", (), fmt) == ([] if eligible else ["blocks.0.attn.q.weight"])


@pytest.mark.parametrize("key", list(QUANT_FORMATS))
def test_planned_sizes_match_the_written_tensors(tmp_path, key):
    write_source(tmp_path / "src.safetensors")
    quantize_file(tmp_path / "src.safetensors", tmp_path / "dst.safetensors", family="krea2", quant_format=key)
    layer = raw_layer(tmp_path / "dst.safetensors")
    expected = {suffix: (dtype, shape) for suffix, dtype, shape in get_format(key).layer_specs((OUT_F, IN_F))}
    for suffix, tensor in layer.items():
        if suffix == ".comfy_quant":
            continue
        assert tuple(tensor.shape) == expected[suffix][1]


@pytest.mark.parametrize("key, allowed", [(k, QUANT_FORMATS[k].text_encoders) for k in QUANT_FORMATS])
def test_text_encoders_take_fp8_and_nvfp4_only(key, allowed):
    plan = [TensorSpec("encoder.block.0.layer.0.SelfAttention.q.weight", "BF16", (512, 512))]
    if allowed:
        quantize_plan(plan, "t5xxl", (), get_format(key))
        assert key in ("fp8_e4m3", "fp8_e5m2", "nvfp4")
    else:
        with pytest.raises(UnsupportedFormatError, match="text encoders"):
            quantize_plan(plan, "t5xxl", (), get_format(key))


def test_an_unknown_format_is_refused():
    with pytest.raises(UnsupportedFormatError, match="int3"):
        get_format("int3")


def moss_float_model():
    torch.manual_seed(0)
    model = MossImage.from_config(MOSS_CONFIG, pick_operations(torch.float32, torch.float32))
    for name, param in model.named_parameters():
        param.data = torch.randn_like(param) * (0.05 if param.ndim > 1 else 0.1)
        if name.endswith(("norm.weight", "norm1.weight", "norm2.weight")):
            param.data.uniform_(0.5, 1.5)
    return model.eval()


@pytest.fixture(scope="module")
def moss_source(tmp_path_factory):
    model = moss_float_model()
    path = tmp_path_factory.mktemp("moss") / "moss.safetensors"
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(path))
    return model, path


@pytest.mark.parametrize("key", list(QUANT_FORMATS))
def test_a_quantized_moss_dit_loads_into_the_real_arch_and_tracks_the_float_model(tmp_path, moss_source, key):
    floats, source = moss_source
    dst = tmp_path / "moss-q.safetensors"
    quantize_file(source, dst, family="moss_image", quant_format=key)
    sd, _ = load_torch_file(dst)
    quantised = MossImage.from_config(MOSS_CONFIG, fp8_ops)
    load_into_module(quantised, convert_moss_image_state_dict(sd), match_model_spec(MOSS_CONFIG))
    quantised.eval()
    qkv = quantised.layers[0].proj["qkv"]
    if key == "nvfp4":
        assert qkv._is_nvfp4
    else:
        assert qkv.weight.dtype == {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}.get(key, torch.int8)

    generator = torch.Generator().manual_seed(1)
    x = torch.randn(1, MOSS_CHANNELS, 4, 3, generator=generator)
    context = torch.randn(1, 5, MOSS_TEXT, generator=generator)
    sigma = torch.tensor([0.6])
    with torch.no_grad():
        expected = floats(x, sigma, context)
        actual = quantised(x, sigma, context)
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    assert MOSS_COSINE_BOUND[key] < cosine < 1.0

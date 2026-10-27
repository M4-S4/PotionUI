from __future__ import annotations

import json

import torch

from src.platform.runtime.native.arch.moss_image.model import MossImage, convert_moss_image_state_dict
from src.platform.runtime.native.base import load_into_module
from src.platform.runtime.native.detect.registry import match_model_spec
from src.platform.runtime.native.kernels import fp8_ops, pick_operations

from ._quant_layouts import convrot_descriptor, descriptor_blob, reference_quantize_convrot

DIM, HEADS, FFN, DEPTH, TEXT_DIM, FREQ, CHANNELS = 64, 2, 96, 2, 48, 32, 16
GROUPSIZE = 16
QUANTISED = ("qkv", "o", "ffn_up_gate", "ffn_down")

CONFIG = {
    "image_model": "moss_image", "hidden_size": DIM, "head_dim": DIM // HEADS, "num_heads": HEADS,
    "ffn_dim": FFN, "depth": DEPTH, "context_in_dim": TEXT_DIM, "in_channels": CHANNELS,
    "freq_dim": FREQ, "rope_base": 10000.0,
}


def _float_model(seed):
    torch.manual_seed(seed)
    model = MossImage.from_config(CONFIG, pick_operations(torch.float32, torch.float32))
    for name, param in model.named_parameters():
        param.data = torch.randn_like(param) * (0.5 if param.ndim > 1 else 0.1)
        if name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            param.data.uniform_(0.5, 1.5)
    return model.eval()


def _int8_checkpoint(model):
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out = dict(state)
    blob = descriptor_blob(convrot_descriptor(GROUPSIZE))
    for layer in range(DEPTH):
        for name in QUANTISED:
            prefix = f"layers.{layer}.proj.{name}"
            codes, scale = reference_quantize_convrot(state[f"{prefix}.weight"], GROUPSIZE)
            out[f"{prefix}.weight"] = codes
            out[f"{prefix}.weight_scale"] = scale
            out[f"{prefix}.comfy_quant"] = blob.clone()
        out[f"layers.{layer}.comfy_attention.config"] = descriptor_blob({"attention": "comfy_kitchen_int8"})
    out["rope.bands"] = torch.ones(DIM // HEADS // 8)
    return out


def test_int8_convrot_checkpoint_with_attention_hints_loads_and_tracks_the_float_model():
    floats = _float_model(0)
    checkpoint = _int8_checkpoint(floats)
    assert any(k.endswith("comfy_attention.config") for k in checkpoint)
    quantised = MossImage.from_config(CONFIG, fp8_ops)
    load_into_module(quantised, convert_moss_image_state_dict(checkpoint), match_model_spec(CONFIG))
    quantised.eval()

    generator = torch.Generator().manual_seed(1)
    x = torch.randn(1, CHANNELS, 4, 3, generator=generator)
    context = torch.randn(1, 5, TEXT_DIM, generator=generator)
    sigma = torch.tensor([0.6])
    with torch.no_grad():
        expected = floats(x, sigma, context)
        actual = quantised(x, sigma, context)
    assert actual.shape == expected.shape
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert cosine > 0.9999
    assert (actual - expected).abs().max() > 0
    assert quantised.layers[0].proj["qkv"].weight.dtype == torch.int8


def test_comfy_attention_hint_is_a_json_blob_the_loader_never_sees():
    hint = descriptor_blob({"attention": "comfy_kitchen_int8"})
    assert json.loads(bytes(hint.tolist())) == {"attention": "comfy_kitchen_int8"}
    converted = convert_moss_image_state_dict({"layers.0.comfy_attention.config": hint})
    assert converted == {}

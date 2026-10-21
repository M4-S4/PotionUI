from __future__ import annotations

import torch

from src.platform.runtime.native.arch.flux3.model import Flux3, convert_flux3_state_dict
from src.platform.runtime.native.base import load_into_module
from src.platform.runtime.native.detect.registry import match_model_spec
from src.platform.runtime.native.detect.unet_detect import detect_unet_config
from vendor.gpl.comfyui.ops import fp8_ops, pick_operations

from ._quant_layouts import convrot_descriptor, descriptor_blob, reference_quantize_convrot

HIDDEN, HEADS, MLP, DEPTH, SINGLE, CHANNELS, CONTEXT = 64, 2, 96, 1, 1, 16, 48
GROUPSIZE = 16
QUANTISED = ("qkv", "mlp_in", "attn_out", "mlp_out")

CONFIG = {
    "image_model": "flux3", "hidden_size": HIDDEN, "num_heads": HEADS, "head_dim": HIDDEN // HEADS,
    "depth": DEPTH, "depth_single_blocks": SINGLE, "in_channels": CHANNELS, "context_in_dim": CONTEXT,
    "mlp_hidden": MLP, "flow_map": False, "has_image_cond": False, "fused_qkv": True,
}


def _float_model(seed):
    torch.manual_seed(seed)
    model = Flux3.from_config(CONFIG, pick_operations(torch.float32, torch.float32))
    for name, param in model.named_parameters():
        param.data = torch.randn_like(param) * (0.3 if param.ndim > 1 else 0.1)
        if name.endswith("scale"):
            param.data.uniform_(0.5, 1.5)
    return model.eval()


def _block_stems(model):
    return [
        name for name, _ in model.named_modules()
        if name.rsplit(".", 1)[-1] in QUANTISED and "blocks" in name
    ]


def _int8_checkpoint(model):
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out = dict(state)
    blob = descriptor_blob(convrot_descriptor(GROUPSIZE))
    for stem in _block_stems(model):
        codes, scale = reference_quantize_convrot(state[f"{stem}.weight"], GROUPSIZE)
        out[f"{stem}.weight"] = codes
        out[f"{stem}.weight_scale"] = scale
        out[f"{stem}.comfy_quant"] = blob.clone()
    return out


def test_comfy_quant_sidecars_are_expected_keys_for_the_detected_spec():
    checkpoint = _int8_checkpoint(_float_model(0))
    spec = match_model_spec(detect_unet_config(checkpoint))
    assert spec.family == "flux3"
    assert spec.key_is_expected_unexpected("single_blocks.0.qkv.comfy_quant")


def test_int8_convrot_fused_checkpoint_loads_and_tracks_the_float_model():
    floats = _float_model(0)
    checkpoint = _int8_checkpoint(floats)
    config = detect_unet_config(checkpoint)
    quantised = Flux3.from_config(config, fp8_ops)
    load_into_module(quantised, convert_flux3_state_dict(checkpoint), match_model_spec(config))
    quantised.eval()

    generator = torch.Generator().manual_seed(1)
    x = torch.randn(1, CHANNELS, 4, 3, generator=generator)
    context = torch.randn(1, 5, CONTEXT, generator=generator)
    sigma = torch.tensor([0.6])
    with torch.no_grad():
        expected = floats(x, sigma, context)
        actual = quantised(x, sigma, context)
    assert actual.shape == expected.shape
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert cosine > 0.999
    assert (actual - expected).abs().max() > 0
    assert quantised.content_mode_blocks["image"][0].qkv.weight.dtype == torch.int8
    assert quantised.single_blocks[0].qkv.weight.dtype == torch.int8

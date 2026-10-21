from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from src.platform.runtime.native.detect.unet_detect import detect_unet_config
from src.platform.runtime.native.io.safetensors_loader import load_torch_file
from src.platform.runtime.native.io.state_dict_utils import is_int4_packed, linear_in_features

from .test_nvfp4_packed_detect import krea2_sd

WQ = "blocks.0.attn.wq.weight"


def blob(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def pack_int4_layer(sd: dict, key: str, *, descriptor: bool = True) -> None:
    out_f, in_f = sd[key].shape
    base = key[: -len(".weight")]
    sd[key] = torch.zeros(out_f, in_f // 2, dtype=torch.int8)
    sd[base + ".weight_scale"] = torch.ones(out_f)
    if descriptor:
        sd[base + ".comfy_quant"] = blob({"format": "convrot_w4a4", "convrot_groupsize": 16})


def test_a_packed_int4_layer_reports_its_true_width():
    sd = krea2_sd()
    pack_int4_layer(sd, WQ)
    assert sd[WQ].shape[1] == 32
    assert is_int4_packed(sd, WQ)
    assert linear_in_features(sd, WQ) == 64


def test_krea2_detection_reads_true_widths_from_int4_layers():
    sd = krea2_sd()
    pack_int4_layer(sd, "txtmlp.1.weight")
    pack_int4_layer(sd, "tmlp.0.weight")
    config = detect_unet_config(sd)
    assert config["image_model"] == "krea2"
    assert (config["txtdim"], config["txtheads"], config["tdim"]) == (32, 4, 16)


def test_a_header_only_state_dict_is_read_from_the_row_scale_shape():
    sd = krea2_sd()
    pack_int4_layer(sd, WQ)
    meta = {k: torch.empty(v.shape, dtype=v.dtype, device="meta") for k, v in sd.items()}
    assert is_int4_packed(meta, WQ)
    assert linear_in_features(meta, WQ) == 64


def test_int8_layers_are_not_mistaken_for_int4():
    sd = krea2_sd()
    sd[WQ] = torch.zeros(64, 64, dtype=torch.int8)
    sd["blocks.0.attn.wq.weight_scale"] = torch.ones(64, 1)
    sd["blocks.0.attn.wq.comfy_quant"] = blob({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16})
    assert not is_int4_packed(sd, WQ)
    assert linear_in_features(sd, WQ) == 64


def test_an_int4_file_read_through_the_loader_is_already_full_width(tmp_path):
    sd = krea2_sd()
    pack_int4_layer(sd, WQ)
    path = tmp_path / "krea2-int4.safetensors"
    save_file(sd, str(path))
    loaded, _ = load_torch_file(path)
    assert loaded[WQ].shape == (64, 64)
    assert not is_int4_packed(loaded, WQ)
    assert linear_in_features(loaded, WQ) == 64

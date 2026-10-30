from __future__ import annotations

import torch

from src.platform.runtime.native.arch.flux3.model import Flux3
from src.platform.runtime.native.lora import map_lora_keys
from src.platform.runtime.native.lora.key_mapping import _select_key_map
from src.platform.runtime.native.kernels import pick_operations

HIDDEN, RANK = 64, 4


def _config(fused):
    return {
        "image_model": "flux3", "hidden_size": HIDDEN, "num_heads": 2, "head_dim": 32, "depth": 1,
        "depth_single_blocks": 1, "in_channels": 16, "context_in_dim": 48, "mlp_hidden": 96,
        "flow_map": False, "has_image_cond": False, "fused_qkv": fused,
    }


def _build(fused):
    return Flux3.from_config(_config(fused), pick_operations(torch.float32, torch.float32))


def _comfy(stem, out, inf, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return {
        f"{stem}.lora_up.weight": torch.randn(out, RANK, generator=gen),
        f"{stem}.lora_down.weight": torch.randn(RANK, inf, generator=gen),
        f"{stem}.alpha": torch.tensor(float(RANK)),
    }


def test_fused_module_maps_split_qkv_loras_onto_row_slices():
    module = _build(True)
    sd = {}
    for part, seed in (("q_proj", 1), ("k_proj", 2), ("v_proj", 3)):
        sd.update(_comfy(f"diffusion_model.content_mode_blocks.image.0.{part}", HIDDEN, HIDDEN, seed))
        sd.update(_comfy(f"diffusion_model.single_blocks.0.{part}", HIDDEN, HIDDEN, seed))
    mapped, unmatched = map_lora_keys(sd, module)
    assert unmatched == []
    assert set(mapped) == {"content_mode_blocks.image.0.qkv.weight", "single_blocks.0.qkv.weight"}
    assert sorted(d.target_slice for d in mapped["single_blocks.0.qkv.weight"]) == [
        (0, 0, HIDDEN), (0, HIDDEN, HIDDEN), (0, 2 * HIDDEN, HIDDEN),
    ]


def test_fused_module_still_accepts_a_whole_qkv_lora():
    module = _build(True)
    mapped, unmatched = map_lora_keys(_comfy("lora_unet_txt_mode_blocks_0_qkv", 3 * HIDDEN, HIDDEN), module)
    assert unmatched == []
    assert [d.target_slice for d in mapped["txt_mode_blocks.0.qkv.weight"]] == [None]


def test_split_module_keeps_the_name_for_name_map():
    module = _build(False)
    km = _select_key_map(module)
    assert km["diffusion_model.single_blocks.0.q_proj"] == ("single_blocks.0.q_proj.weight", None)
    assert not any(key.endswith(".qkv") for key in km)

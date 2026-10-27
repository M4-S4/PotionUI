from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.arch.moss_image.config import MossImageConfig
from src.platform.runtime.native.arch.moss_image.model import MossImage
from src.platform.runtime.native.lora import apply_loras, map_lora_keys, remove_loras
from src.platform.runtime.native.lora.key_mapping import _select_key_map, build_moss_image_lora_key_map
from src.platform.runtime.native.kernels import pick_operations

DIM = 16
RANK = 4

CFG = MossImageConfig(
    dim=DIM, heads=2, head_dim=8, ffn_dim=24, depth=2, text_dim=12,
    in_channels=4, freq_dim=8, rope_base=10000.0,
)


def _build() -> MossImage:
    module = MossImage(CFG, operations=pick_operations(torch.float32, torch.float32))
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn_like(param) * 0.02)
    module.eval()
    return module


def _pair(stem: str, out: int, inf: int, up: str, down: str, seed: int, alpha: bool = False):
    gen = torch.Generator().manual_seed(seed)
    sd = {
        f"{stem}.{up}.weight": torch.randn(out, RANK, generator=gen) * 0.1,
        f"{stem}.{down}.weight": torch.randn(RANK, inf, generator=gen) * 0.1,
    }
    if alpha:
        sd[f"{stem}.alpha"] = torch.tensor(float(RANK))
    return sd


def _comfy(stem: str, out: int, inf: int, seed: int = 1):
    return _pair(stem, out, inf, "lora_up", "lora_down", seed, alpha=True)


def _peft(stem: str, out: int, inf: int, seed: int = 1):
    return _pair(stem, out, inf, "lora_B", "lora_A", seed)


def test_select_key_map_routes_moss_image_module():
    km = _select_key_map(_build())
    assert km["layers.0.proj.q"] == ("layers.0.proj.qkv.weight", (0, 0, DIM))
    assert "blocks.0.attn.wq" not in km


def test_key_map_splits_qkv_into_row_slices_for_every_dialect():
    km = build_moss_image_lora_key_map(_build())
    for prefix in ("", "diffusion_model.", "transformer.", "base_model.model."):
        assert km[f"{prefix}layers.1.proj.q"] == ("layers.1.proj.qkv.weight", (0, 0, DIM))
        assert km[f"{prefix}layers.1.proj.k"] == ("layers.1.proj.qkv.weight", (0, DIM, DIM))
        assert km[f"{prefix}layers.1.proj.v"] == ("layers.1.proj.qkv.weight", (0, 2 * DIM, DIM))
    assert km["lora_unet_layers_1_proj_k"] == ("layers.1.proj.qkv.weight", (0, DIM, DIM))


def test_key_map_keeps_fused_and_unsplit_targets_whole():
    km = build_moss_image_lora_key_map(_build())
    assert km["diffusion_model.layers.0.proj.qkv"] == ("layers.0.proj.qkv.weight", None)
    assert km["diffusion_model.layers.0.proj.o"] == ("layers.0.proj.o.weight", None)
    assert km["transformer.layers.0.proj.ffn_up_gate"] == ("layers.0.proj.ffn_up_gate.weight", None)
    assert km["diffusion_model.text_embedder.blocks.0.qkv"] == ("text_embedder.blocks.0.qkv.weight", None)
    assert km["lora_unet_text_embedder_blocks_1_ffn_down"] == ("text_embedder.blocks.1.ffn_down.weight", None)
    assert "diffusion_model.text_embedder.blocks.0.q" not in km


def test_comfy_dialect_maps_split_qkv_onto_three_row_slices():
    module = _build()
    sd = {}
    for part, seed in (("q", 1), ("k", 2), ("v", 3)):
        sd.update(_comfy(f"diffusion_model.layers.0.proj.{part}", DIM, DIM, seed))
    sd.update(_comfy("diffusion_model.layers.0.proj.ffn_up_gate", 2 * CFG.ffn_dim, DIM, 4))
    mapped, unmatched = map_lora_keys(sd, module)
    assert unmatched == []
    assert set(mapped) == {"layers.0.proj.qkv.weight", "layers.0.proj.ffn_up_gate.weight"}
    assert sorted(d.target_slice for d in mapped["layers.0.proj.qkv.weight"]) == [
        (0, 0, DIM), (0, DIM, DIM), (0, 2 * DIM, DIM),
    ]


def test_peft_transformer_dialect_maps_split_qkv():
    module = _build()
    sd = {}
    for part, seed in (("q", 1), ("k", 2), ("v", 3)):
        sd.update(_peft(f"transformer.layers.1.proj.{part}", DIM, DIM, seed))
    sd.update(_peft("transformer.layers.1.proj.o", DIM, DIM, 4))
    sd.update(_peft("transformer.layers.1.proj.ffn_down", DIM, CFG.ffn_dim, 5))
    mapped, unmatched = map_lora_keys(sd, module)
    assert unmatched == []
    assert set(mapped) == {
        "layers.1.proj.qkv.weight", "layers.1.proj.o.weight", "layers.1.proj.ffn_down.weight",
    }
    assert len(mapped["layers.1.proj.qkv.weight"]) == 3


def test_ai_toolkit_fused_qkv_dialect_maps_whole_weight():
    module = _build()
    sd = _peft("diffusion_model.layers.0.proj.qkv", 3 * DIM, DIM)
    mapped, unmatched = map_lora_keys(sd, module)
    assert unmatched == []
    assert [d.target_slice for d in mapped["layers.0.proj.qkv.weight"]] == [None]


def test_unknown_keys_are_reported_unmatched():
    module = _build()
    sd = _comfy("diffusion_model.layers.0.proj.q", DIM, DIM)
    sd.update(_comfy("diffusion_model.layers.9.proj.q", DIM, DIM, 2))
    sd.update(_comfy("diffusion_model.layers.0.proj.nope", DIM, DIM, 3))
    mapped, unmatched = map_lora_keys(sd, module)
    assert set(mapped) == {"layers.0.proj.qkv.weight"}
    assert sorted(unmatched) == [
        "diffusion_model.layers.0.proj.nope", "diffusion_model.layers.9.proj.q",
    ]


@pytest.mark.parametrize("part,row", [("q", 0), ("k", 1), ("v", 2)])
def test_apply_patches_only_the_matching_qkv_rows_and_remove_restores(part, row):
    module = _build()
    weight = module.layers[0].proj["qkv"].weight
    before = weight.detach().clone()
    stem = f"diffusion_model.layers.0.proj.{part}"
    sd = _comfy(stem, DIM, DIM, seed=7)
    patched, unmatched = apply_loras(module, [(sd, 0.5)])
    assert patched == 1 and unmatched == []
    expected = before.clone()
    expected[row * DIM:(row + 1) * DIM] += 0.5 * (sd[f"{stem}.lora_up.weight"] @ sd[f"{stem}.lora_down.weight"])
    assert torch.allclose(weight, expected, atol=1e-6)
    untouched = torch.ones(3 * DIM, dtype=torch.bool)
    untouched[row * DIM:(row + 1) * DIM] = False
    assert torch.equal(weight[untouched], before[untouched])
    remove_loras(module)
    assert torch.allclose(weight, before, atol=1e-6)


def test_peft_split_qkv_matches_the_equivalent_fused_lora():
    split_module, fused_module = _build(), _build()
    fused_module.load_state_dict(split_module.state_dict())
    sd = {}
    for part, seed in (("q", 1), ("k", 2), ("v", 3)):
        sd.update(_peft(f"transformer.layers.0.proj.{part}", DIM, DIM, seed))
    parts = ("q", "k", "v")
    stacked_up = torch.zeros(3 * DIM, 3 * RANK)
    for index, part in enumerate(parts):
        stacked_up[index * DIM:(index + 1) * DIM, index * RANK:(index + 1) * RANK] = (
            sd[f"transformer.layers.0.proj.{part}.lora_B.weight"]
        )
    fused = {
        "diffusion_model.layers.0.proj.qkv.lora_B.weight": stacked_up,
        "diffusion_model.layers.0.proj.qkv.lora_A.weight": torch.cat(
            [sd[f"transformer.layers.0.proj.{part}.lora_A.weight"] for part in parts], dim=0,
        ),
    }
    apply_loras(split_module, [(sd, 1.0)])
    apply_loras(fused_module, [(fused, 1.0)])
    assert torch.allclose(
        split_module.layers[0].proj["qkv"].weight, fused_module.layers[0].proj["qkv"].weight, atol=1e-5,
    )

from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.arch.flux3.model import convert_flux3_state_dict
from src.platform.runtime.native.kernels import fp8_ops

FP8 = torch.float8_e4m3fn


def _rowwise_fp8(out_features, in_features, seed):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(out_features, in_features, generator=generator) * 0.05
    scale = weight.abs().amax(dim=1) / 448.0
    return (weight / scale[:, None]).to(FP8), scale


def _dequantized_linear(weight, scale, x):
    sd = convert_flux3_state_dict({"proj.weight": weight, "proj.weight_scale": scale})
    lin = fp8_ops.Linear(weight.shape[1], weight.shape[0], bias=False)
    lin.comfy_cast_weights = True
    lin._load_from_state_dict(
        {"weight": sd["proj.weight"], "weight_scale": sd["proj.weight_scale"]}, "", {}, True, [], [], [],
    )
    return lin(x), sd


@pytest.mark.parametrize("out_features,in_features", [(48, 32), (32, 32), (32, 48)])
def test_per_row_scale_dequantizes_each_row_with_its_own_scale(out_features, in_features):
    weight, scale = _rowwise_fp8(out_features, in_features, seed=out_features)
    x = torch.randn(3, in_features, generator=torch.Generator().manual_seed(1))
    out, sd = _dequantized_linear(weight, scale, x)
    expected = x @ (weight.float() * scale[:, None]).T
    assert sd["proj.weight_scale"].shape == (out_features, 1)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_only_one_dimensional_scales_of_fp8_weights_are_reshaped():
    fp8_weight, row_scale = _rowwise_fp8(8, 16, seed=3)
    bf16_weight = torch.randn(8, 16, dtype=torch.bfloat16)
    sd = {
        "a.weight": fp8_weight,
        "a.weight_scale": row_scale,
        "b.weight": bf16_weight,
        "b.weight_scale": torch.ones(8),
        "c.weight": fp8_weight.clone(),
        "c.weight_scale": torch.tensor(0.5),
        "d.weight": fp8_weight.clone(),
        "d.weight_scale": torch.ones(8, 1),
        "e.scale": torch.ones(8),
    }
    out = convert_flux3_state_dict(sd)
    assert out["a.weight_scale"].shape == (8, 1)
    assert out["b.weight_scale"].shape == (8,)
    assert out["c.weight_scale"].shape == ()
    assert out["d.weight_scale"].shape == (8, 1)
    assert out["e.scale"].shape == (8,)
    assert sd["a.weight_scale"].shape == (8,)


def test_scales_of_e5m2_weights_are_reshaped_like_e4m3():
    e5m2_weight = torch.zeros(8, 16).to(torch.float8_e5m2)

    out = convert_flux3_state_dict({"a.weight": e5m2_weight, "a.weight_scale": torch.ones(8)})

    assert out["a.weight_scale"].shape == (8, 1)


def test_comfy_attention_hints_are_dropped():
    sd = {
        "single_blocks.0.attn.qkv.weight": torch.ones(2, 2),
        "single_blocks.0.comfy_attention.config": torch.zeros(4, dtype=torch.uint8),
        "content_mode_blocks.image.0.comfy_attention.config": torch.zeros(4, dtype=torch.uint8),
    }
    assert list(convert_flux3_state_dict(sd)) == ["single_blocks.0.attn.qkv.weight"]

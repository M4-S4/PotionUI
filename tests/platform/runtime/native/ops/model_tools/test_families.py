from __future__ import annotations

import pytest

from src.platform.runtime.native.ops.model_tools.errors import UnsupportedFamilyError
from src.platform.runtime.native.ops.model_tools.families import detect_family, is_kept, keep_patterns
from src.platform.runtime.native.ops.model_tools.spec import TensorSpec

from .fixtures import FAMILY_LAYOUTS

FLUX = {
    "img_in.weight": (256, 64),
    "txt_in.weight": (256, 128),
    "double_blocks.0.img_attn.norm.key_norm.scale": (128,),
    "double_blocks.0.img_attn.qkv.weight": (768, 256),
    "single_blocks.0.linear1.weight": (10, 256),
}


def plan_of(layout, prefix=""):
    return [TensorSpec(prefix + k, "BF16", s) for k, s in layout.items()]


def test_detects_flux_from_registry():
    assert detect_family(plan_of(FLUX)) == "flux"


def test_detects_flux_with_checkpoint_prefix():
    assert detect_family(plan_of(FLUX, "model.diffusion_model.")) == "flux"


def test_detects_text_encoder():
    quant, kept = FAMILY_LAYOUTS["t5xxl"]
    assert detect_family(plan_of({**quant, **kept})) == "t5xxl"


def test_unknown_layout_is_refused():
    with pytest.raises(UnsupportedFamilyError):
        detect_family(plan_of({"foo.weight": (4, 4)}))


@pytest.mark.parametrize(
    "key",
    [
        "x.norm.weight",
        "a.modulation.lin.weight",
        "blocks.0.scale_shift_table",
        "t_embedder.mlp.0.weight",
        "double_blocks.0.img_mod.lin.weight",
        "final_layer.linear.weight",
        "img_in.weight",
    ],
)
def test_common_keep_rules(key):
    assert is_kept(key, keep_patterns("flux"))


@pytest.mark.parametrize("key", ["blocks.0.attn.q.weight", "single_blocks.0.linear1.weight", "transformer_blocks.0.ff.net.2.weight"])
def test_ordinary_linears_are_not_kept(key):
    assert not is_kept(key, keep_patterns("flux"))

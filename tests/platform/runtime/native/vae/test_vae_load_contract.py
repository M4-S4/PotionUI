from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from src.platform.runtime.native.base import load_into_module
from src.platform.runtime.native.vae.ae_2d import AutoEncoder2D
from src.platform.runtime.native.vae.ltx_audio import LTXAudioAutoencoder, LTXVocoder, LTXVocoderAMP
from src.platform.runtime.native.vae.ltx_latent_upsampler import LTXLatentUpsampler
from src.platform.runtime.native.vae.minimax_h3_latent_upsampler import MiniMaxH3LatentUpsampler
from src.platform.runtime.native.vae.loader import _VaeSpec
from src.platform.runtime.native.kernels import disable_weight_init

from . import (
    test_ae_2d,
    test_causal_3d,
    test_causal_3d_v2,
    test_ltx_audio,
    test_ltx_causal_video,
    test_ltx_diffusion_video,
    test_ltx_latent_upsampler,
    test_minimax_h3_audio,
    test_minimax_h3_latent_upsampler,
    test_minimax_h3_video,
    test_minimax_music3_dav,
    test_qwen_image21_vae,
    test_seedvr2_causal_video,
)


@dataclass(frozen=True)
class VaeFamily:
    id: str
    build: Callable[[], torch.nn.Module]
    variant: str


def _ae_2d():
    config = test_ae_2d._tiny_config(vae_type="flux_ae", latent=16, has_quant_conv=False, has_batchnorm=False)
    return AutoEncoder2D.from_config(config, disable_weight_init)


FAMILIES = [
    VaeFamily("ae_2d", _ae_2d, "flux_ae"),
    VaeFamily("causal_3d", test_causal_3d._build_tiny, "qwen_image"),
    VaeFamily("causal_3d_v2", test_causal_3d_v2._build_tiny, "wan2.2"),
    VaeFamily(
        "ltx_audio",
        lambda: LTXAudioAutoencoder.from_config(
            test_ltx_audio._TINY_AUDIO_VAE_CONFIG, disable_weight_init
        ),
        "ltx_audio",
    ),
    VaeFamily(
        "ltx_vocoder",
        lambda: LTXVocoder.from_config(test_ltx_audio._TINY_VOCODER_CONFIG, disable_weight_init),
        "ltx_vocoder",
    ),
    VaeFamily(
        "ltx_vocoder_amp",
        lambda: LTXVocoderAMP.from_config(
            test_ltx_audio._TINY_AMP_NESTED_CONFIG, disable_weight_init
        ),
        "ltx_vocoder_amp",
    ),
    VaeFamily("ltx_causal_video", test_ltx_causal_video._build_tiny, "ltx_causal_video"),
    VaeFamily("ltx_diffusion_video", test_ltx_diffusion_video._build_tiny, "ltx_diffusion_video"),
    VaeFamily(
        "ltx_latent_upsampler",
        lambda: LTXLatentUpsampler.from_config(
            test_ltx_latent_upsampler._TINY_CONFIG, disable_weight_init
        ),
        "ltx_latent_upsampler",
    ),
    VaeFamily("minimax_h3_audio", test_minimax_h3_audio._build, "minimax_h3_audio"),
    VaeFamily(
        "minimax_h3_latent_upsampler",
        lambda: MiniMaxH3LatentUpsampler.from_config(
            test_minimax_h3_latent_upsampler._TINY_CONFIG, disable_weight_init
        ),
        "minimax_h3_latent_upsampler",
    ),
    VaeFamily("minimax_h3_video", test_minimax_h3_video._build, "minimax_h3_video"),
    VaeFamily("minimax_music3_dav", test_minimax_music3_dav._build, "minimax_music3_dav"),
    VaeFamily("qwen_image21", test_qwen_image21_vae._build_qwen21, "qwen_image21"),
    VaeFamily("seedvr2_causal_video", test_seedvr2_causal_video._build_tiny, "seedvr2_causal_video"),
]

families = pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.id)


def _randomize(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                p.normal_(std=0.02)
        for b in module.buffers():
            if b.is_floating_point():
                b.normal_(std=0.02)


@families
def test_post_load_leaves_the_state_dict_untouched(family):
    module = family.build()
    _randomize(module)
    before = {k: v.detach().clone() for k, v in module.state_dict().items()}
    module.post_load()
    after = module.state_dict()
    assert after.keys() == before.keys()
    assert all(torch.equal(after[k], before[k]) for k in before)


@families
def test_self_consistent_state_dict_passes_load_integrity(family):
    source = family.build()
    _randomize(source)
    sd = source.state_dict()
    target = family.build()
    load_into_module(target, sd, _VaeSpec(family="vae", variant=family.variant))
    loaded = target.state_dict()
    assert loaded.keys() == sd.keys()
    assert all(torch.equal(loaded[k], sd[k]) for k in sd)

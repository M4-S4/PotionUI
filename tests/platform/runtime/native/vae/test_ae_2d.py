"""Tiny-config smoke tests + real-checkpoint load/roundtrip for AutoEncoder2D."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from src.platform.runtime.native.base import load_into_module
from src.platform.runtime.native.detect.vae_detect import detect_vae_config
from src.platform.runtime.native.kernels import disable_weight_init
from src.platform.runtime.native.vae.ae_2d import AutoEncoder2D
from src.platform.runtime.native.vae.key_convert import hoist_nested_quant_convs
from src.platform.runtime.native.vae.loader import _VaeSpec, load_vae

_FLUX_AE_PATH = Path("models/vae/ae.sft")
_FLUX2_VAE_PATH = Path("models/vae/flux2-vae.safetensors")


def _randomize_weights(module: torch.nn.Module) -> None:
    # `disable_weight_init` deliberately skips `reset_parameters` (real
    # loading always overwrites weights via `load_state_dict`), so a freshly
    # constructed module's tensors are raw `torch.empty` memory -- not a fair
    # stand-in for "real" weights in a test. Fill them with finite values.
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                p.normal_(std=0.02)
        for name, b in module.named_buffers():
            if b is not None and b.is_floating_point() and "running_var" in name:
                b.fill_(1.0)
            elif b is not None and b.is_floating_point():
                b.zero_()


def _tiny_config(*, vae_type: str, latent: int, has_quant_conv: bool, has_batchnorm: bool) -> dict:
    return {
        "vae_type": vae_type,
        "latent_channels": latent,
        "in_channels": 3,
        "out_channels": 3,
        "key_layout": "diffusers" if has_quant_conv else "ldm",
        "has_quant_conv": has_quant_conv,
        "has_batchnorm": has_batchnorm,
    }


@pytest.mark.parametrize(
    "vae_type,latent,has_quant_conv,has_batchnorm",
    [
        ("flux_ae", 16, False, False),
        ("flux2_ae", 32, True, True),
    ],
)
def test_tiny_config_encode_decode_roundtrip_shape(vae_type, latent, has_quant_conv, has_batchnorm):
    config = _tiny_config(vae_type=vae_type, latent=latent, has_quant_conv=has_quant_conv, has_batchnorm=has_batchnorm)
    module = AutoEncoder2D.from_config(config, disable_weight_init)
    module.eval()
    _randomize_weights(module)

    # self-consistent state dict (random init) must pass the same load-integrity
    # gate the real loader uses -- proves the module's own keys are internally
    # consistent (no accidental duplicate/missing submodule wiring).
    sd = module.state_dict()
    spec = _VaeSpec(family="vae", variant=vae_type)
    load_into_module(module, sd, spec)

    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        z = module.encode(pixels)
        out = module.decode(z)

    downscale = 8 * (2 if has_batchnorm else 1)
    expected_latent_channels = latent * (4 if has_batchnorm else 1)
    assert z.shape == (1, expected_latent_channels, 32 // downscale, 32 // downscale)
    assert out.shape == (1, 3, 32, 32)
    assert torch.isfinite(out).all()


def _bfl_layout(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    sd = {}
    for key, value in module.state_dict().items():
        if key.startswith("quant_conv."):
            key = "encoder." + key
        elif key.startswith("post_quant_conv."):
            key = "decoder." + key
        sd[key] = value.to(torch.bfloat16) if value.is_floating_point() else value
    return sd


def test_bfl_native_layout_is_detected_as_flux2_ae_in_ldm_layout():
    config = _tiny_config(vae_type="flux2_ae", latent=32, has_quant_conv=True, has_batchnorm=True)
    module = AutoEncoder2D.from_config(config, disable_weight_init)
    detected = detect_vae_config(_bfl_layout(module))
    assert detected["vae_type"] == "flux2_ae"
    assert detected["key_layout"] == "ldm"
    assert detected["has_quant_conv"] is True
    assert detected["has_batchnorm"] is True
    assert detected["latent_channels"] == 32


def test_hoist_nested_quant_convs_only_moves_the_two_convs():
    sd = {
        "encoder.quant_conv.weight": torch.zeros(1),
        "decoder.post_quant_conv.bias": torch.zeros(1),
        "encoder.conv_in.weight": torch.zeros(1),
        "bn.running_mean": torch.zeros(1),
    }
    assert set(hoist_nested_quant_convs(sd)) == {
        "quant_conv.weight", "post_quant_conv.bias", "encoder.conv_in.weight", "bn.running_mean",
    }


def test_bfl_native_layout_loads_with_no_missing_or_unexpected_keys(tmp_path):
    config = _tiny_config(vae_type="flux2_ae", latent=32, has_quant_conv=True, has_batchnorm=True)
    source = AutoEncoder2D.from_config(config, disable_weight_init)
    _randomize_weights(source)
    sd = _bfl_layout(source)
    path = tmp_path / "ae.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))

    vae = load_vae(path, disable_weight_init, device="cpu")

    assert vae.vae_type == "flux2_ae"
    assert vae.z_channels == 32
    assert vae.has_quant_conv and vae.has_batchnorm
    torch.testing.assert_close(vae.quant_conv.weight.float(), sd["encoder.quant_conv.weight"].float())
    torch.testing.assert_close(vae.post_quant_conv.weight.float(), sd["decoder.post_quant_conv.weight"].float())
    torch.testing.assert_close(vae.bn.running_mean.float(), sd["bn.running_mean"].float())


def test_bn_pixel_unshuffle_matches_the_reference_reshape_permute():
    z = torch.randn(1, 32, 6, 10)
    h, w = 3, 5
    reference = z.reshape(1, 32, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4).reshape(1, 128, h, w)
    assert torch.equal(torch.nn.functional.pixel_unshuffle(z, 2), reference)
    packed = torch.randn(1, 128, h, w)
    reference_back = packed.reshape(1, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, 2 * h, 2 * w)
    assert torch.equal(torch.nn.functional.pixel_shuffle(packed, 2), reference_back)


@pytest.mark.requires_models
@pytest.mark.skipif(not _FLUX_AE_PATH.exists(), reason="models/vae/ae.sft not present")
def test_real_flux_ae_load_and_roundtrip():
    vae = load_vae(_FLUX_AE_PATH, disable_weight_init, device="cpu")
    vae.eval()
    assert vae.vae_type == "flux_ae"
    assert vae.z_channels == 16

    pixels = torch.rand(1, 3, 64, 64) * 2.0 - 1.0
    with torch.no_grad():
        latent = vae.encode(pixels)
        recon = vae.decode(latent)

    assert latent.shape == (1, 16, 8, 8)
    assert recon.shape == (1, 3, 64, 64)
    assert torch.isfinite(latent).all()
    assert torch.isfinite(recon).all()


@pytest.mark.requires_models
@pytest.mark.skipif(not _FLUX2_VAE_PATH.exists(), reason="models/vae/flux2-vae.safetensors not present")
def test_real_flux2_vae_load_and_roundtrip():
    vae = load_vae(_FLUX2_VAE_PATH, disable_weight_init, device="cpu")
    vae.eval()
    assert vae.vae_type == "flux2_ae"
    assert vae.z_channels == 32

    pixels = torch.rand(1, 3, 64, 64) * 2.0 - 1.0
    with torch.no_grad():
        latent = vae.encode(pixels)
        recon = vae.decode(latent)

    # 8x conv downscale * 2x batchnorm pixel-unshuffle = 16x; 32*4=128 packed channels.
    assert latent.shape == (1, 128, 4, 4)
    assert recon.shape == (1, 3, 64, 64)
    assert torch.isfinite(latent).all()
    assert torch.isfinite(recon).all()

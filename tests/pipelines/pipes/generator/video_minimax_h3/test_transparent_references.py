from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.pipelines.pipes.generator.video_minimax_h3.conditioning import (
    ReferenceMedia,
    _pixels_from_image,
    keyframe_visual_cache_key,
    normalize_reference_video,
    normalize_references,
    prepare_keyframe_condition_rows,
    prepare_reference_conditioning,
    reference_visual_cache_key,
)

LATENT_CHANNELS = 24
MEAN = [0.0] * LATENT_CHANNELS
STD = [1.0] * LATENT_CHANNELS
RED = (255, 0, 0)
WHITE = (255, 255, 255)


class _RecordingVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.inputs: list = []

    def encode(self, x: torch.Tensor, *, sample_posterior: bool = False, generator=None) -> torch.Tensor:
        self.inputs.append(x.detach().clone())
        b, _c, f, h, w = x.shape
        return torch.zeros(b, LATENT_CHANNELS, f, h, w)


def _cutout(size=(32, 32), hidden=RED, subject=None) -> Image.Image:
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[..., :3] = hidden
    if subject is not None:
        pixels[8:24, 8:24, :3] = subject
        pixels[8:24, 8:24, 3] = 255
    return Image.fromarray(pixels, mode="RGBA")


def _reference_input(image: Image.Image) -> torch.Tensor:
    vae = _RecordingVae()
    prepare_reference_conditioning(
        [ReferenceMedia(kind="image", image=image)], vae_module=vae, patch_size=(1, 2, 2), device="cpu",
        dtype=torch.float32, latents_mean=MEAN, latents_std=STD, generator=torch.Generator().manual_seed(0),
    )
    assert len(vae.inputs) == 1
    return vae.inputs[0]


def _keyframe_input(image: Image.Image) -> torch.Tensor:
    vae = _RecordingVae()
    prepare_keyframe_condition_rows(
        [image], (0,), vae_module=vae, height=32, width=32, patch_size=(1, 2, 2), device="cpu",
        dtype=torch.float32, latents_mean=MEAN, latents_std=STD, generator=torch.Generator().manual_seed(0),
    )
    assert len(vae.inputs) == 1
    return vae.inputs[0]


_KEY_VAE = _RecordingVae()


def _cache_key(image: Image.Image):
    return reference_visual_cache_key(
        ReferenceMedia(kind="image", image=image), vae_module=_KEY_VAE, weight_revision=1,
        latents_mean=MEAN, latents_std=STD, device="cpu",
    )


def test_a_transparent_reference_encodes_white_not_its_hidden_red():
    encoded = _reference_input(_cutout())
    assert torch.equal(encoded, _reference_input(Image.new("RGB", (32, 32), WHITE)))
    assert not torch.equal(encoded, _reference_input(Image.new("RGB", (32, 32), RED)))


def test_a_transparent_keyframe_encodes_white_not_its_hidden_red():
    encoded = _keyframe_input(_cutout())
    assert torch.equal(encoded, _keyframe_input(Image.new("RGB", (32, 32), WHITE)))
    assert not torch.equal(encoded, _keyframe_input(Image.new("RGB", (32, 32), RED)))


def test_a_cutout_keeps_its_subject_and_whitens_only_the_transparent_part():
    flat = Image.new("RGB", (32, 32), WHITE)
    flat.paste((0, 0, 255), (8, 8, 24, 24))
    assert torch.equal(_reference_input(_cutout(subject=(0, 0, 255))), _reference_input(flat))


def test_a_resized_transparent_reference_still_encodes_white():
    normalized = normalize_references(
        [ReferenceMedia(kind="image", image=_cutout(size=(48, 48)))],
        num_frames=22, canvas_multiple=16, reference_short_edge=32,
    )
    assert normalized[0].image.size == (32, 32)
    assert torch.equal(_reference_input(normalized[0].image), _reference_input(Image.new("RGB", (32, 32), WHITE)))


@pytest.mark.parametrize("mode", ["RGB", "L", "P"])
def test_an_opaque_reference_encodes_exactly_what_it_did_before(mode):
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).convert(mode)
    vae = _RecordingVae()
    prepare_reference_conditioning(
        [ReferenceMedia(kind="image", image=image)], vae_module=vae, patch_size=(1, 2, 2), device="cpu",
        dtype=torch.float32, latents_mean=MEAN, latents_std=STD, generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(vae.inputs[0], _reference_input(image.convert("RGB")))


def test_a_fully_opaque_rgba_reference_matches_its_rgb_pixels():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    rgba = Image.fromarray(np.dstack([rgb, np.full((32, 32), 255, np.uint8)]), mode="RGBA")
    assert torch.equal(_reference_input(rgba), _reference_input(Image.fromarray(rgb)))


def test_the_latent_cache_keys_a_cutout_by_what_is_encoded():
    assert _cache_key(_cutout()) == _cache_key(Image.new("RGB", (32, 32), WHITE))
    assert _cache_key(_cutout()) != _cache_key(Image.new("RGB", (32, 32), RED))


def test_the_keyframe_cache_probe_keys_a_cutout_by_what_is_encoded():
    def key(image):
        return keyframe_visual_cache_key(
            image, 0, height=32, width=32, vae_module=_KEY_VAE, weight_revision=1,
            latents_mean=MEAN, latents_std=STD, device="cpu",
        )

    assert key(_cutout()) == key(Image.new("RGB", (32, 32), WHITE))
    assert key(_cutout()) != key(Image.new("RGB", (32, 32), RED))


def test_transparent_reference_video_frames_flatten_onto_white():
    frames = normalize_reference_video([_cutout()] * 4, fps=24.0, num_frames=4, canvas_multiple=16, short_edge=32)
    assert np.array_equal(frames, np.full((4, 32, 32, 3), 255, dtype=np.uint8))


def test_single_image_pixels_flatten_onto_white():
    assert torch.equal(_pixels_from_image(_cutout(), "cpu"), torch.full((1, 3, 1, 32, 32), 255, dtype=torch.uint8))

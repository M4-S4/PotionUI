from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.pipelines.pipes.latent_upscaler.minimax_h3.geometry import resolve_target_geometry
from src.pipelines.pipes.latent_upscaler.minimax_h3.main import LatentUpscalerMinimaxH3Pipe
from src.platform.runtime.native.engine import NativeEngineLoader
from src.platform.runtime.native.vae.minimax_h3_latent_upsampler import MEAN, STD

CHECKPOINT = (
    Path(__file__).resolve().parents[5] / "models" / "upscalers" / "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
)

pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="real MiniMax-H3 latent upscaler checkpoint not present")


def test_real_upsampler_keeps_a_unit_normal_vae_latent_in_range_through_the_pipe():
    upsampler = NativeEngineLoader(device="cpu")._load_latent_upscaler(CHECKPOINT)
    upsampler.module.float()
    upsampler.compute_dtype = torch.float32
    vae_module = SimpleNamespace(latents_mean=torch.tensor(MEAN), latents_std=torch.tensor(STD))
    mean = torch.tensor(MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(STD).view(1, -1, 1, 1, 1)

    torch.manual_seed(0)
    normalized = torch.randn(1, 24, 5, 16, 16)
    geometry = resolve_target_geometry(256, 256, mode="scale", scale=1.5)

    raw_out = LatentUpscalerMinimaxH3Pipe._upsample(upsampler, normalized * std + mean, geometry, "cpu", vae_module)
    out = (raw_out - mean) / std

    assert out.shape == (1, 24, 5, 24, 24)
    assert torch.isfinite(out).all()
    assert abs(out.mean().item()) < 0.25
    assert 0.4 < out.std().item() < 1.4
    assert out.abs().max().item() < 8.0

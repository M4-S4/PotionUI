from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import src.pipelines.pipes._shared.video.wan_experts as wan_experts
from src.pipelines.pipes._shared.video.wan_experts import ExpertRouter, decode_wan_video
from src.platform.runtime import cuda_allocator
from src.platform.runtime.native.memory.residency import GpuResidencyRegistry

LATENT_480P_5S = (1, 16, 21, 60, 104)


class _Expert:
    def __init__(self, gb):
        self.estimated_vram_gb = gb
        self.moved = []
        self.streamed = []
        self.module = lambda x, t, ctx, **kw: torch.zeros_like(x)

    def move_to(self, device):
        self.moved.append(device)

    def stream_to(self, device, budget):
        self.streamed.append((device, budget))

    def offload(self):
        pass


class _Vae:
    def __init__(self, module, gb=0.24):
        self.module = module
        self.compute_dtype = torch.float32
        self.estimated_vram_gb = gb
        self.offloads = 0

    def move_to(self, device):
        pass

    def offload(self):
        self.offloads += 1


def _decode_ctx(latent_channels, spatial_downscale):
    return SimpleNamespace(vae=_Vae(None), device="cpu", latent_channels=latent_channels,
                           spatial_downscale=spatial_downscale, spec=SimpleNamespace(latent_format={}))


def _place_high_expert(monkeypatch, expert, free_reads):
    reads = iter(free_reads)
    monkeypatch.setattr(wan_experts, "free_vram_gb", lambda device: next(reads))
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: next(reads))
    monkeypatch.setattr(wan_experts.torch.cuda, "is_available", lambda: False)
    router = ExpertRouter(expert, None, 0.875, "cuda:0", latents_shape=LATENT_480P_5S)
    router(torch.zeros(1, 16, 1, 2, 2), torch.tensor([0.95]), {"context": torch.zeros(1, 2, 4)})


def _fatal_driver_oom_policy(monkeypatch):
    monkeypatch.setattr(cuda_allocator, "_damage", None)
    monkeypatch.setattr(cuda_allocator, "_stop_growing_expandable_segments", lambda: None)
    monkeypatch.setenv(cuda_allocator.DRIVER_OOM_POLICY_ENV, "fatal")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _stream_failing(expert, error):
    def stream_to(device, budget):
        expert.streamed.append((device, budget))
        if budget > 0:
            raise error

    expert.stream_to = stream_to


def test_a_fatal_driver_oom_while_placing_an_expert_is_not_retried(monkeypatch):
    _fatal_driver_oom_policy(monkeypatch)
    monkeypatch.setattr(wan_experts, "get_residency_registry", GpuResidencyRegistry)
    expert = _Expert(13.0)
    _stream_failing(expert, RuntimeError("CUDA driver error: device not ready"))

    with pytest.raises(RuntimeError, match="device not ready"):
        _place_high_expert(monkeypatch, expert, [20.0, 20.0])

    assert len(expert.streamed) == 1
    assert expert.streamed[0][1] > 0
    assert cuda_allocator.allocator_damage() == "CUDA driver error: device not ready"


def test_a_retried_driver_oom_while_placing_an_expert_streams_it_fully(monkeypatch):
    monkeypatch.setattr(cuda_allocator, "_damage", None)
    monkeypatch.setenv(cuda_allocator.DRIVER_OOM_POLICY_ENV, "retry")
    monkeypatch.setattr(wan_experts, "get_residency_registry", GpuResidencyRegistry)
    expert = _Expert(13.0)
    _stream_failing(expert, RuntimeError("CUDA driver error: device not ready"))

    _place_high_expert(monkeypatch, expert, [20.0, 20.0])

    assert [budget for _, budget in expert.streamed][1:] == [0.0]
    assert cuda_allocator.allocator_damage() is None


def test_a_fatal_driver_oom_in_a_tiled_decode_is_not_retried(monkeypatch):
    _fatal_driver_oom_policy(monkeypatch)
    tiles = []

    def tiled(vae, z, *, tile_size, **kw):
        tiles.append(tile_size)
        raise RuntimeError("CUDA driver error: device not ready")

    cleared = []
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 0.001)
    monkeypatch.setattr(wan_experts, "causal3d_decode_tile", lambda latent_hw, free, **kw: 64)
    monkeypatch.setattr(wan_experts, "tiled_decode_causal3d", tiled)
    monkeypatch.setattr(wan_experts, "_clear_card_for_decode", cleared.append)

    with pytest.raises(RuntimeError, match="device not ready"):
        decode_wan_video(_decode_ctx(48, 16), torch.zeros(1, 48, 1, 4, 4))

    assert tiles == [64]
    assert cleared == []
    assert cuda_allocator.allocator_damage() == "CUDA driver error: device not ready"


def _sizing_spies(monkeypatch, need_gb):
    seen = {}

    def need(latent_hw, **kw):
        seen["need"] = kw
        return need_gb

    def tile(latent_hw, free, **kw):
        seen["tile"] = kw
        return 8

    def chunk(vae_module, latents, **kw):
        seen["chunk"] = kw
        return None

    monkeypatch.setattr(wan_experts, "causal3d_frame_need_gb", need)
    monkeypatch.setattr(wan_experts, "causal3d_decode_tile", tile)
    monkeypatch.setattr(wan_experts, "causal3d_chunk_frames", chunk)
    monkeypatch.setattr(wan_experts, "tiled_decode_causal3d", lambda vae, z, **kw: torch.zeros(1, 3, 1, 64, 64))
    monkeypatch.setattr(wan_experts, "_decode_latent", lambda vae_module, z, chunk_frames: torch.zeros(1, 3, 1, 64, 64))
    return seen


def test_an_8x_vae_keeps_the_calibrated_decode_cost(monkeypatch):
    seen = _sizing_spies(monkeypatch, need_gb=100.0)
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 1.0)

    decode_wan_video(_decode_ctx(16, 8), torch.zeros(1, 16, 1, 4, 4))

    assert seen["need"]["decode_mb_per_latent_px"] == 1.2
    assert seen["tile"]["decode_mb_per_latent_px"] == 1.2

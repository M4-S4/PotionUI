from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import src.pipelines.pipes._shared.video.wan_experts as wan_experts
from src.pipelines.pipes._shared.video.wan_experts import ExpertRouter, decode_wan_video
from src.platform.runtime import cuda_allocator
from src.platform.runtime.native.detect.registry import arch_registry
from src.platform.runtime.native.memory.residency import GpuResidencyRegistry
from src.platform.runtime.native.vae.causal_3d import AutoEncoderCausal3D
from src.platform.runtime.native.vae.tiling import causal3d_frame_need_gb
from vendor.gpl.comfyui.ops import disable_weight_init

LATENT_480P_5S = (1, 16, 21, 60, 104)
UMT5_FP8_GB = 6.27


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


class _TextEncoder:
    def __init__(self):
        self.offloads = 0

    def offload(self):
        self.offloads += 1


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


def _tiny_wan_vae() -> AutoEncoderCausal3D:
    with patch("src.platform.runtime.native.vae.causal_3d._DIM", 8):
        module = AutoEncoderCausal3D.from_config({}, disable_weight_init)
    module.eval()
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                p.normal_(std=0.02)
    return module


def _text_encoder_on_the_card(monkeypatch):
    registry = GpuResidencyRegistry()
    encoder = _TextEncoder()
    registry.note_resident(encoder, "cuda:0", UMT5_FP8_GB)
    monkeypatch.setattr(wan_experts, "get_residency_registry", lambda: registry)
    return registry, encoder


def _place_high_expert(monkeypatch, expert, free_reads):
    reads = iter(free_reads)
    monkeypatch.setattr(wan_experts, "free_vram_gb", lambda device: next(reads))
    monkeypatch.setattr(wan_experts.torch.cuda, "is_available", lambda: False)
    router = ExpertRouter(expert, None, 0.875, "cuda:0", latents_shape=LATENT_480P_5S)
    router(torch.zeros(1, 16, 1, 2, 2), torch.tensor([0.95]), {"context": torch.zeros(1, 2, 4)})


def test_a_text_encoder_left_on_a_12gb_card_is_evicted_before_a_small_expert_samples(monkeypatch):
    registry, encoder = _text_encoder_on_the_card(monkeypatch)
    expert = _Expert(2.64)

    _place_high_expert(monkeypatch, expert, [3.83, 10.53])

    assert encoder.offloads == 1
    assert registry.resident_gb("cuda:0") == 0.0
    assert expert.streamed == [("cuda:0", 0.0)]


def test_a_text_encoder_stays_when_the_expert_and_its_activations_still_fit_beside_it(monkeypatch):
    registry, encoder = _text_encoder_on_the_card(monkeypatch)
    expert = _Expert(2.64)

    _place_high_expert(monkeypatch, expert, [20.0, 20.0])

    assert encoder.offloads == 0
    assert registry.resident_gb("cuda:0") == pytest.approx(UMT5_FP8_GB)
    assert expert.moved == ["cuda:0"]


def test_the_vae_decode_evicts_a_text_encoder_that_leaves_no_room_for_one_latent_frame(monkeypatch):
    registry, encoder = _text_encoder_on_the_card(monkeypatch)
    vae = _Vae(None)
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 3.8)
    monkeypatch.setattr(wan_experts.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wan_experts.torch.cuda, "empty_cache", lambda: None)
    registry.note_resident(vae, "cuda:0", vae.estimated_vram_gb)

    wan_experts._make_room_for_decode(SimpleNamespace(vae=vae, device="cuda:0"), torch.zeros(LATENT_480P_5S))

    assert encoder.offloads == 1
    assert vae.offloads == 0
    assert registry.resident_gb("cuda:0") == pytest.approx(vae.estimated_vram_gb)


def test_the_vae_decode_keeps_a_text_encoder_that_leaves_room_for_one_latent_frame(monkeypatch):
    registry, encoder = _text_encoder_on_the_card(monkeypatch)
    need = causal3d_frame_need_gb((60, 104), vae_resident_gb=0.24)
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: need + 0.1)
    monkeypatch.setattr(wan_experts.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wan_experts.torch.cuda, "empty_cache", lambda: None)

    wan_experts._make_room_for_decode(SimpleNamespace(vae=_Vae(None), device="cuda:0"), torch.zeros(LATENT_480P_5S))

    assert encoder.offloads == 0


def test_a_clip_whose_single_latent_frame_overflows_is_decoded_in_spatial_tiles(monkeypatch):
    torch.manual_seed(0)
    vae_module = _tiny_wan_vae()
    latent = torch.randn(1, 16, 3, 4, 12)
    whole = decode_wan_video(SimpleNamespace(vae=_Vae(vae_module), device="cpu", latent_channels=16), latent.clone())
    real_tiled = wan_experts.tiled_decode_causal3d
    seen = []

    def spy(vae, z, **kw):
        seen.append(kw)
        return real_tiled(vae, z, **kw)

    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 0.001)
    monkeypatch.setattr(wan_experts, "causal3d_decode_tile", lambda latent_hw, free, **kw: 8)
    monkeypatch.setattr(wan_experts, "tiled_decode_causal3d", spy)
    monkeypatch.setattr(wan_experts, "chunked_decode_causal3d", Mock(side_effect=AssertionError("chunked")))
    c = SimpleNamespace(vae=_Vae(vae_module), device="cpu", latent_channels=16)

    out = decode_wan_video(c, latent.clone())

    assert seen == [{"tile_size": 8, "overlap": 4, "scale": 8, "accumulate_device": "cpu", "chunk_latent_frames": 1}]
    assert out.shape == whole.shape
    assert c.vae.offloads == 1


def test_each_tile_of_a_long_clip_is_decoded_one_latent_frame_at_a_time(monkeypatch):
    torch.manual_seed(0)
    vae_module = _tiny_wan_vae()
    latent = torch.randn(1, 16, 5, 4, 12)
    whole = decode_wan_video(SimpleNamespace(vae=_Vae(vae_module), device="cpu", latent_channels=16), latent.clone())
    real_decode = vae_module.decode
    frames = []

    def spy(z, **kw):
        frames.append(z.shape[2])
        return real_decode(z, **kw)

    vae_module.decode = spy
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 0.001)
    monkeypatch.setattr(wan_experts, "causal3d_decode_tile", lambda latent_hw, free, **kw: 8)

    out = decode_wan_video(SimpleNamespace(vae=_Vae(vae_module), device="cpu", latent_channels=16), latent.clone())

    assert frames == [1] * 10
    assert out.shape == whole.shape


def test_a_clip_whose_single_latent_frame_fits_is_not_tiled(monkeypatch):
    vae_module = _tiny_wan_vae()
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: causal3d_frame_need_gb((4, 4)))
    monkeypatch.setattr(wan_experts, "tiled_decode_causal3d", Mock(side_effect=AssertionError("tiled")))

    decode_wan_video(SimpleNamespace(vae=_Vae(vae_module), device="cpu", latent_channels=16), torch.randn(1, 16, 3, 4, 4))


def test_the_tiles_follow_the_spatial_factor_of_a_wan22_vae(monkeypatch):
    scales = []

    def tiled(vae, z, **kw):
        scales.append(kw["scale"])
        return torch.zeros(1, 3, 1, 64, 64)

    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 0.001)
    monkeypatch.setattr(wan_experts, "tiled_decode_causal3d", tiled)
    c = SimpleNamespace(vae=_Vae(None), device="cpu", latent_channels=48, spatial_downscale=16)

    decode_wan_video(c, torch.zeros(1, 48, 1, 4, 4))

    assert scales == [16]


def test_a_tiled_decode_short_of_vram_clears_the_card_and_shrinks_its_tile(monkeypatch):
    tiles = []

    def tiled(vae, z, *, tile_size, **kw):
        tiles.append(tile_size)
        if tile_size > 16:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return torch.zeros(1, 3, 1, 32, 32)

    cleared = []
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 0.001)
    monkeypatch.setattr(wan_experts, "causal3d_decode_tile", lambda latent_hw, free, **kw: 64)
    monkeypatch.setattr(wan_experts, "tiled_decode_causal3d", tiled)
    monkeypatch.setattr(wan_experts, "_clear_card_for_decode", cleared.append)
    c = SimpleNamespace(vae=_Vae(None), device="cpu", latent_channels=48)

    decode_wan_video(c, torch.zeros(1, 48, 1, 4, 4))

    assert tiles == [64, 32, 16]
    assert cleared == [c, c]


@pytest.mark.parametrize(
    ("first_tile", "error"),
    [
        (16, torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")),
        (64, RuntimeError("CUDA error: an illegal memory access was encountered")),
    ],
)
def test_a_tiled_decode_raises_at_the_smallest_tile_or_on_an_error_not_about_vram(monkeypatch, first_tile, error):
    tiles = []

    def tiled(vae, z, *, tile_size, **kw):
        tiles.append(tile_size)
        raise error

    cleared = []
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 0.001)
    monkeypatch.setattr(wan_experts, "causal3d_decode_tile", lambda latent_hw, free, **kw: first_tile)
    monkeypatch.setattr(wan_experts, "tiled_decode_causal3d", tiled)
    monkeypatch.setattr(wan_experts, "_clear_card_for_decode", cleared.append)

    with pytest.raises(type(error)):
        decode_wan_video(SimpleNamespace(vae=_Vae(None), device="cpu", latent_channels=48), torch.zeros(1, 48, 1, 4, 4))

    assert tiles == [first_tile]
    assert cleared == []


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


def test_a_fatal_driver_oom_in_the_decode_is_not_retried(monkeypatch):
    _fatal_driver_oom_policy(monkeypatch)
    vae_module = _tiny_wan_vae()
    frames_per_call = []

    def decode_failing(z, **kw):
        frames_per_call.append(z.shape[2])
        raise RuntimeError("CUDA driver error: device not ready")

    vae_module.decode = decode_failing
    cleared = []
    monkeypatch.setattr(wan_experts, "_clear_card_for_decode", cleared.append)

    with pytest.raises(RuntimeError, match="device not ready"):
        decode_wan_video(SimpleNamespace(vae=_Vae(vae_module), device="cpu", latent_channels=16), torch.randn(1, 16, 5, 4, 4))

    assert frames_per_call == [5]
    assert cleared == []
    assert cuda_allocator.allocator_damage() == "CUDA driver error: device not ready"


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
        decode_wan_video(SimpleNamespace(vae=_Vae(None), device="cpu", latent_channels=48), torch.zeros(1, 48, 1, 4, 4))

    assert tiles == [64]
    assert cleared == []
    assert cuda_allocator.allocator_damage() == "CUDA driver error: device not ready"


def test_a_decode_the_driver_fails_for_lack_of_vram_is_retried_one_latent_frame_at_a_time(monkeypatch):
    monkeypatch.setattr(cuda_allocator, "_damage", None)
    monkeypatch.setenv(cuda_allocator.DRIVER_OOM_POLICY_ENV, "retry")
    torch.manual_seed(0)
    vae_module = _tiny_wan_vae()
    latent = torch.randn(1, 16, 5, 4, 4)
    expected = decode_wan_video(SimpleNamespace(vae=_Vae(vae_module), device="cpu", latent_channels=16), latent.clone())
    real_decode = vae_module.decode
    frames_per_call = []

    def decode_failing_once(z, **kw):
        frames_per_call.append(z.shape[2])
        if len(frames_per_call) == 1:
            raise RuntimeError("CUDA driver error: device not ready")
        return real_decode(z, **kw)

    vae_module.decode = decode_failing_once
    cleared = []
    monkeypatch.setattr(wan_experts, "_clear_card_for_decode", cleared.append)
    c = SimpleNamespace(vae=_Vae(vae_module), device="cpu", latent_channels=16)

    out = decode_wan_video(c, latent.clone())

    assert frames_per_call == [5, 1, 1, 1, 1, 1]
    assert cleared == [c]
    assert np.array_equal(out, expected)
    assert c.vae.offloads == 1


def test_a_decode_that_hits_an_out_of_memory_error_is_retried_too(monkeypatch):
    vae_module = _tiny_wan_vae()
    real_decode = vae_module.decode
    calls = []

    def decode_failing_once(z, **kw):
        calls.append(z.shape[2])
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return real_decode(z, **kw)

    vae_module.decode = decode_failing_once
    monkeypatch.setattr(wan_experts, "_clear_card_for_decode", lambda c: None)

    decode_wan_video(SimpleNamespace(vae=_Vae(vae_module), device="cpu", latent_channels=16), torch.randn(1, 16, 3, 4, 4))

    assert calls == [3, 1, 1, 1]


def test_a_decode_error_that_is_not_about_vram_propagates_without_a_retry(monkeypatch):
    vae_module = _tiny_wan_vae()
    cleared = []
    monkeypatch.setattr(wan_experts, "_clear_card_for_decode", cleared.append)

    with patch.object(vae_module, "decode", side_effect=RuntimeError("CUDA error: an illegal memory access was encountered")):
        with pytest.raises(RuntimeError, match="illegal memory access"):
            decode_wan_video(SimpleNamespace(vae=_Vae(vae_module), device="cpu", latent_channels=16), torch.randn(1, 16, 3, 4, 4))

    assert cleared == []


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


def test_an_undeclared_16x_vae_prices_its_tiled_decode_four_times_dearer_per_latent_px(monkeypatch):
    seen = _sizing_spies(monkeypatch, need_gb=100.0)
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 1.0)

    decode_wan_video(SimpleNamespace(vae=_Vae(None), device="cpu", latent_channels=48, spatial_downscale=16), torch.zeros(1, 48, 1, 4, 4))

    assert seen["need"]["decode_mb_per_latent_px"] == pytest.approx(4.8)
    assert seen["tile"]["decode_mb_per_latent_px"] == pytest.approx(4.8)
    assert "chunk" not in seen


def test_an_undeclared_16x_vae_prices_its_temporal_chunks_by_the_same_cost(monkeypatch):
    seen = _sizing_spies(monkeypatch, need_gb=0.0)
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 1.0)

    decode_wan_video(SimpleNamespace(vae=_Vae(None), device="cpu", latent_channels=48, spatial_downscale=16), torch.zeros(1, 48, 3, 4, 4))

    assert seen["chunk"]["decode_mb_per_latent_px"] == pytest.approx(4.8)
    assert "tile" not in seen


def test_an_8x_vae_keeps_the_calibrated_decode_cost(monkeypatch):
    seen = _sizing_spies(monkeypatch, need_gb=100.0)
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 1.0)

    decode_wan_video(SimpleNamespace(vae=_Vae(None), device="cpu", latent_channels=16), torch.zeros(1, 16, 1, 4, 4))

    assert seen["need"]["decode_mb_per_latent_px"] == 1.2
    assert seen["tile"]["decode_mb_per_latent_px"] == 1.2


def test_a_spec_that_declares_its_measured_decode_cost_is_believed_over_the_spatial_factor(monkeypatch):
    seen = _sizing_spies(monkeypatch, need_gb=100.0)
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 1.0)
    spec = SimpleNamespace(latent_format={"spatial_downscale": 16, "decode_mb_per_latent_px": 2.0})

    decode_wan_video(SimpleNamespace(vae=_Vae(None), device="cpu", latent_channels=48, spatial_downscale=16, spec=spec), torch.zeros(1, 48, 1, 4, 4))

    assert seen["need"]["decode_mb_per_latent_px"] == 2.0
    assert seen["tile"]["decode_mb_per_latent_px"] == 2.0


def test_making_room_before_an_undeclared_16x_decode_asks_for_the_dearer_frame(monkeypatch):
    seen = _sizing_spies(monkeypatch, need_gb=3.0)
    asked = []
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 10.0)
    monkeypatch.setattr(wan_experts.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wan_experts, "get_residency_registry",
                        lambda: SimpleNamespace(ensure_free=lambda device, need, free, exclude: asked.append(need) or []))

    wan_experts._make_room_for_decode(SimpleNamespace(vae=_Vae(None), device="cuda:0", spatial_downscale=16), torch.zeros(1, 48, 1, 4, 4))

    assert seen["need"]["decode_mb_per_latent_px"] == pytest.approx(4.8)
    assert seen["need"]["vae_resident_gb"] == 0.24
    assert asked == [3.0]


def _the_5b_conductor(module=None):
    return SimpleNamespace(vae=_Vae(module), device="cpu", latent_channels=48, spatial_downscale=16,
                           spec=arch_registry.get("wan", "wan_ti2v_5b"))


def test_the_5b_prices_its_decode_by_the_cost_measured_for_its_vae(monkeypatch):
    seen = _sizing_spies(monkeypatch, need_gb=100.0)
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 1.0)

    decode_wan_video(_the_5b_conductor(), torch.zeros(1, 48, 1, 4, 4))

    assert seen["need"]["decode_mb_per_latent_px"] == 2.6
    assert seen["tile"]["decode_mb_per_latent_px"] == 2.6


def test_the_5b_prices_its_temporal_chunks_by_the_measured_cost(monkeypatch):
    seen = _sizing_spies(monkeypatch, need_gb=0.0)
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 1.0)

    decode_wan_video(_the_5b_conductor(), torch.zeros(1, 48, 3, 4, 4))

    assert seen["chunk"]["decode_mb_per_latent_px"] == 2.6


def test_making_room_before_a_5b_decode_asks_for_the_measured_frame(monkeypatch):
    seen = _sizing_spies(monkeypatch, need_gb=3.0)
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 10.0)
    monkeypatch.setattr(wan_experts.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wan_experts, "get_residency_registry",
                        lambda: SimpleNamespace(ensure_free=lambda device, need, free, exclude: []))
    c = _the_5b_conductor()
    c.device = "cuda:0"

    wan_experts._make_room_for_decode(c, torch.zeros(1, 48, 1, 4, 4))

    assert seen["need"]["decode_mb_per_latent_px"] == 2.6


@pytest.mark.parametrize("latent_hw,free_gb,decoded", [
    ((30, 52), 9.4, ("whole", 1)),
    ((30, 52), 5.5, ("tiles", 24)),
    ((44, 80), 20.5, ("whole", 1)),
    ((44, 80), 9.4, ("tiles", 40)),
    ((44, 80), 5.5, ("tiles", 32)),
])
def test_the_5b_decodes_a_121_frame_clip_whole_or_in_tiles_by_its_measured_cost(monkeypatch, latent_hw, free_gb, decoded):
    calls = []
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: free_gb)
    monkeypatch.setattr(wan_experts, "_decode_latent",
                        lambda vae_module, z, chunk_frames: calls.append(("whole", chunk_frames)) or z)
    monkeypatch.setattr(wan_experts, "tiled_decode_causal3d",
                        lambda vae, z, **kw: calls.append(("tiles", kw["tile_size"])) or z)

    wan_experts._decode_pixels(_the_5b_conductor(Mock(spec=["new_feat_cache"])), torch.zeros(1, 48, 31, *latent_hw))

    assert calls == [decoded]

def test_the_host_assembly_of_a_long_1080p_clip_is_priced_as_the_half_precision_accumulator_plus_the_result():
    pixels = 3 * (1 + 4 * 40) * 1088 * 1920

    assert wan_experts.host_assembly_gb((1, 16, 41, 136, 240), 8, 2) == pytest.approx(pixels * 4 / 1024 ** 3)
    assert wan_experts.host_assembly_gb((1, 48, 21, 44, 80), 16, 4) == pytest.approx(3 * 81 * 704 * 1280 * 6 / 1024 ** 3)
    assert 3.5 < wan_experts.host_assembly_gb((1, 16, 41, 136, 240), 8, 2) < 4.0


def _tiled_decode_of_a_long_1080p_clip(monkeypatch, available_gb):
    monkeypatch.setattr(wan_experts, "effective_free_vram_gb", lambda device: 0.001)
    monkeypatch.setattr(wan_experts, "causal3d_decode_tile", lambda latent_hw, free, **kw: 8)
    monkeypatch.setattr(wan_experts, "tiled_decode_causal3d", lambda vae, z, **kw: torch.zeros(1, 3, 1, 64, 64))
    monkeypatch.setattr(wan_experts, "get_system_memory", lambda: SimpleNamespace(available_gb=available_gb, total_gb=32.0))
    decode_wan_video(SimpleNamespace(vae=_Vae(None), device="cpu", latent_channels=16), torch.zeros(1, 16, 41, 136, 240))


def test_a_tiled_decode_warns_when_the_assembled_video_outgrows_the_free_host_ram_and_goes_on(monkeypatch, caplog):
    with caplog.at_level("INFO", logger=wan_experts.logger.name):
        _tiled_decode_of_a_long_1080p_clip(monkeypatch, available_gb=0.5)

    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "assembling the 161-frame video on the host needs about 5.6GB of RAM and 0.5GB are available" in warnings[0]
    assert "the decode goes on" in warnings[0]


def test_a_tiled_decode_that_fits_the_host_ram_only_notes_the_estimate(monkeypatch, caplog):
    with caplog.at_level("INFO", logger=wan_experts.logger.name):
        _tiled_decode_of_a_long_1080p_clip(monkeypatch, available_gb=20.0)

    assert not [record for record in caplog.records if record.levelname == "WARNING"]
    assert any("needs about 5.6GB of RAM (20.0GB available)" in record.getMessage() for record in caplog.records)

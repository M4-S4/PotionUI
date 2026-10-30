from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from src.pipelines.pipes._shared.vae import minimax_h3_encode as h3e
from src.platform.runtime.native.kernels import disable_weight_init, manual_cast
from src.platform.runtime.native.vae.minimax_h3_video import MiniMaxH3VideoVAE

_ENC = "src.pipelines.pipes._shared.vae.minimax_h3_encode"

_TINY = dict(
    latent_channels=4,
    block_out_channels=(8, 16),
    layers_per_block=1,
    spatial_downsample_factors=(2, 2),
    temporal_downsample_factors=(1, 2),
    norm_num_groups=2,
    decoder_num_layers=1,
    decoder_num_attention_heads=2,
    decoder_attention_head_dim=8,
    decoder_num_register_tokens=2,
    decoder_ffn_mult=2,
    clip_length=5,
    token_drop=1,
)


class _Recorder:
    def __init__(self):
        self.marks = []

    def mark(self, event, **fields):
        self.marks.append((event, fields))

    def events(self, name):
        return [fields for event, fields in self.marks if event == name]


def _vae(ops=disable_weight_init) -> MiniMaxH3VideoVAE:
    torch.manual_seed(0)
    module = MiniMaxH3VideoVAE.from_config(_TINY, ops)
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                p.normal_(std=0.2)
    return module


def _real_widths() -> MiniMaxH3VideoVAE:
    with torch.device("meta"):
        return MiniMaxH3VideoVAE.from_config({}, disable_weight_init)


class TestEncodeComputeDtype:
    def test_a_cast_weight_encoder_runs_in_fp32_like_the_reference(self):
        assert h3e.encode_compute_dtype(_vae(manual_cast)) == torch.float32

    def test_a_plain_encoder_runs_in_its_own_weight_dtype(self):
        module = _vae().to(torch.bfloat16)
        assert h3e.encode_compute_dtype(module) == torch.bfloat16


class TestBudget:
    def test_one_reference_tile_at_the_real_widths_is_a_few_gigabytes_in_fp32(self):
        module = _real_widths()
        gb = h3e.encode_tile_gb(module, 256, 1344, 768, torch.float32)
        assert 2.0 < gb < 6.0
        assert h3e.encode_tile_gb(module, 128, 1344, 768, torch.float32) < gb / 3

    def test_activation_need_grows_with_the_canvas(self):
        module = _real_widths()
        small = h3e.encode_activation_gb(module, 256, 256, torch.float32)
        big = h3e.encode_activation_gb(module, 1344, 768, torch.float32)
        assert big > small

    def test_without_a_vram_reading_the_plan_is_the_reference_tile_one_at_a_time(self):
        with patch(f"{_ENC}.free_vram_gb", return_value=None):
            plan = h3e.plan_encode(_real_widths(), 1344, 768, torch.float32, "cpu")
        assert plan["tile_px"] == 256
        assert plan["batch_size"] == 1

    def test_plenty_of_room_keeps_the_reference_tile_and_batches_several(self):
        with patch(f"{_ENC}.free_vram_gb", return_value=40.0):
            plan = h3e.plan_encode(_real_widths(), 1344, 768, torch.float32, "cuda")
        assert plan["tile_px"] == 256
        assert plan["batch_size"] > 1

    def test_too_little_room_for_a_reference_tile_drops_to_a_smaller_one(self):
        module = _real_widths()
        need_128 = h3e.encode_tile_gb(module, 128, 1344, 768, torch.float32)
        need_192 = h3e.encode_tile_gb(module, 192, 1344, 768, torch.float32)
        pixels = h3e.encode_clip_pixels_gb(module, 1344, 768, torch.float32)
        free = (pixels + (need_128 + need_192) / 2) / h3e.ENCODE_VRAM_BUDGET_FRACTION
        with patch(f"{_ENC}.free_vram_gb", return_value=free):
            plan = h3e.plan_encode(module, 1344, 768, torch.float32, "cuda")
        assert plan["tile_px"] == 128
        assert plan["batch_size"] == 1

    def test_no_room_at_all_still_plans_the_smallest_tile(self):
        with patch(f"{_ENC}.free_vram_gb", return_value=0.1):
            plan = h3e.plan_encode(_real_widths(), 1344, 768, torch.float32, "cuda")
        assert plan["tile_px"] == h3e.ENCODE_TILE_PX_LADDER[-1]


class TestEncodeVideo:
    def test_matches_a_direct_encode_and_restores_the_module_tiling(self):
        module = _vae()
        x = torch.rand(1, 3, 10, 64, 48) * 2 - 1
        with torch.no_grad():
            direct = module.encode(x)
        recorder = _Recorder()
        with patch(f"{_ENC}.get_profiler", return_value=recorder):
            latent = h3e.encode_video(module, x, "cpu")
        assert torch.equal(latent, direct)
        assert (module.use_tiling, module.tile_sample_min_height, module.encode_tile_batch_size) == (True, 256, 1)
        (mark,) = recorder.events("minimax_h3.encode")
        assert mark["tile_px"] == 256
        assert mark["retries"] == []
        assert mark["latent_frames"] == latent.shape[2]

    def test_oom_steps_down_the_tile_ladder_and_reports_it(self):
        module = _vae()
        real = module.encode
        tried = []

        def encode(pixels):
            tried.append(module.tile_sample_min_height)
            if module.tile_sample_min_height > 128:
                raise torch.cuda.OutOfMemoryError("boom")
            return real(pixels)

        module.encode = encode
        recorder = _Recorder()
        with patch(f"{_ENC}.get_profiler", return_value=recorder):
            h3e.encode_video(module, torch.rand(1, 3, 5, 32, 32), "cpu")
        assert tried == [256, 192, 128]
        (mark,) = recorder.events("minimax_h3.encode")
        assert mark["retries"] == ["oom@256", "oom@192"]
        assert module.tile_sample_min_height == 256

    def test_free_more_gets_one_chance_before_the_tile_shrinks(self):
        module = _vae()
        real = module.encode
        state = {"freed": False}
        tried = []

        def encode(pixels):
            tried.append(module.tile_sample_min_height)
            if not state["freed"]:
                raise torch.cuda.OutOfMemoryError("boom")
            return real(pixels)

        def free_more():
            state["freed"] = True
            return True

        module.encode = encode
        with patch(f"{_ENC}.get_profiler", return_value=_Recorder()):
            h3e.encode_video(module, torch.rand(1, 3, 5, 32, 32), "cpu", free_more=free_more)
        assert tried == [256, 256]

    def test_running_out_at_the_smallest_tile_raises_a_clear_error_and_restores_tiling(self):
        module = _vae()

        def encode(pixels):
            module.encode_tile_batch_size = 99
            raise torch.cuda.OutOfMemoryError("boom")

        module.encode = encode
        with patch(f"{_ENC}.get_profiler", return_value=_Recorder()):
            with pytest.raises(torch.cuda.OutOfMemoryError, match="even at 128px tiles"):
                h3e.encode_video(module, torch.rand(1, 3, 5, 32, 32), "cpu")
        assert (module.tile_sample_min_height, module.encode_tile_batch_size) == (256, 1)

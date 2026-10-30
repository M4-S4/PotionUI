from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.platform.runtime.native.kernels import disable_weight_init
from src.platform.runtime.native.vae.minimax_h3_video import MiniMaxH3VideoVAE

TILED_ENCODE_REL_TOL = 0.15

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
    tile_sample_min_height=64,
    tile_sample_min_width=64,
    tile_sample_min_overlap_height=32,
    tile_sample_min_overlap_width=32,
)

_REAL_GEOMETRY = dict(
    _TINY,
    block_out_channels=(8, 8, 8, 8),
    spatial_downsample_factors=(2, 2, 2, 2),
    temporal_downsample_factors=(1, 2, 1, 1),
    tile_sample_min_height=256,
    tile_sample_min_width=256,
    tile_sample_min_overlap_height=64,
    tile_sample_min_overlap_width=64,
)


def _build(config: dict = _TINY, seed: int = 0) -> MiniMaxH3VideoVAE:
    torch.manual_seed(seed)
    module = MiniMaxH3VideoVAE.from_config(config, disable_weight_init)
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                p.normal_(std=0.2)
    return module


class _PoolEncoder(torch.nn.Module):
    def __init__(self, ratio: int, out_channels: int):
        super().__init__()
        self.ratio = ratio
        self.out_channels = out_channels

    def forward(self, x):
        pooled = F.avg_pool3d(x[:, :, ::2], kernel_size=(1, self.ratio, self.ratio))
        return pooled.repeat(1, self.out_channels // 3 + 1, 1, 1, 1)[:, : self.out_channels]


def _pointwise(module: MiniMaxH3VideoVAE) -> MiniMaxH3VideoVAE:
    conv_in = module.encoder.conv_in
    module.encoder = _PoolEncoder(module.spatial_compression_ratio, module.quant_conv.in_channels)
    module.encoder.conv_in = conv_in
    return module


class TestSplitTilesAtEdges:
    @pytest.mark.parametrize("length", [256, 288, 320, 768, 1088, 1344, 1952])
    def test_tiles_cover_the_frame_exactly_with_latent_aligned_overlaps(self, length):
        module = MiniMaxH3VideoVAE.from_config(_REAL_GEOMETRY, disable_weight_init)
        starts, sizes, overlaps = module._split_tiles(length, 256, 64)

        assert starts[0] == 0
        assert starts[-1] + sizes[-1] == length
        assert all(size == min(256, length) for size in sizes)
        assert all(o >= 64 and (o - 64) % 16 == 0 for o in overlaps)
        for i, overlap in enumerate(overlaps):
            assert starts[i + 1] == starts[i] + sizes[i] - overlap
        latent_kept = sum(s // 16 for s in sizes) - sum(o // 16 for o in overlaps)
        assert latent_kept == length // 16

    @pytest.mark.parametrize("tile_px", [256, 192, 128])
    def test_every_fallback_tile_size_still_covers_a_real_canvas(self, tile_px):
        module = MiniMaxH3VideoVAE.from_config(_REAL_GEOMETRY, disable_weight_init)
        for length in (768, 1344):
            starts, sizes, overlaps = module._split_tiles(length, tile_px, 64)
            assert starts[-1] + sizes[-1] == length
            assert sum(sizes) - sum(overlaps) == length

    def test_encode_tile_grid_matches_the_split(self):
        module = MiniMaxH3VideoVAE.from_config(_REAL_GEOMETRY, disable_weight_init)
        assert module.encode_tile_grid(1344, 768) == (
            len(module._split_tiles(1344, 256, 64)[0]), len(module._split_tiles(768, 256, 64)[0]),
        )
        module.use_tiling = False
        assert module.encode_tile_grid(1344, 768) == (1, 1)


class TestTiledEncode:
    @pytest.mark.parametrize("height,width", [(128, 96), (160, 64), (96, 192)])
    def test_stitching_is_exact_for_a_translation_equivariant_encoder(self, height, width):
        module = _pointwise(_build())
        torch.manual_seed(1)
        x = torch.rand(1, 3, 10, height, width) * 2 - 1
        with torch.no_grad():
            module.use_tiling = False
            full = module.encode(x)
            module.use_tiling = True
            tiled = module.encode(x)
        assert tiled.shape == full.shape
        assert torch.allclose(tiled, full, atol=1e-6)

    def test_tiled_encode_approximates_the_full_encode_on_the_real_tiny_arch(self):
        module = _build()
        torch.manual_seed(2)
        x = torch.rand(1, 3, 5, 128, 96) * 2 - 1
        with torch.no_grad():
            module.use_tiling = False
            full = module.encode(x)
            module.use_tiling = True
            tiled = module.encode(x)
            swapped = torch.cat([tiled[..., tiled.shape[-2] // 2 :, :], tiled[..., : tiled.shape[-2] // 2, :]], dim=-2)
        assert tiled.shape == full.shape
        rel = ((tiled - full).norm() / full.norm()).item()
        misplaced = ((swapped - full).norm() / full.norm()).item()
        assert rel < TILED_ENCODE_REL_TOL
        assert misplaced > 3 * rel

    def test_a_single_tile_covering_the_frame_is_the_full_encode(self):
        module = _build()
        x = torch.rand(1, 3, 5, 64, 48) * 2 - 1
        with torch.no_grad():
            module.use_tiling = False
            full = module.encode(x)
            module.use_tiling = True
            tiled = module.encode(x)
        assert torch.equal(tiled, full)

    def test_batched_tiles_match_one_tile_at_a_time(self):
        module = _build()
        x = torch.rand(1, 3, 10, 128, 96) * 2 - 1
        with torch.no_grad():
            sequential = module.encode(x)
            module.encode_tile_batch_size = 8
            batched = module.encode(x)
        assert torch.allclose(batched, sequential, atol=1e-5)

    def test_smaller_fallback_tiles_stay_close_to_the_reference_tile(self):
        module = _build()
        x = torch.rand(1, 3, 5, 128, 96) * 2 - 1
        with torch.no_grad():
            module.use_tiling = False
            full = module.encode(x)
            module.use_tiling = True
            module.tile_sample_min_height = module.tile_sample_min_width = 48
            module.tile_sample_min_overlap_height = module.tile_sample_min_overlap_width = 16
            small = module.encode(x)
        assert small.shape == full.shape
        assert ((small - full).norm() / full.norm()).item() < 2 * TILED_ENCODE_REL_TOL

    def test_clips_reach_the_encoder_one_clip_at_a_time_on_its_device(self):
        module = _build()
        seen = []
        real = module._encode_clip

        def spy(clip):
            seen.append((clip.device.type, clip.shape[2]))
            return real(clip)

        module._encode_clip = spy
        x = torch.rand(1, 3, 12, 64, 48) * 2 - 1
        with torch.no_grad():
            module.encode(x)
        assert seen == [("cpu", 5), ("cpu", 5), ("cpu", 5)]

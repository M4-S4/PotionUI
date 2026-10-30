from __future__ import annotations

from typing import Any

import torch

ENCODE_LIVE_FULL_RES_TENSORS = 6
_BYTES_PER_GB = 1 << 30
_CAUSAL_TEMPORAL_PAD = 2
_REFLECT_PAD = 2


def encode_compute_dtype(vae_module: Any) -> torch.dtype:
    if getattr(vae_module.encoder.conv_in, "comfy_cast_weights", False):
        return torch.float32
    for parameter in vae_module.encoder.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _element_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def encode_bytes_per_voxel(vae_module: Any, dtype: torch.dtype) -> int:
    conv_in = vae_module.encoder.conv_in
    elements = ENCODE_LIVE_FULL_RES_TENSORS * conv_in.out_channels + conv_in.in_channels
    return _element_bytes(dtype) * elements


def encode_tile_voxels(vae_module: Any, tile_px: int, height: int, width: int) -> int:
    tile_h = height if tile_px <= 0 else min(height, tile_px)
    tile_w = width if tile_px <= 0 else min(width, tile_px)
    frames = vae_module.clip_length + _CAUSAL_TEMPORAL_PAD
    return frames * (tile_h + _REFLECT_PAD) * (tile_w + _REFLECT_PAD)


def encode_tile_gb(vae_module: Any, tile_px: int, height: int, width: int, dtype: torch.dtype) -> float:
    voxels = encode_tile_voxels(vae_module, tile_px, height, width)
    return voxels * encode_bytes_per_voxel(vae_module, dtype) / _BYTES_PER_GB


def encode_clip_pixels_gb(vae_module: Any, height: int, width: int, dtype: torch.dtype) -> float:
    return vae_module.clip_length * 3 * height * width * _element_bytes(dtype) / _BYTES_PER_GB


def module_tile_px(vae_module: Any) -> int:
    return vae_module.tile_sample_min_height if vae_module.use_tiling else 0


def encode_activation_gb(vae_module: Any, height: int, width: int, dtype: torch.dtype) -> float:
    tile_gb = encode_tile_gb(vae_module, module_tile_px(vae_module), height, width, dtype)
    return tile_gb + encode_clip_pixels_gb(vae_module, height, width, dtype)

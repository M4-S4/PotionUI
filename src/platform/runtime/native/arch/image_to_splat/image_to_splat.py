from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

from src.platform.runtime.native.errors import SamplingCancelled

from .config import DEFAULT_DETAIL, DETAIL_LONG_SIDE, MOGE_MASK_THRESHOLD, MOGE_RESOLUTION_LEVEL
from .lift import LiftedSplat, lift_to_splat

__all__ = ["STAGE_ESTIMATE", "STAGE_LIFT", "detail_long_side", "prepare_image", "run_image_to_splat"]

STAGE_ESTIMATE = "estimate"
STAGE_LIFT = "lift"

_NO_GEOMETRY = (
    "MoGe-2 recovered no geometry from this image: too little of it reads as a surface with depth. "
    "Try a photograph or render of a scene rather than a flat graphic."
)


def detail_long_side(detail: str) -> int:
    if detail not in DETAIL_LONG_SIDE:
        raise ValueError(f"unknown detail {detail!r}; expected one of {list(DETAIL_LONG_SIDE)}")
    return DETAIL_LONG_SIDE[detail]


def prepare_image(image: Image.Image, detail: str = DEFAULT_DETAIL) -> Image.Image:
    rgb = image.convert("RGB")
    long_side = detail_long_side(detail)
    scale = long_side / max(rgb.size)
    if scale >= 1.0:
        return rgb
    size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    return rgb.resize(size, Image.LANCZOS)


def _release_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def _on_device(module, device):
    module.to(device)
    try:
        yield module
    finally:
        module.to("cpu")
        _release_cuda()


@torch.no_grad()
def run_image_to_splat(
    estimator,
    image: Image.Image,
    *,
    detail: str = DEFAULT_DETAIL,
    device: str = "cuda",
    resolution_level: int = MOGE_RESOLUTION_LEVEL,
    on_stage: Optional[Callable[[str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> LiftedSplat:
    source = prepare_image(image, detail)
    pixels = torch.from_numpy(np.asarray(source, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    if on_stage is not None:
        on_stage(STAGE_ESTIMATE)
    with _on_device(estimator, device) as model:
        [geometry] = model.estimate_geometry(pixels, resolution_level, MOGE_MASK_THRESHOLD)
    if geometry is None:
        raise ValueError(_NO_GEOMETRY)
    fx, fy = geometry.focal_pixels()
    depth = geometry.depth.float().cpu().numpy()
    valid = geometry.mask.cpu().numpy()
    del geometry, pixels
    _release_cuda()
    if is_cancelled is not None and is_cancelled():
        raise SamplingCancelled()
    if on_stage is not None:
        on_stage(STAGE_LIFT)
    return lift_to_splat(np.asarray(source), depth, valid, fx, fy)

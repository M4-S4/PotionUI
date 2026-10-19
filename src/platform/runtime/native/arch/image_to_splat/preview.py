from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from PIL import Image

from src.platform.filesystem.splat_formats import SplatView

from .config import PREVIEW_BACKGROUND, PREVIEW_FILL_PASSES, PREVIEW_MAX_SIZE, PREVIEW_MIN_OPACITY, PREVIEW_NEAR
from .ply_export import SplatArrays, logit_to_opacity, sh0_to_rgb

__all__ = ["preview_size", "render_splat_preview"]


def preview_size(view: SplatView, max_size: int = PREVIEW_MAX_SIZE) -> Tuple[int, int]:
    longest = max(1, min(int(max_size), max(int(view.width), int(view.height))))
    if view.aspect >= 1.0:
        return longest, max(1, round(longest / view.aspect))
    return max(1, round(longest * view.aspect)), longest


def render_splat_preview(
    splats: SplatArrays,
    view: SplatView,
    max_size: int = PREVIEW_MAX_SIZE,
    background: Tuple[int, int, int] = PREVIEW_BACKGROUND,
) -> Image.Image:
    width, height = preview_size(view, max_size)
    focal = (height / 2.0) / math.tan(math.radians(view.fov_y) / 2.0)
    backdrop = np.asarray(background, dtype=np.float32) / 255.0
    canvas = np.empty((height * width, 3), dtype=np.float32)
    canvas[:] = backdrop
    filled = np.zeros(height * width, dtype=bool)

    x, y, z = splats.means[:, 0], splats.means[:, 1], splats.means[:, 2]
    alpha = logit_to_opacity(splats.opacity)
    visible = np.isfinite(splats.means).all(axis=1) & (z > PREVIEW_NEAR) & (alpha >= PREVIEW_MIN_OPACITY)
    safe_z = np.where(visible, z, 1.0)
    columns = np.floor(focal * x / safe_z + width / 2.0)
    rows = np.floor(focal * y / safe_z + height / 2.0)
    visible &= (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)

    index = np.flatnonzero(visible)
    if index.size:
        pixels = rows[index].astype(np.int64) * width + columns[index].astype(np.int64)
        order = np.lexsort((z[index], pixels))
        ordered = pixels[order]
        nearest = np.ones(ordered.shape[0], dtype=bool)
        nearest[1:] = ordered[1:] != ordered[:-1]
        chosen = index[order[nearest]]
        weight = alpha[chosen][:, None]
        canvas[ordered[nearest]] = sh0_to_rgb(splats.f_dc[chosen]) * weight + backdrop * (1.0 - weight)
        filled[ordered[nearest]] = True

    image = canvas.reshape(height, width, 3)
    _fill_holes(image, filled.reshape(height, width), PREVIEW_FILL_PASSES)
    return Image.fromarray(np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8), "RGB")


def _fill_holes(image: np.ndarray, filled: np.ndarray, passes: int) -> None:
    for _ in range(passes):
        holes = ~filled
        if not holes.any():
            return
        total = np.zeros_like(image)
        count = np.zeros(filled.shape, dtype=np.float32)
        for axis, step in ((0, 1), (0, -1), (1, 1), (1, -1)):
            shifted_filled = _shift(filled, axis, step)
            total += _shift(image, axis, step) * shifted_filled[..., None]
            count += shifted_filled
        grow = holes & (count > 0)
        image[grow] = total[grow] / count[grow][:, None]
        filled |= grow


def _shift(array: np.ndarray, axis: int, step: int) -> np.ndarray:
    shifted = np.zeros_like(array)
    if axis == 0:
        if step > 0:
            shifted[step:] = array[:-step]
        else:
            shifted[:step] = array[-step:]
    elif step > 0:
        shifted[:, step:] = array[:, :-step]
    else:
        shifted[:, :step] = array[:, -step:]
    return shifted

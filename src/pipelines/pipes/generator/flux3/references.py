from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

GRID = 16
MATCH_MAX_PIXELS = 4 * 1024 * 1024
MULTI_MAX_PIXELS = 1024 * 1024
MIN_SIDE = 64
MAX_SIDE = 4096

CropBox = Tuple[int, int, int, int]


def _ceil_to_grid(value: int) -> int:
    return -(-value // GRID) * GRID


def reference_canvas(width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = (
            min(MAX_SIDE, max(MIN_SIDE, _ceil_to_grid(fit[0]))),
            min(MAX_SIDE, max(MIN_SIDE, _ceil_to_grid(fit[1]))),
        )
        if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image: Image.Image) -> Tuple[Image.Image, Optional[CropBox]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (width, height) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (width, height) == image.size:
        return image, None
    left = (width - image.width) // 2
    top = (height - image.height) // 2
    padding = ((top, height - image.height - top), (left, width - image.width - left), (0, 0))
    mode = "reflect" if min(image.size) > GRID else "symmetric"
    padded = Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))
    return padded, (left, top, left + image.width, top + image.height)


def prepare_multi_reference(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > MULTI_MAX_PIXELS:
        scale = math.sqrt(MULTI_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    grid_w, grid_h = width // GRID * GRID, height // GRID * GRID
    left, top = (width - grid_w) // 2, (height - grid_h) // 2
    return image.crop((left, top, left + grid_w, top + grid_h))

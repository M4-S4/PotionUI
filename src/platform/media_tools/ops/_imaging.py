from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageColor, ImageFilter

from src.platform.media_tools.probe import has_alpha_mode

MAX_IMAGE_DIMENSION = 8192
MAX_VIDEO_DIMENSION = 4096
BORDER_RING_PX = 6
MAX_CHROMA_DISTANCE = (2 * 255.0 ** 2) ** 0.5


def even(value: int) -> int:
    return value - (value % 2)


def resample_ready(image: Image.Image) -> Image.Image:
    if image.mode in ("P", "1"):
        return image.convert("RGBA" if has_alpha_mode(image) else "RGB")
    return image


def parse_hex_color(color: str) -> tuple[int, int, int]:
    try:
        red, green, blue = ImageColor.getcolor(color, "RGB")
    except ValueError as exc:
        raise ValueError(f"Invalid colour: {color!r}") from exc
    return (red, green, blue)


def alpha_bbox(image: Image.Image, threshold: int = 16) -> Optional[tuple[int, int, int, int]]:
    alpha = np.array(image.convert("RGBA"))[..., 3]
    mask = alpha > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    return (int(left), int(top), int(right - left + 1), int(bottom - top + 1))


def feather_alpha(alpha: np.ndarray, feather_px: float) -> np.ndarray:
    if feather_px <= 0:
        return alpha
    blurred = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=feather_px / 2.0))
    return np.array(blurred)


def border_ring_color(rgb: np.ndarray) -> tuple[int, int, int]:
    height, width = rgb.shape[0], rgb.shape[1]
    ring = max(min(BORDER_RING_PX, height // 2, width // 2), 1)
    pixels = np.concatenate([
        rgb[:ring, :, :].reshape(-1, 3),
        rgb[-ring:, :, :].reshape(-1, 3),
        rgb[:, :ring, :].reshape(-1, 3),
        rgb[:, -ring:, :].reshape(-1, 3),
    ])
    values, counts = np.unique(pixels, axis=0, return_counts=True)
    mode = values[np.argmax(counts)]
    return (int(mode[0]), int(mode[1]), int(mode[2]))


def rgb_to_cbcr(rgb: np.ndarray) -> np.ndarray:
    red = rgb[..., 0].astype(np.float64)
    green = rgb[..., 1].astype(np.float64)
    blue = rgb[..., 2].astype(np.float64)
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0
    return np.stack([cb, cr], axis=-1)


def despill(rgb: np.ndarray, key_rgb: tuple[int, int, int]) -> np.ndarray:
    dominant = int(np.argmax(key_rgb))
    others = [channel for channel in (0, 1, 2) if channel != dominant]
    out = rgb.astype(np.float64).copy()
    average_others = (out[..., others[0]] + out[..., others[1]]) / 2.0
    out[..., dominant] = np.minimum(out[..., dominant], average_others)
    return np.clip(out, 0, 255).astype(np.uint8)


def soft_alpha(distance: np.ndarray, threshold: float, ramp_half: float) -> np.ndarray:
    low = max(0.0, threshold - ramp_half)
    high = min(MAX_CHROMA_DISTANCE, threshold + ramp_half)
    if high <= low:
        return np.where(distance <= threshold, 0, 255).astype(np.uint8)
    t = np.clip((distance - low) / (high - low), 0.0, 1.0)
    smoothed = t * t * (3.0 - 2.0 * t)
    return np.round(smoothed * 255.0).astype(np.uint8)

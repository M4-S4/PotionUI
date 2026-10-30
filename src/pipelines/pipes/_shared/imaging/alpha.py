"""Alpha-channel helpers for the generator and matting pipes: matte-strength
smoothstep tightening, opaque-alpha dropping and flattening onto a background.
Pure numpy/PIL - no torch, no GPU, deterministic. Bounding boxes, feathering and
colour keying live in `src.platform.imaging.pixels`.
"""

from typing import Tuple

import numpy as np
from PIL import Image


def apply_matte_strength(alpha: np.ndarray, strength: int) -> np.ndarray:
    """Levels/smoothstep tightening of a matting model's raw soft alpha.

    A matting model's raw output (e.g. BiRefNet's sigmoid probability, scaled
    to 0-255) carries no threshold of its own: a pixel the model is unsure
    about lands at a "wishy-washy" mid-gray value instead of committing to
    fully transparent or fully opaque. Left alone, that reads as flicker
    across a sequence where most frames matte cleanly but a few leave
    semi-transparent background residue.

    ``strength`` 0-100: 0 is the raw alpha unchanged (identity - every value
    maps to itself); 100 is a hard step at the midpoint (128): >=128 -> 255,
    <128 -> 0. Between the two, a smoothstep's transition band shrinks
    linearly from a half-width of 128 (no cut, strength 0) to 0 (hard
    threshold, strength 100) as strength rises.
    """
    strength = max(0, min(100, int(strength)))
    if strength <= 0:
        return alpha

    half_width = 128.0 * (1.0 - strength / 100.0)
    low = 128.0 - half_width
    high = 128.0 + half_width

    if high <= low:  # strength == 100: the smoothstep band has zero width.
        return np.where(alpha >= 128, 255, 0).astype(np.uint8)

    t = np.clip((alpha.astype(np.float64) - low) / (high - low), 0.0, 1.0)
    smoothed = t * t * (3.0 - 2.0 * t)
    return np.round(smoothed * 255.0).astype(np.uint8)


def drop_opaque_alpha(image: Image.Image) -> Image.Image:
    if image.mode != "RGBA":
        return image if image.mode == "RGB" else image.convert("RGB")
    alpha = np.asarray(image)[..., 3]
    if bool((alpha == 255).all()):
        return image.convert("RGB")
    return image


def flatten_onto(image: Image.Image, background: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if image.mode not in ("RGBA", "LA", "PA") and not (image.mode == "P" and "transparency" in image.info):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (*background, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")

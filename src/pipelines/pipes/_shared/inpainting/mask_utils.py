"""
Mask processing utilities for inpainting pipes.

Provides functions for mask-based color correction and blending
used across multiple inpaint pipes.
"""

import numpy as np
from PIL import Image, ImageFilter


def color_correction(inpainted_image, original_image, mask):
    """Apply color correction to blend inpainted region with original image.

    Uses weighted blending based on mask intensity to create smooth transitions
    between inpainted and original regions.

    Args:
        inpainted_image: Numpy array (H, W, C) of the inpainted result.
        original_image: Numpy array (H, W, C) of the original image.
        mask: Numpy array (H, W) with values 0-255 indicating blend weight.

    Returns:
        Color-corrected numpy array (H, W, C) with values clipped to 0-255.
    """
    fg = inpainted_image.astype(np.float32)
    bg = original_image.astype(np.float32)

    # Ensure mask has correct shape
    if mask.ndim == 2:
        mask = mask[:, :, None]

    w = mask.astype(np.float32) / 255.0
    y = fg * w + bg * (1 - w)

    return y.clip(0, 255).astype(np.uint8)


def dilate_box(region, radius):
    out = np.asarray(region, dtype=bool)
    radius = int(radius)
    if radius <= 0:
        return out.copy()
    for axis in (0, 1):
        moved = np.moveaxis(out, axis, 0)
        padded = np.pad(moved.astype(np.int32), [(radius, radius)] + [(0, 0)] * (moved.ndim - 1))
        total = np.concatenate(
            [np.zeros((1,) + padded.shape[1:], dtype=np.int32), np.cumsum(padded, axis=0, dtype=np.int32)]
        )
        width = 2 * radius + 1
        out = np.moveaxis((total[width:] - total[:-width]) > 0, 0, axis)
    return out


def feather_alpha(region, feather_px):
    region = np.asarray(region, dtype=bool)
    feather_px = int(feather_px)
    if feather_px <= 0:
        return region.astype(np.float32)
    core = dilate_box(region, feather_px // 2)
    soft = Image.fromarray(core.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(radius=feather_px / 6.0))
    alpha = np.asarray(soft, dtype=np.float32) / 255.0
    alpha[region] = 1.0
    alpha[~dilate_box(region, feather_px)] = 0.0
    return alpha


def match_ring_statistics(content, original, region, inner, outer):
    ring = dilate_box(region, outer) & ~dilate_box(region, inner)
    if int(ring.sum()) < 64:
        return content
    src = content[ring]
    ref = original[ring]
    mean_src = src.mean(axis=0)
    mean_ref = ref.mean(axis=0)
    gain = np.clip(ref.std(axis=0) / np.maximum(src.std(axis=0), 1e-3), 0.8, 1.25)
    return (content - mean_src) * gain + mean_ref


def composite_region(original, content, mask, feather, color_match):
    region = np.asarray(mask) >= 128
    if not region.any():
        return original.copy()
    ys, xs = np.nonzero(region)
    size = max(int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1)
    feather_px = max(2, int(round(float(feather) * size)))
    alpha = feather_alpha(region, feather_px)
    base = original.astype(np.float32)
    fresh = content.astype(np.float32)
    if color_match:
        fresh = match_ring_statistics(fresh, base, region, feather_px, feather_px + max(8, 2 * feather_px))
    blended = base + (fresh - base) * alpha[:, :, None]
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)

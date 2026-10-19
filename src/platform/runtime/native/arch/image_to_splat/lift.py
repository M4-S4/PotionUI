from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.platform.filesystem.splat_formats import SplatView

from .config import (
    BACKGROUND_DEPTH_MARGIN,
    BACKGROUND_OPACITY,
    DEPTH_EDGE_LOG_RATIO,
    DEPTH_EDGE_WINDOW_PX,
    EDGE_OPACITY,
    FOREGROUND_VOTE_RATIO,
    INPAINT_DILATE_PX,
    INPAINT_RADIUS_PX,
    INPAINT_SCALE,
    KEEP_SKY,
    MEDIAN_DEPTH,
    MIN_DEPTH,
    OPACITY,
    RADIAL_FLATTEN,
    SCALE_PER_PIXEL,
    SKY_BEYOND_FARTHEST,
    SKY_DEPTH_MIN,
    SKY_FARTHEST_PERCENTILE,
    SKY_OPACITY,
)
from .ply_export import SplatArrays, opacity_to_logit, rgb_to_sh0

__all__ = [
    "DepthLayers",
    "LiftedSplat",
    "build_layers",
    "depth_edges",
    "foreground_mask",
    "inpaint_background",
    "lift_to_splat",
    "normalize_depth",
    "pixel_gaussians",
    "ray_quaternions",
    "sky_depth",
]


@dataclass(frozen=True)
class DepthLayers:
    depth: np.ndarray
    present: np.ndarray
    sky: np.ndarray
    edges: np.ndarray
    foreground: np.ndarray
    background_rgb: Optional[np.ndarray]
    background_depth: Optional[np.ndarray]


@dataclass(frozen=True)
class LiftedSplat:
    splats: SplatArrays
    view: SplatView
    front_count: int
    back_count: int


def normalize_depth(depth: np.ndarray, valid: np.ndarray, target: float = MEDIAN_DEPTH) -> Tuple[np.ndarray, np.ndarray, float]:
    depth = np.asarray(depth, dtype=np.float64)
    usable = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > MIN_DEPTH)
    if not usable.any():
        raise ValueError("no pixel of the image has a usable depth")
    scale = target / float(np.median(depth[usable]))
    return np.where(usable, depth * scale, 0.0).astype(np.float32), usable, scale


def sky_depth(depth: np.ndarray, usable: np.ndarray) -> float:
    farthest = float(np.percentile(depth[usable], SKY_FARTHEST_PERCENTILE))
    return max(SKY_DEPTH_MIN, SKY_BEYOND_FARTHEST * farthest)


def depth_edges(
    depth: np.ndarray,
    present: np.ndarray,
    log_ratio: float = DEPTH_EDGE_LOG_RATIO,
    window: int = DEPTH_EDGE_WINDOW_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    log_depth = np.log(np.where(present, np.maximum(depth, MIN_DEPTH), 1.0)).astype(np.float32)
    kernel = np.ones((window, window), np.uint8)
    high = cv2.dilate(np.where(present, log_depth, -np.inf).astype(np.float32), kernel)
    low = cv2.erode(np.where(present, log_depth, np.inf).astype(np.float32), kernel)
    edges = present & ((high - low) > log_ratio)
    near = edges & ((log_depth - low) < (high - log_depth))
    return edges, near


def foreground_mask(edges: np.ndarray, near: np.ndarray, present: np.ndarray, sky: np.ndarray) -> np.ndarray:
    import cv2

    interior = present & ~edges
    count, labels = cv2.connectedComponents(interior.astype(np.uint8), connectivity=4)
    far = edges & ~near
    near_votes = np.zeros(count, dtype=np.float64)
    far_votes = np.zeros(count, dtype=np.float64)
    pairs = (
        (labels[:-1], near[1:], far[1:]), (labels[1:], near[:-1], far[:-1]),
        (labels[:, :-1], near[:, 1:], far[:, 1:]), (labels[:, 1:], near[:, :-1], far[:, :-1]),
    )
    for component, near_side, far_side in pairs:
        flat = component.ravel()
        near_votes += np.bincount(flat, weights=near_side.ravel(), minlength=count)
        far_votes += np.bincount(flat, weights=far_side.ravel(), minlength=count)
    total = near_votes + far_votes
    ratio = np.divide(near_votes, total, out=np.zeros_like(total), where=total > 0)
    chosen = ratio >= FOREGROUND_VOTE_RATIO
    chosen[0] = False
    return (chosen[labels] | near) & ~sky


def inpaint_background(
    rgb: np.ndarray, depth: np.ndarray, foreground: np.ndarray, present: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    height, width = depth.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INPAINT_DILATE_PX + 1, 2 * INPAINT_DILATE_PX + 1))
    region = (cv2.dilate(foreground.astype(np.uint8), kernel) > 0) | ~present
    small = (max(1, round(width * INPAINT_SCALE)), max(1, round(height * INPAINT_SCALE)))
    mask_small = (cv2.resize(region.astype(np.uint8) * 255, small, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    log_depth = np.log(np.maximum(depth, MIN_DEPTH)).astype(np.float32)
    radius = max(1, round(INPAINT_RADIUS_PX * INPAINT_SCALE))
    rgb_fill = cv2.inpaint(
        cv2.resize(np.ascontiguousarray(rgb), small, interpolation=cv2.INTER_AREA), mask_small, radius, cv2.INPAINT_TELEA
    )
    depth_fill = _inpaint_telea_float(cv2.resize(log_depth, small, interpolation=cv2.INTER_AREA), mask_small, radius)
    background_rgb = cv2.resize(rgb_fill, (width, height), interpolation=cv2.INTER_LINEAR)
    background_depth = np.exp(cv2.resize(depth_fill, (width, height), interpolation=cv2.INTER_LINEAR))
    background_depth = np.maximum(background_depth, depth * (1.0 + BACKGROUND_DEPTH_MARGIN))
    return background_rgb.astype(np.uint8), background_depth.astype(np.float32)


def _inpaint_telea_float(values: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    import cv2

    known = values[mask == 0]
    if known.size == 0:
        return values
    low, high = float(known.min()), float(known.max())
    span = max(high - low, 1e-6)
    # OpenCV's Telea returns noise for float32 input, so it runs on a 16-bit quantisation of the range.
    quantised = np.round((np.clip(values, low, high) - low) / span * 65535.0).astype(np.uint16)
    filled = cv2.inpaint(quantised, mask, radius, cv2.INPAINT_TELEA)
    return (filled.astype(np.float32) / 65535.0 * span + low).astype(np.float32)


def ray_quaternions(rays: np.ndarray) -> np.ndarray:
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    quaternions = np.stack([1.0 + rays[:, 2], -rays[:, 1], rays[:, 0], np.zeros(rays.shape[0])], axis=-1)
    return (quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)).astype(np.float32)


def pixel_gaussians(
    rgb: np.ndarray, depth: np.ndarray, select: np.ndarray, alpha: np.ndarray, fx: float, fy: float
) -> SplatArrays:
    height, width = depth.shape
    rows, cols = np.nonzero(select)
    z = depth[rows, cols].astype(np.float64)
    means = np.stack([
        (cols + 0.5 - width / 2.0) / fx * z,
        (rows + 0.5 - height / 2.0) / fy * z,
        z,
    ], axis=-1)
    log_sigma = np.log(np.maximum(z / (0.5 * (fx + fy)) * SCALE_PER_PIXEL, MIN_DEPTH * 1e-3))
    log_scales = np.stack([log_sigma, log_sigma, log_sigma + math.log(RADIAL_FLATTEN)], axis=-1)
    colours = rgb[rows, cols].astype(np.float32) / 255.0
    return SplatArrays(
        means=means,
        f_dc=rgb_to_sh0(colours),
        opacity=opacity_to_logit(alpha[rows, cols]),
        log_scales=log_scales,
        rotations=ray_quaternions(means) if means.shape[0] else np.zeros((0, 4)),
    )


def build_layers(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray) -> DepthLayers:
    depth, usable, _scale = normalize_depth(depth, valid)
    sky = ~usable if KEEP_SKY else np.zeros_like(usable)
    if sky.any():
        depth = np.where(sky, sky_depth(depth, usable), depth).astype(np.float32)
    present = usable | sky
    edges, near = depth_edges(depth, present)
    foreground = foreground_mask(edges, near, present, sky)
    background_rgb = background_depth = None
    if foreground.any():
        background_rgb, background_depth = inpaint_background(rgb, depth, foreground, present)
    return DepthLayers(
        depth=depth,
        present=present,
        sky=sky,
        edges=edges,
        foreground=foreground,
        background_rgb=background_rgb,
        background_depth=background_depth,
    )


def lift_to_splat(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, fx: float, fy: float) -> LiftedSplat:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != np.shape(depth):
        raise ValueError(f"image {rgb.shape} and depth {np.shape(depth)} must share one height and width")
    layers = build_layers(rgb, depth, valid)
    height, width = layers.depth.shape

    alpha = np.where(layers.sky, SKY_OPACITY, OPACITY).astype(np.float32)
    alpha[layers.edges] = EDGE_OPACITY
    front = pixel_gaussians(rgb, layers.depth, layers.present, alpha, fx, fy)
    parts = [front]
    back_count = 0
    if layers.background_depth is not None:
        back_alpha = np.full(layers.depth.shape, BACKGROUND_OPACITY, dtype=np.float32)
        back = pixel_gaussians(layers.background_rgb, layers.background_depth, layers.foreground, back_alpha, fx, fy)
        parts.append(back)
        back_count = back.count

    view = SplatView(
        fov_y=math.degrees(2.0 * math.atan(height / 2.0 / fy)),
        aspect=width / height,
        depth=MEDIAN_DEPTH,
        width=width,
        height=height,
    )
    return LiftedSplat(SplatArrays.concat(parts), view, front.count, back_count)

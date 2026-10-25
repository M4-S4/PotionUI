from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.platform.filesystem.splat_formats import SplatView

from .config import (
    BACKGROUND_DEPTH_MARGIN,
    BACKGROUND_OPACITY,
    DISOCCLUSION_BASELINE,
    EDGE_DROP,
    EDGE_JUMP_LOG_RATIO,
    EDGE_OPACITY,
    EDGE_PIXELS,
    EDGE_SNAP_WINDOW_PX,
    FOREGROUND_VOTE_RATIO,
    HIDDEN_MAX_FRACTION,
    HIDDEN_MIN_PX,
    INPAINT_DILATE_PX,
    INPAINT_RADIUS_PX,
    INPAINT_SCALE,
    KEEP_SKY,
    MEDIAN_DEPTH,
    MIN_DEPTH,
    NORMAL_THICKNESS,
    OPACITY,
    SKY_BEYOND_FARTHEST,
    SKY_DEPTH_MIN,
    SKY_FARTHEST_PERCENTILE,
    SKY_OPACITY,
    TANGENT_MAX_FOOTPRINTS,
    TANGENT_SCALE,
)
from .ply_export import SplatArrays, opacity_to_logit, rgb_to_sh0

__all__ = [
    "DepthCuts",
    "DepthLayers",
    "LiftedSplat",
    "build_layers",
    "depth_jumps",
    "foreground_mask",
    "hidden_bands",
    "inpaint_background",
    "lift_to_splat",
    "matrix_quaternions",
    "normalize_depth",
    "pixel_gaussians",
    "sky_depth",
    "surface_tangents",
    "tangent_frames",
    "unproject",
]


@dataclass(frozen=True)
class DepthLayers:
    depth: np.ndarray
    present: np.ndarray
    sky: np.ndarray
    cuts: np.ndarray
    hidden: np.ndarray
    background_rgb: Optional[np.ndarray]
    background_depth: Optional[np.ndarray]


@dataclass(frozen=True)
class DepthCuts:
    cuts: np.ndarray
    near_side: np.ndarray
    snapped: np.ndarray
    near_log: np.ndarray
    far_log: np.ndarray


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


def foreground_mask(cuts: np.ndarray, near: np.ndarray, present: np.ndarray, sky: np.ndarray) -> np.ndarray:
    import cv2

    interior = present & ~cuts
    count, labels = cv2.connectedComponents(interior.astype(np.uint8), connectivity=4)
    far = cuts & ~near
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
    rgb: np.ndarray, depth: np.ndarray, fill: np.ndarray, present: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    height, width = depth.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INPAINT_DILATE_PX + 1, 2 * INPAINT_DILATE_PX + 1))
    region = (cv2.dilate(fill.astype(np.uint8), kernel) > 0) | ~present
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


def depth_jumps(
    depth: np.ndarray,
    present: np.ndarray,
    log_ratio: float = EDGE_JUMP_LOG_RATIO,
    window: int = EDGE_SNAP_WINDOW_PX,
) -> DepthCuts:
    import cv2

    log_depth = np.log(np.maximum(depth, MIN_DEPTH)).astype(np.float32)
    kernel = np.ones((window, window), np.uint8)
    far = cv2.dilate(np.where(present, log_depth, -np.inf).astype(np.float32), kernel)
    near = cv2.erode(np.where(present, log_depth, np.inf).astype(np.float32), kernel)
    cuts = present & ((far - near) > log_ratio)
    near_side = cuts & ((log_depth - near) <= (far - log_depth))
    snapped = np.where(cuts, np.exp(np.where(near_side, near, far)), depth).astype(np.float32)
    return DepthCuts(cuts=cuts, near_side=near_side, snapped=snapped, near_log=near, far_log=far)


def hidden_bands(
    cuts: DepthCuts, depth: np.ndarray, present: np.ndarray, sky: np.ndarray, focal: float
) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    seeds = cuts.near_side & ~sky
    if not seeds.any():
        empty = np.zeros(depth.shape, dtype=bool)
        return empty, empty
    baseline = DISOCCLUSION_BASELINE * MEDIAN_DEPTH
    rows, cols = np.nonzero(seeds)
    near, far = cuts.near_log[rows, cols], cuts.far_log[rows, cols]
    width = focal * baseline * (np.exp(-near) - np.exp(-far))
    width = np.clip(width, HIDDEN_MIN_PX, HIDDEN_MAX_FRACTION * max(depth.shape))
    distance, labels = cv2.distanceTransformWithLabels(
        (~seeds).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    by_label = np.zeros(labels.max() + 1)
    middle = np.zeros(labels.max() + 1)
    by_label[labels[rows, cols]] = width
    middle[labels[rows, cols]] = 0.5 * (near + far)
    occluder_side = present & ~sky & (np.log(np.maximum(depth, MIN_DEPTH)) <= middle[labels])
    reach = by_label[labels]
    return occluder_side & (distance <= reach), occluder_side & (distance <= 2.0 * reach)


def unproject(depth: np.ndarray, fx: float, fy: float) -> np.ndarray:
    height, width = depth.shape
    cols = np.arange(width, dtype=np.float64) + 0.5 - width / 2.0
    rows = np.arange(height, dtype=np.float64) + 0.5 - height / 2.0
    z = depth.astype(np.float64)
    return np.stack([cols[None, :] / fx * z, np.broadcast_to(rows[:, None] / fy, z.shape) * z, z], axis=-1)


def _axis_tangent(
    points: np.ndarray, present: np.ndarray, limit: np.ndarray, jump: float
) -> Tuple[np.ndarray, np.ndarray]:
    step = points[:, 1:] - points[:, :-1]
    length = np.linalg.norm(step, axis=-1)
    log_z = np.log(np.maximum(points[..., 2], MIN_DEPTH))
    continuous = present[:, 1:] & present[:, :-1] & (np.abs(log_z[:, 1:] - log_z[:, :-1]) <= jump)
    step = step * np.minimum(1.0, np.minimum(limit[:, 1:], limit[:, :-1]) / np.maximum(length, 1e-12))[..., None]
    forward = np.zeros_like(points)
    backward = np.zeros_like(points)
    forward_ok = np.zeros(present.shape, dtype=bool)
    backward_ok = np.zeros(present.shape, dtype=bool)
    forward[:, :-1], forward_ok[:, :-1] = step, continuous
    backward[:, 1:], backward_ok[:, 1:] = step, continuous
    both = (forward_ok & backward_ok)[..., None]
    tangent = np.where(both, 0.5 * (forward + backward), np.where(forward_ok[..., None], forward, backward))
    return tangent, forward_ok | backward_ok


def surface_tangents(
    points: np.ndarray,
    present: np.ndarray,
    fx: float,
    fy: float,
    max_footprints: float = TANGENT_MAX_FOOTPRINTS,
    jump: float = EDGE_JUMP_LOG_RATIO,
) -> Tuple[np.ndarray, np.ndarray]:
    z = points[..., 2]
    limit = max_footprints * z / (0.5 * (fx + fy))
    tangent_u, found_u = _axis_tangent(points, present, limit, jump)
    tangent_v, found_v = _axis_tangent(points.swapaxes(0, 1), present.T, limit.T, jump)
    tangent_v, found_v = tangent_v.swapaxes(0, 1), found_v.T
    fronto_u = np.zeros_like(points)
    fronto_v = np.zeros_like(points)
    fronto_u[..., 0] = z / fx
    fronto_v[..., 1] = z / fy
    tangent_u = np.where(found_u[..., None], tangent_u, fronto_u)
    tangent_v = np.where(found_v[..., None], tangent_v, fronto_v)
    normal_length = np.linalg.norm(np.cross(tangent_u, tangent_v), axis=-1)
    flat = normal_length <= 1e-6 * np.linalg.norm(tangent_u, axis=-1) * np.linalg.norm(tangent_v, axis=-1)
    return np.where(flat[..., None], fronto_u, tangent_u), np.where(flat[..., None], fronto_v, tangent_v)


def matrix_quaternions(matrices: np.ndarray) -> np.ndarray:
    m = matrices
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    candidates = np.stack([
        np.stack([1 + trace, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], axis=-1),
        np.stack([m[:, 2, 1] - m[:, 1, 2], 1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], m[:, 0, 1] + m[:, 1, 0],
                  m[:, 0, 2] + m[:, 2, 0]], axis=-1),
        np.stack([m[:, 0, 2] - m[:, 2, 0], m[:, 0, 1] + m[:, 1, 0], 1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                  m[:, 1, 2] + m[:, 2, 1]], axis=-1),
        np.stack([m[:, 1, 0] - m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1],
                  1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], axis=-1),
    ], axis=1)
    # Shepperd: start from the largest of 4w², 4x², 4y², 4z² so a turn near 180 degrees keeps the sign of its axis.
    choice = np.argmax(np.stack([candidates[:, k, k] for k in range(4)], axis=-1), axis=-1)
    quaternions = candidates[np.arange(m.shape[0]), choice]
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    quaternions *= np.where(quaternions[:, :1] < 0, -1.0, 1.0)
    return quaternions.astype(np.float32)


def tangent_frames(
    tangent_u: np.ndarray, tangent_v: np.ndarray, thickness: np.ndarray, scale: float = TANGENT_SCALE
) -> Tuple[np.ndarray, np.ndarray]:
    normal = np.cross(tangent_u, tangent_v)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    length_u = np.linalg.norm(tangent_u, axis=-1)
    first = tangent_u / length_u[:, None]
    second = np.cross(normal, first)
    along = np.einsum("ij,ij->i", tangent_v, first)
    across = np.einsum("ij,ij->i", tangent_v, second)
    p = scale * scale * (length_u * length_u + along * along)
    q = scale * scale * along * across
    r = scale * scale * across * across
    angle = 0.5 * np.arctan2(2.0 * q, p - r)
    half = 0.5 * (p + r)
    spread = np.sqrt((0.5 * (p - r)) ** 2 + q * q)
    major = np.sqrt(half + spread)
    minor = np.sqrt(np.maximum(half - spread, 1e-12 * half))
    cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
    basis = np.stack([cos * first + sin * second, cos * second - sin * first, normal], axis=-1)
    log_scales = np.log(np.stack([major, minor, np.maximum(thickness, 1e-12)], axis=-1))
    return matrix_quaternions(basis), log_scales.astype(np.float32)


def pixel_gaussians(
    rgb: np.ndarray,
    depth: np.ndarray,
    select: np.ndarray,
    alpha: np.ndarray,
    fx: float,
    fy: float,
    present: Optional[np.ndarray] = None,
) -> SplatArrays:
    present = select if present is None else present
    points = unproject(depth, fx, fy)
    rows, cols = np.nonzero(select)
    if rows.size == 0:
        empty = np.zeros((0, 3))
        return SplatArrays(empty, empty, np.zeros(0), empty, np.zeros((0, 4)))
    tangent_u, tangent_v = surface_tangents(points, present, fx, fy, TANGENT_MAX_FOOTPRINTS, EDGE_JUMP_LOG_RATIO)
    z = points[rows, cols, 2]
    rotations, log_scales = tangent_frames(
        tangent_u[rows, cols], tangent_v[rows, cols], NORMAL_THICKNESS * z / (0.5 * (fx + fy)), TANGENT_SCALE
    )
    colours = rgb[rows, cols].astype(np.float32) / 255.0
    return SplatArrays(
        means=points[rows, cols],
        f_dc=rgb_to_sh0(colours),
        opacity=opacity_to_logit(alpha[rows, cols]),
        log_scales=log_scales,
        rotations=rotations,
    )


def build_layers(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, fx: float, fy: float) -> DepthLayers:
    depth, usable, _scale = normalize_depth(depth, valid)
    sky = ~usable if KEEP_SKY else np.zeros_like(usable)
    if sky.any():
        depth = np.where(sky, sky_depth(depth, usable), depth).astype(np.float32)
    present = usable | sky
    cuts = depth_jumps(depth, present, EDGE_JUMP_LOG_RATIO, EDGE_SNAP_WINDOW_PX)
    hidden, fill = hidden_bands(cuts, depth, present, sky, 0.5 * (fx + fy))
    segments = foreground_mask(cuts.cuts, cuts.near_side, present, sky)
    hidden, fill = hidden | segments, fill | segments
    if EDGE_PIXELS == EDGE_DROP:
        hidden, fill = hidden | cuts.cuts, fill | cuts.cuts
        visible = present & ~cuts.cuts
    else:
        depth = cuts.snapped
        visible = present
    background_rgb = background_depth = None
    if hidden.any():
        background_rgb, background_depth = inpaint_background(rgb, depth, fill, present)
    return DepthLayers(
        depth=depth,
        present=visible,
        sky=sky,
        cuts=cuts.cuts,
        hidden=hidden,
        background_rgb=background_rgb,
        background_depth=background_depth,
    )


def lift_to_splat(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, fx: float, fy: float) -> LiftedSplat:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != np.shape(depth):
        raise ValueError(f"image {rgb.shape} and depth {np.shape(depth)} must share one height and width")
    layers = build_layers(rgb, depth, valid, fx, fy)
    height, width = layers.depth.shape

    alpha = np.where(layers.sky, SKY_OPACITY, OPACITY).astype(np.float32)
    alpha[layers.cuts] = EDGE_OPACITY
    front = pixel_gaussians(rgb, layers.depth, layers.present, alpha, fx, fy)
    parts = [front]
    back_count = 0
    if layers.background_depth is not None:
        back_alpha = np.full(layers.depth.shape, BACKGROUND_OPACITY, dtype=np.float32)
        back = pixel_gaussians(
            layers.background_rgb, layers.background_depth, layers.hidden, back_alpha, fx, fy,
            present=np.ones_like(layers.hidden),
        )
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

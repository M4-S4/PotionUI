
from __future__ import annotations

from dataclasses import dataclass

from src.platform.runtime.native.resolution import snap_resolution, snap_to_multiple
from src.platform.runtime.native.vae.ltx_latent_upsampler import (
    _rational_for_scale,
    rational_resample_out_size,
)
from src.pipelines.pipes._shared.video.ltx_constants import LTX_SPATIAL_DOWNSCALE


def temporal_upsample_out_frames(frames: int) -> int:
    """Frame count a temporal x2 upsample produces from ``frames``.

    ``LTXLatentUpsampler``'s temporal branch pixel-shuffles the frame axis by
    2 and then drops the first frame (see that module's docstring for the
    reference citations), so ``T -> 2T - 1``.

    The same expression holds for LATENT frames and for PIXEL frames, which is
    why one helper serves both: a latent count ``T`` corresponds to
    ``(T - 1) * 8 + 1`` pixel frames, and ``(2T - 1 - 1) * 8 + 1`` is exactly
    ``2 * ((T - 1) * 8 + 1) - 1``. It is also the mapping the DFR facts spec
    states for one temporal round (``N -> 2(N - 1) + 1``).
    """
    n = int(frames)
    if n < 1:
        raise ValueError(f"temporal_upsample_out_frames needs at least 1 frame, got {n}")
    return 2 * n - 1


def required_axis_divisor(scale: float) -> int:
    _, den = _rational_for_scale(scale)
    return LTX_SPATIAL_DOWNSCALE * den


@dataclass(frozen=True)
class LtxTwoStageGeometry:
    stage1_width: int
    stage1_height: int
    # Stage 2's OWN configured resolution (independently snapped -- see module
    # docstring point 2), NOT what the resampler will actually hand it.
    stage2_width: int
    stage2_height: int
    actual_width_lat: int
    actual_height_lat: int
    expected_width_lat: int
    expected_height_lat: int

    @property
    def ok(self) -> bool:
        return (
            self.actual_width_lat == self.expected_width_lat
            and self.actual_height_lat == self.expected_height_lat
        )


def compute_two_stage_geometry(width: int, height: int, scale: float) -> LtxTwoStageGeometry:
    """Predict whether ``latent_upscaler/ltx``'s real output will match
    ``generator_stage2``'s own configured resolution for a ``width x height``
    request at ``scale`` (the preset's ``upscale`` selection: 1.5 or 2.0).
    """
    stage1_w, stage1_h = snap_resolution(width, height, LTX_SPATIAL_DOWNSCALE, 1)
    stage1_w_lat, stage1_h_lat = stage1_w // LTX_SPATIAL_DOWNSCALE, stage1_h // LTX_SPATIAL_DOWNSCALE

    actual_w_lat = rational_resample_out_size(stage1_w_lat, scale)
    actual_h_lat = rational_resample_out_size(stage1_h_lat, scale)

    # Mirrors pipeline.yml's generator_stage2 `resolution:` Jinja expression
    # exactly: `(stage1 * factor) | round | int`, then re-snapped the same
    # way generator/video_ltx's build_context snaps ANY configured resolution.
    target_w_px = round(stage1_w * scale)
    target_h_px = round(stage1_h * scale)
    stage2_w, stage2_h = snap_resolution(target_w_px, target_h_px, LTX_SPATIAL_DOWNSCALE, 1)

    return LtxTwoStageGeometry(
        stage1_width=stage1_w,
        stage1_height=stage1_h,
        stage2_width=stage2_w,
        stage2_height=stage2_h,
        actual_width_lat=actual_w_lat,
        actual_height_lat=actual_h_lat,
        expected_width_lat=stage2_w // LTX_SPATIAL_DOWNSCALE,
        expected_height_lat=stage2_h // LTX_SPATIAL_DOWNSCALE,
    )


def _axis_matches(pixel: int, scale: float) -> bool:
    """Whether a single already-32-snapped axis of size ``pixel`` round-trips
    cleanly through both stage-2 geometry sources (module docstring points
    1/2) at ``scale``."""
    lat = pixel // LTX_SPATIAL_DOWNSCALE
    actual = rational_resample_out_size(lat, scale)
    expected_px = snap_to_multiple(round(pixel * scale), LTX_SPATIAL_DOWNSCALE)
    return actual == expected_px // LTX_SPATIAL_DOWNSCALE


def nearest_achievable_pixel(pixel: int, scale: float, *, max_steps: int = 8) -> int:
    """Nearest 32px-grid value to ``pixel`` whose two-stage geometry agrees
    (module docstring) -- searches outward one 32px step at a time, trying
    the SMALLER candidate first at each distance (matches
    ``snap_to_multiple``'s own round-down-on-ties direction elsewhere in this
    family). The mismatch pattern is periodic in stage-1's latent axis (every
    ``den`` steps of 32px, ``den`` from the scale's rational num/den -- at
    most 4 for the supported scales), so ``max_steps`` never needs to be
    large; it exists only as a hard stop, never expected to bind.
    """
    base = snap_to_multiple(pixel, LTX_SPATIAL_DOWNSCALE)
    if _axis_matches(base, scale):
        return base
    for step in range(1, max_steps + 1):
        lower = base - step * LTX_SPATIAL_DOWNSCALE
        if lower >= LTX_SPATIAL_DOWNSCALE and _axis_matches(lower, scale):
            return lower
        higher = base + step * LTX_SPATIAL_DOWNSCALE
        if _axis_matches(higher, scale):
            return higher
    return base  # pragma: no cover - unreachable for the supported scale map


def nearest_achievable_resolution(width: int, height: int, scale: float) -> tuple[int, int]:
    """Nearest ``width x height`` (each axis independently) whose two-stage
    geometry agrees at ``scale`` -- the suggestion surfaced by the
    preflight error message."""
    return (
        nearest_achievable_pixel(width, scale),
        nearest_achievable_pixel(height, scale),
    )

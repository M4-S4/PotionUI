from __future__ import annotations

from typing import Any, Tuple

from src.pipelines.pipes._shared.video.ltx_constants import LTX_SPATIAL_DOWNSCALE, LTX_TEMPORAL_DOWNSCALE


def probe_video_geometry(path: Any) -> Tuple[int, int, int, float]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"generator/video_ltx: cannot open input video {path!r}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if width <= 0 or height <= 0 or frames <= 0:
        raise ValueError(f"generator/video_ltx: cannot read the size or length of input video {path!r}")
    return width, height, frames, fps


def fit_to_input(
    width: int, height: int, frames: int, fps: float, *,
    scale: float, max_edge: int, max_frames: int, fallback_fps: float,
) -> Tuple[int, int, int, float]:
    target_w, target_h = width * scale, height * scale
    longest = max(target_w, target_h)
    if longest > max_edge:
        target_w, target_h = target_w * max_edge / longest, target_h * max_edge / longest
    grid = LTX_SPATIAL_DOWNSCALE * 2
    ceiling = max(grid, max_edge // grid * grid)
    out_w = min(ceiling, max(grid, int(round(target_w / grid)) * grid))
    out_h = min(ceiling, max(grid, int(round(target_h / grid)) * grid))
    usable = max(1, min(frames, max_frames))
    out_frames = (usable - 1) // LTX_TEMPORAL_DOWNSCALE * LTX_TEMPORAL_DOWNSCALE + 1
    out_fps = min(60.0, max(1.0, fps)) if fps > 0 else fallback_fps
    return out_w, out_h, out_frames, out_fps

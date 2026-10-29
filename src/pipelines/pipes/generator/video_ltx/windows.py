from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from src.pipelines.pipes._shared.video.ltx_constants import LTX_TEMPORAL_DOWNSCALE


def snap_frames_down(frames: int) -> int:
    return max(1, (int(frames) - 1) // LTX_TEMPORAL_DOWNSCALE * LTX_TEMPORAL_DOWNSCALE + 1)


def snap_frames_up(frames: int) -> int:
    return -(-(max(1, int(frames)) - 1) // LTX_TEMPORAL_DOWNSCALE) * LTX_TEMPORAL_DOWNSCALE + 1


def min_window_frames(overlap: int) -> int:
    return max(LTX_TEMPORAL_DOWNSCALE + 1, snap_frames_up(2 * max(0, int(overlap)) + 1))


def plan_windows(total: int, cap: int, overlap: int) -> Tuple[int, List[int]]:
    cap = snap_frames_down(cap)
    if total <= cap:
        return snap_frames_down(total), [0]
    overlap = max(0, min(int(overlap), cap - 1))
    count = 2
    while True:
        length = snap_frames_up(-(-(total + (count - 1) * overlap) // count))
        if length <= cap:
            break
        count += 1
    span = total - length
    return length, [round(i * span / (count - 1)) for i in range(count)]


def largest_fitting_frames(total: int, fits: Callable[[int], bool], floor: int = 1) -> Optional[int]:
    frames = snap_frames_down(total)
    floor = max(1, int(floor))
    while frames >= floor:
        if fits(frames):
            return frames
        frames -= LTX_TEMPORAL_DOWNSCALE
    return None


def crossfade_windows(chunks: Sequence[Tuple[int, np.ndarray]], total: int) -> np.ndarray:
    first = chunks[0][1]
    out = np.zeros((total, *first.shape[1:]), dtype=np.float32)
    end = 0
    for start, frames in chunks:
        length = frames.shape[0]
        overlap = max(0, end - start)
        if overlap:
            weight = (np.arange(1, overlap + 1, dtype=np.float32) / (overlap + 1)).reshape(-1, *([1] * (frames.ndim - 1)))
            out[start:end] = out[start:end] * (1.0 - weight) + frames[:overlap].astype(np.float32) * weight
        out[start + overlap:start + length] = frames[overlap:]
        end = max(end, start + length)
    return out.round().clip(0, 255).astype(first.dtype) if first.dtype == np.uint8 else out.astype(first.dtype)

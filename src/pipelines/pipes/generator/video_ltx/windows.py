from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def plan_windows(total: int, window: int, overlap: int) -> List[int]:
    if total <= window:
        return [0]
    step = max(1, window - max(0, overlap))
    starts = list(range(0, total - window, step))
    starts.append(total - window)
    return starts


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

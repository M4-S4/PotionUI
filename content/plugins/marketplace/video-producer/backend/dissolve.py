import math
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence

KEY = "dissolve"
MAX_DISSOLVE = 2
GRID = 0.04
STEP_SLACK = 1e-6


def _seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) and number > 0 else 0.0


def dissolve_of(before: Optional[Mapping[str, Any]], clip: Optional[Mapping[str, Any]]) -> float:
    if before is None or clip is None:
        return 0.0
    asked = _seconds(clip.get(KEY))
    if asked <= 0:
        return 0.0
    length = min(asked, MAX_DISSOLVE, _seconds(before.get("dur")) / 2, _seconds(clip.get("dur")) / 2)
    steps = math.floor(length / GRID + STEP_SLACK)
    return round(steps * GRID, 6) if steps >= 1 else 0.0


def dissolves(clips: Sequence[Optional[Mapping[str, Any]]]) -> List[float]:
    return [0.0, *(dissolve_of(before, clip) for before, clip in zip(clips, clips[1:]))][: len(clips)]


@dataclass(frozen=True)
class Cut:
    start: int
    end: int
    overlap: int
    head: int


def cuts(clips: Sequence[Mapping[str, Any]], frames: Callable[[Any], int]) -> List[Optional[Cut]]:
    joins = dissolves(clips)
    overlaps = [0] * len(clips)
    for index in range(1, len(clips)):
        if joins[index] > 0:
            before = clips[index - 1]
            overlaps[index] = max(frames(before["start"] + before["dur"]) - frames(clips[index]["start"]), 0)
    if not any(overlaps):
        return [None] * len(clips)
    edits = [frames(clip["start"]) + overlap // 2 for clip, overlap in zip(clips, overlaps)]
    result: List[Optional[Cut]] = []
    for index, clip in enumerate(clips):
        start = edits[index] if overlaps[index] else frames(clip["start"])
        end = edits[index + 1] if index + 1 < len(clips) and overlaps[index + 1] else frames(clip["start"] + clip["dur"])
        touched = overlaps[index] or (index + 1 < len(clips) and overlaps[index + 1])
        result.append(Cut(start, end, overlaps[index], start - frames(clip["start"])) if touched else None)
    return result

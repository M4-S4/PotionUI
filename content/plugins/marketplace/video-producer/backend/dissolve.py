import math
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence

KEY = "dissolve"
MAX_DISSOLVE = 2
STEP_SLACK = 1e-4
PLACE_SLACK = 0.5


def _seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) and number > 0 else 0.0


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0


def frames_of(seconds: Any, fps: Any) -> int:
    return math.floor(_number(seconds) * _seconds(fps) + 0.5)


def seconds_of(frames: int, fps: Any) -> float:
    rate = _seconds(fps)
    return round(frames / rate, 6) if rate and frames else 0.0


def dissolve_frames(before: Optional[Mapping[str, Any]], clip: Optional[Mapping[str, Any]], fps: Any) -> int:
    rate = _seconds(fps)
    if before is None or clip is None or not rate:
        return 0
    asked = _seconds(clip.get(KEY))
    if asked <= 0:
        return 0
    frames = min(
        math.floor(asked * rate + STEP_SLACK),
        math.floor(MAX_DISSOLVE * rate + STEP_SLACK),
        frames_of(before.get("dur"), rate) // 2,
        frames_of(clip.get("dur"), rate) // 2,
    )
    return max(frames, 0)


def dissolve_of(before: Optional[Mapping[str, Any]], clip: Optional[Mapping[str, Any]], fps: Any) -> float:
    return seconds_of(dissolve_frames(before, clip, fps), fps)


def joined_frames(before: Optional[Mapping[str, Any]], clip: Optional[Mapping[str, Any]], fps: Any) -> int:
    frames = dissolve_frames(before, clip, fps)
    if not frames:
        return 0
    overlap = (_number(before.get("start")) + _number(before.get("dur")) - _number(clip.get("start"))) * _seconds(fps)
    return frames if abs(overlap - frames) <= PLACE_SLACK else 0


def joins(clips: Sequence[Optional[Mapping[str, Any]]], fps: Any) -> List[int]:
    return [0, *(joined_frames(before, clip, fps) for before, clip in zip(clips, clips[1:]))][: len(clips)]


def dissolves(clips: Sequence[Optional[Mapping[str, Any]]], fps: Any) -> List[float]:
    return [seconds_of(frames, fps) for frames in joins(clips, fps)]


def layout(clips: Sequence[Mapping[str, Any]], fps: Any, snap: Callable[[float], float]) -> List[float]:
    starts: List[float] = []
    at, shift = 0.0, 0.0
    for index, clip in enumerate(clips):
        if index:
            shift += dissolve_of(clips[index - 1], clip, fps)
        starts.append(snap(at) if not shift else round(snap(at) - shift, 6))
        at += _number(clip.get("dur"))
    return starts


@dataclass(frozen=True)
class Cut:
    start: int
    end: int
    overlap: int
    head: int


def cuts(clips: Sequence[Mapping[str, Any]], fps: Any, frames: Callable[[Any], int]) -> List[Optional[Cut]]:
    overlaps = joins(clips, fps)
    if not any(overlaps):
        return [None] * len(clips)
    edits = [frames(clip["start"]) + overlap // 2 for clip, overlap in zip(clips, overlaps)]
    result: List[Optional[Cut]] = []
    for index, clip in enumerate(clips):
        following = index + 1 < len(clips) and overlaps[index + 1] > 0
        if not overlaps[index] and not following:
            result.append(None)
            continue
        start = edits[index] if overlaps[index] else frames(clip["start"])
        end = edits[index + 1] if following else frames(clip["start"] + clip["dur"])
        result.append(Cut(start, end, overlaps[index], start - frames(clip["start"])))
    return result

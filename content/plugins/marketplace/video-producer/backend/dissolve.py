import math
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence

KEY = "dissolve"
MAX_DISSOLVE = 2
GRID = 0.04
MIN_STEPS = 2
STEP_SLACK = 1e-6
EPSILON = 1e-6


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


def dissolve_of(before: Optional[Mapping[str, Any]], clip: Optional[Mapping[str, Any]]) -> float:
    if before is None or clip is None:
        return 0.0
    asked = _seconds(clip.get(KEY))
    if asked <= 0:
        return 0.0
    length = min(asked, MAX_DISSOLVE, _seconds(before.get("dur")) / 2, _seconds(clip.get("dur")) / 2)
    steps = math.floor(length / GRID + STEP_SLACK)
    return round(steps * GRID, 6) if steps >= MIN_STEPS else 0.0


def played(before: Optional[Mapping[str, Any]], clip: Optional[Mapping[str, Any]]) -> float:
    if not dissolve_of(before, clip):
        return 0.0
    overlap = _number(before.get("start")) + _number(before.get("dur")) - _number(clip.get("start"))
    if overlap <= GRID / 2 or overlap > min(_seconds(before.get("dur")), _seconds(clip.get("dur"))) / 2 + 1.5 * GRID:
        return 0.0
    return round(overlap, 6)


@dataclass(frozen=True)
class Place:
    start: float
    end: float
    overlap: float


def placement(clips: Sequence[Mapping[str, Any]]) -> List[Optional[Place]]:
    places: List[Optional[Place]] = [None] * len(clips)
    last: Optional[int] = None
    for index, clip in enumerate(clips):
        dur = _seconds(clip.get("dur"))
        if dur <= EPSILON:
            continue
        start, overlap = _number(clip.get("start")), 0.0
        if last is not None:
            before = places[last]
            joined = played(clips[last], clip) if last == index - 1 and before.start == _number(clips[last].get("start")) else 0.0
            if joined:
                overlap = joined
            elif start < before.end - EPSILON:
                if start > before.start + before.overlap + EPSILON:
                    places[last] = Place(before.start, start, before.overlap)
                else:
                    start = before.end
        places[index] = Place(start, start + dur, overlap)
        last = index
    return places


def moved(clip: Mapping[str, Any], place: Optional[Place]) -> bool:
    return place is not None and (place.start != _number(clip.get("start")) or place.end != place.start + _seconds(clip.get("dur")))


def joins(clips: Sequence[Mapping[str, Any]]) -> List[float]:
    return [place.overlap if place is not None else 0.0 for place in placement(clips)]


def dissolving(clips: Sequence[Mapping[str, Any]]) -> bool:
    return any(joins(clips))


def shown(clips: Sequence[Mapping[str, Any]]) -> List[Place]:
    places = placement(clips)
    if not any(place is not None and place.overlap for place in places):
        return [Place(_number(clip.get("start")), _number(clip.get("start")) + _number(clip.get("dur")), 0.0) for clip in clips]
    result: List[Place] = []
    for index, place in enumerate(places):
        if place is None:
            start = _number(clips[index].get("start"))
            result.append(Place(start, start, 0.0))
            continue
        tail = places[index + 1].overlap / 2 if index + 1 < len(places) and places[index + 1] is not None else 0.0
        start = place.start + place.overlap / 2
        result.append(Place(start, max(place.end - tail, start), place.overlap))
    return result


def layout(clips: Sequence[Mapping[str, Any]], snap: Callable[[float], float]) -> List[float]:
    starts: List[float] = []
    at, shift = 0.0, 0.0
    for index, clip in enumerate(clips):
        if index:
            shift += dissolve_of(clips[index - 1], clip)
        starts.append(snap(at) if not shift else float(f"{snap(at) - shift:.3f}"))
        at += _number(clip.get("dur"))
    return starts


@dataclass(frozen=True)
class Cut:
    start: int
    end: int
    overlap: int
    head: int


def cuts(clips: Sequence[Mapping[str, Any]], frames: Callable[[Any], int]) -> List[Optional[Cut]]:
    places = placement(clips)
    if not any(place is not None and place.overlap for place in places):
        return [None] * len(clips)
    starts = [frames(place.start) if place is not None else 0 for place in places]
    ends = [frames(place.end) if place is not None else 0 for place in places]
    overlaps = [0] * len(clips)
    for index in range(1, len(clips)):
        if places[index] is not None and places[index].overlap:
            ends[index - 1] = max(ends[index - 1], starts[index] + 1)
            overlaps[index] = ends[index - 1] - starts[index]
    edits = [start + overlap // 2 for start, overlap in zip(starts, overlaps)]
    result: List[Optional[Cut]] = []
    for index, clip in enumerate(clips):
        place = places[index]
        following = index + 1 < len(clips) and overlaps[index + 1] > 0
        if place is None or not (overlaps[index] or following or moved(clip, place)):
            result.append(None)
            continue
        start = edits[index] if overlaps[index] else starts[index]
        end = edits[index + 1] if following else ends[index]
        result.append(Cut(start, end, overlaps[index], start - starts[index]))
    return result

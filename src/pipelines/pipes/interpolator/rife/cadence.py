"""Output-frame schedule of a frame-rate change: which source frame, or which instant between two, each output frame shows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

_EPSILON = Fraction(1, 10**6)
_SNAP = 1e-3
_DENOMINATOR = 100000


@dataclass(frozen=True)
class Slot:
    """One output frame: the source frame on its left and how far it sits toward the next one, 0 when it falls on the source frame."""

    index: int
    left: int
    t: float


def source_per_output(src_fps: float, factor: int = 2, target_fps: float = 0.0) -> Fraction:
    """Source frames per output frame: 1/factor in factor mode, src_fps/target_fps when a target rate is set."""
    if target_fps and target_fps > 0:
        return Fraction(src_fps).limit_denominator(_DENOMINATOR) / Fraction(target_fps).limit_denominator(_DENOMINATOR)
    return Fraction(1, int(factor))


def output_fps(src_fps: float, factor: int = 2, target_fps: float = 0.0) -> float:
    """The output frame rate: target_fps when set, else src_fps times factor."""
    return float(target_fps) if target_fps and target_fps > 0 else float(src_fps) * int(factor)


def output_count(n_source: int, ratio: Fraction) -> int:
    """Output frames for n_source source frames: round(n_source x out_fps / src_fps), at least 1, 0 for an empty clip."""
    if n_source <= 0:
        return 0
    return max(1, round(Fraction(n_source) / ratio))


def slot(index: int, ratio: Fraction) -> Slot:
    """Where output frame ``index`` falls on the source timeline."""
    position = index * ratio
    left = math.floor(position + _EPSILON)
    t = float(position - left)
    return Slot(index, left, 0.0 if t < _SNAP else t)

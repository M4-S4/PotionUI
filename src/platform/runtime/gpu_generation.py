from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple


class GpuGeneration(Enum):
    NONE = "none"
    PRE_AMPERE = "pre_ampere"
    AMPERE = "ampere"
    ADA = "ada"
    HOPPER = "hopper"
    BLACKWELL = "blackwell"
    UNKNOWN = "unknown"


def generation_for(capability: Optional[Tuple[int, int]]) -> GpuGeneration:
    if not capability:
        return GpuGeneration.NONE
    major, minor = capability
    if major <= 0:
        return GpuGeneration.NONE
    if major < 8:
        return GpuGeneration.PRE_AMPERE
    if major == 8:
        return GpuGeneration.ADA if minor == 9 else GpuGeneration.AMPERE
    if major == 9:
        return GpuGeneration.HOPPER
    if major in (10, 11, 12):
        return GpuGeneration.BLACKWELL
    return GpuGeneration.UNKNOWN

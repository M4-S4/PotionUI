from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from threading import Lock

from .errors import ModelToolsError
from .format import FP8_DTYPES, SIDECAR_SUFFIXES
from .source import read_header
from .spec import TensorSpec

QUANT_MARKERS = ("scaled_fp8",)
QUANT_KEY_SUFFIXES = (*SIDECAR_SUFFIXES, ".SCB", ".qweight", ".scales")
SAFETENSORS_SUFFIXES = (".safetensors", ".sft")
CACHE_SIZE = 4096


def is_quantized_plan(plan: Sequence[TensorSpec]) -> bool:
    for spec in plan:
        if spec.dtype in FP8_DTYPES or spec.key in QUANT_MARKERS or spec.key.endswith(QUANT_KEY_SUFFIXES):
            return True
    return False


_cache: OrderedDict[tuple[str, int, int], bool | None] = OrderedDict()
_lock = Lock()


def _signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def is_unquantized_file(path: str | os.PathLike[str]) -> bool:
    path = Path(path)
    if path.suffix.lower() not in SAFETENSORS_SUFFIXES:
        return False
    try:
        key = _signature(path)
    except OSError:
        return False
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return bool(_cache[key])
    try:
        plan, _ = read_header(path)
        value: bool | None = not is_quantized_plan(plan)
    except (ModelToolsError, OSError, ValueError):
        value = None
    with _lock:
        _cache[key] = value
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return bool(value)

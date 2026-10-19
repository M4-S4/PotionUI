from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open

from src.platform.runtime.model_headers.reader import read_safetensors_header

from .errors import SourceFormatError
from .spec import DTYPE_BYTES, TensorSpec


def _fail(path: Path, key: str | None, why: str) -> SourceFormatError:
    where = f"{path.name}: tensor {key!r}" if key is not None else path.name
    return SourceFormatError(f"{where}: {why}")


def _parse_entry(path: Path, key: str, info: object, data_size: int) -> tuple[int, TensorSpec]:
    try:
        if not isinstance(info, dict):
            raise ValueError("entry is not an object")
        dtype = info["dtype"]
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype {dtype!r}")
        shape = info["shape"]
        if not isinstance(shape, list) or any(
            isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in shape
        ):
            raise ValueError("shape is not a list of non-negative integers")
        offsets = info["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(o, bool) or not isinstance(o, int) or o < 0 for o in offsets)
        ):
            raise ValueError("data_offsets is not two non-negative integers")
        start, end = offsets
        spec = TensorSpec(key, dtype, tuple(shape))
        if end - start != spec.nbytes:
            raise ValueError("data_offsets disagree with shape and dtype")
        if end > data_size:
            raise ValueError("data extends past the end of the file")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise _fail(path, key, f"invalid header entry ({exc})") from exc
    return start, spec


def read_header(path: str | Path) -> tuple[list[TensorSpec], dict[str, str]]:
    path = Path(path)
    try:
        header = read_safetensors_header(path)
    except ValueError as exc:
        raise _fail(path, None, f"unreadable safetensors header ({exc})") from exc
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
    data_size = path.stat().st_size - 8 - header_len
    raw_meta = header.pop("__metadata__", None)
    metadata = {str(k): str(v) for k, v in raw_meta.items()} if isinstance(raw_meta, dict) else {}
    entries = [_parse_entry(path, key, info, data_size) for key, info in header.items()]
    entries.sort(key=lambda e: e[0])
    return [spec for _, spec in entries], metadata


def _iter_tensors(path: Path, plan: list[TensorSpec]) -> Iterator[tuple[str, torch.Tensor]]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for spec in plan:
            tensor = f.get_tensor(spec.key).contiguous()
            yield spec.key, tensor
            del tensor


def read_stream(
    path: str | Path,
) -> tuple[list[TensorSpec], Iterator[tuple[str, torch.Tensor]], dict[str, str]]:
    path = Path(path)
    plan, metadata = read_header(path)
    return plan, _iter_tensors(path, plan), metadata

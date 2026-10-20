from __future__ import annotations

import json
import os
import struct
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from .errors import StreamMismatchError, WriteCancelledError
from .spec import TORCH_TO_DTYPE, TensorSpec

ALIGNMENT = 8
SYNC_EVERY_BYTES = 256 * 1024 * 1024

_FADVISE_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", None)


@dataclass(frozen=True, slots=True)
class WriteResult:
    path: Path
    size_bytes: int
    tensor_count: int


def build_header(plan: Iterable[TensorSpec], metadata: dict[str, str]) -> bytes:
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    for spec in plan:
        if spec.key in header:
            raise StreamMismatchError(f"duplicate tensor key in plan: {spec.key}")
        end = offset + spec.nbytes
        header[spec.key] = {"dtype": spec.dtype, "shape": list(spec.shape), "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    pad = -len(raw) % ALIGNMENT
    return raw + b" " * pad


def _check(spec: TensorSpec, key: str, tensor: torch.Tensor) -> None:
    if key != spec.key:
        raise StreamMismatchError(f"stream yielded {key!r} where the plan expects {spec.key!r}")
    dtype = TORCH_TO_DTYPE.get(tensor.dtype)
    if dtype != spec.dtype:
        raise StreamMismatchError(f"{spec.key}: dtype {dtype} does not match planned {spec.dtype}")
    if tuple(tensor.shape) != spec.shape:
        raise StreamMismatchError(f"{spec.key}: shape {tuple(tensor.shape)} does not match planned {spec.shape}")
    if tensor.device.type != "cpu":
        raise StreamMismatchError(f"{spec.key}: tensor is on {tensor.device}, streams must be CPU")


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    flat = tensor.detach().contiguous().reshape(-1)
    return memoryview(flat.view(torch.uint8).numpy())


def _sync_range(f, start: int) -> int:
    f.flush()
    end = f.tell()
    fd = f.fileno()
    sync = getattr(os, "fdatasync", os.fsync)
    sync(fd)
    if _FADVISE_DONTNEED is not None and hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, start, end - start, _FADVISE_DONTNEED)
        except OSError:
            pass
    return end


def write_stream(
    dest: str | Path,
    plan: list[TensorSpec],
    tensors: Iterator[tuple[str, torch.Tensor]],
    metadata: dict[str, str],
    *,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> WriteResult:
    dest = Path(dest)
    plan = list(plan)
    header = build_header(plan, metadata)
    total = sum(spec.nbytes for spec in plan)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.{uuid.uuid4().hex[:8]}.tmp"
    written = 0
    synced_written = 0
    synced_pos = 0
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            iterator = iter(tensors)
            for spec in plan:
                if is_cancelled is not None and is_cancelled():
                    raise WriteCancelledError(f"writing {dest.name} was cancelled")
                try:
                    key, tensor = next(iterator)
                except StopIteration:
                    raise StreamMismatchError(f"stream ended before {spec.key!r}") from None
                _check(spec, key, tensor)
                data = _tensor_bytes(tensor)
                if len(data) != spec.nbytes:
                    raise StreamMismatchError(f"{spec.key}: {len(data)} bytes where {spec.nbytes} are planned")
                f.write(data)
                written += len(data)
                if written - synced_written >= SYNC_EVERY_BYTES:
                    synced_pos = _sync_range(f, synced_pos)
                    synced_written = written
                del data, tensor
                if on_progress is not None:
                    on_progress(written, total)
            if next(iterator, None) is not None:
                raise StreamMismatchError("stream yielded more tensors than the plan")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        close = getattr(tensors, "close", None)
        if close is not None:
            close()
        tmp.unlink(missing_ok=True)
        raise
    return WriteResult(path=dest, size_bytes=8 + len(header) + total, tensor_count=len(plan))


__all__ = ["ALIGNMENT", "WriteResult", "build_header", "write_stream"]

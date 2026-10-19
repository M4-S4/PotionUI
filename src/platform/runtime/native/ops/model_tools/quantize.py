from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import torch

from ..fp8_quant import quantize_tensor_to_fp8_scaled
from .errors import AlreadyQuantizedError, StreamMismatchError, WriteCancelledError
from .format import (
    DESCRIPTOR_BYTES,
    DESCRIPTOR_SUFFIX,
    FP8_DTYPES,
    QUANT_DESCRIPTOR,
    SCALE_SUFFIX,
    SIDECAR_SUFFIXES,
    WEIGHT_SUFFIX,
)
from .families import detect_family, is_kept, keep_patterns
from .source import read_stream
from .spec import FLOAT_DTYPES, TensorSpec
from .writer import WriteResult, write_stream

def _check_not_quantized(plan: Sequence[TensorSpec]) -> None:
    for spec in plan:
        if spec.dtype in FP8_DTYPES or spec.key.endswith(SIDECAR_SUFFIXES) or spec.key == "scaled_fp8":
            raise AlreadyQuantizedError(f"source already carries quantized data ({spec.key})")


def _should_quantize(spec: TensorSpec, patterns, keep_extra: Sequence[str]) -> bool:
    if not spec.key.endswith(WEIGHT_SUFFIX) or len(spec.shape) != 2 or spec.dtype not in FLOAT_DTYPES:
        return False
    return not is_kept(spec.key, patterns, keep_extra)


def quantize_plan(
    plan: Sequence[TensorSpec], family: str, keep_extra: Sequence[str] = ()
) -> tuple[list[TensorSpec], set[str]]:
    patterns = keep_patterns(family)
    _check_not_quantized(plan)
    out: list[TensorSpec] = []
    quantized: set[str] = set()
    for spec in plan:
        if _should_quantize(spec, patterns, keep_extra):
            base = spec.key[: -len(WEIGHT_SUFFIX)]
            out.append(TensorSpec(spec.key, "F8_E4M3", spec.shape))
            out.append(TensorSpec(base + SCALE_SUFFIX, "F32", ()))
            out.append(TensorSpec(base + DESCRIPTOR_SUFFIX, "U8", (len(DESCRIPTOR_BYTES),)))
            quantized.add(spec.key)
        else:
            out.append(spec)
    return out, quantized


def _quantized_tensors(
    tensors: Iterator[tuple[str, torch.Tensor]], quantized: set[str]
) -> Iterator[tuple[str, torch.Tensor]]:
    descriptor = torch.frombuffer(bytearray(DESCRIPTOR_BYTES), dtype=torch.uint8)
    for key, tensor in tensors:
        if key not in quantized:
            yield key, tensor
            continue
        q, scale = quantize_tensor_to_fp8_scaled(tensor)
        del tensor
        base = key[: -len(WEIGHT_SUFFIX)]
        yield key, q.contiguous()
        del q
        yield base + SCALE_SUFFIX, scale.to(torch.float32).reshape(()).contiguous()
        yield base + DESCRIPTOR_SUFFIX, descriptor.clone()


def quantize_stream(
    plan: Sequence[TensorSpec],
    tensors: Iterator[tuple[str, torch.Tensor]],
    family: str,
    keep_extra: Sequence[str] = (),
) -> tuple[list[TensorSpec], Iterator[tuple[str, torch.Tensor]]]:
    new_plan, quantized = quantize_plan(plan, family, keep_extra)
    return new_plan, _quantized_tensors(tensors, quantized)


HASH_CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: str | Path, is_cancelled: Callable[[], bool] | None = None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK_BYTES):
            if is_cancelled is not None and is_cancelled():
                raise WriteCancelledError(f"hashing {Path(path).name} was cancelled")
            digest.update(chunk)
    return digest.hexdigest()


def build_recipe(family: str, keep_extra: Sequence[str], plan: Sequence[TensorSpec]) -> dict:
    layers = sum(1 for s in plan if s.key.endswith(DESCRIPTOR_SUFFIX))
    return {
        "tool": "quantize",
        "format": QUANT_DESCRIPTOR["format"],
        "scaling": "per_tensor_absmax",
        "family": family,
        "keep_extra": list(keep_extra),
        "quantized_layers": layers,
    }


def quantized_metadata(
    source_metadata: dict[str, str],
    source_sha256: str,
    family: str,
    keep_extra: Sequence[str],
    plan: Sequence[TensorSpec],
) -> dict[str, str]:
    meta = dict(source_metadata)
    meta["potionui.quantized_from"] = source_sha256
    meta["potionui.recipe"] = json.dumps(build_recipe(family, keep_extra, plan), sort_keys=True)
    return meta


def quantize_file(
    source: str | Path,
    dest: str | Path,
    *,
    family: str | None = None,
    keep_extra: Sequence[str] = (),
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> WriteResult:
    source = Path(source)
    if Path(dest).resolve() == source.resolve():
        raise StreamMismatchError("the destination must differ from the source")
    plan, tensors, metadata = read_stream(source)
    family = family or detect_family(plan, metadata)
    new_plan, new_tensors = quantize_stream(plan, tensors, family, keep_extra)
    digest = sha256_file(source)
    meta = quantized_metadata(metadata, digest, family, keep_extra, new_plan)
    return write_stream(dest, new_plan, new_tensors, meta, on_progress=on_progress, is_cancelled=is_cancelled)

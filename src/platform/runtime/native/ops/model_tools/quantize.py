from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import torch

from .errors import AlreadyQuantizedError, StreamMismatchError, UnsupportedFormatError, WriteCancelledError
from .format import DESCRIPTOR_SUFFIX, FP8_DTYPES, SIDECAR_SUFFIXES, WEIGHT_SUFFIX
from .formats import DEFAULT_FORMAT, QUANT_FORMATS, QuantFormat, get_format
from .families import TE_FAMILY_KEEP, detect_family, is_kept, keep_patterns
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


def _check_format_family(fmt: QuantFormat, family: str) -> None:
    if family in TE_FAMILY_KEEP and not fmt.text_encoders:
        allowed = ", ".join(f.label for f in QUANT_FORMATS.values() if f.text_encoders)
        raise UnsupportedFormatError(f"text encoders can be quantized to {allowed} only, not {fmt.label}")


def _candidates(plan: Sequence[TensorSpec], family: str, keep_extra: Sequence[str]) -> list[TensorSpec]:
    patterns = keep_patterns(family)
    return [spec for spec in plan if _should_quantize(spec, patterns, keep_extra)]


def shape_kept_layers(
    plan: Sequence[TensorSpec], family: str, keep_extra: Sequence[str], fmt: QuantFormat
) -> list[str]:
    return [spec.key for spec in _candidates(plan, family, keep_extra) if not fmt.eligible(spec.shape)]


def quantize_plan(
    plan: Sequence[TensorSpec], family: str, keep_extra: Sequence[str] = (), fmt: QuantFormat | None = None
) -> tuple[list[TensorSpec], set[str]]:
    fmt = fmt or get_format(DEFAULT_FORMAT)
    _check_format_family(fmt, family)
    candidates = {spec.key for spec in _candidates(plan, family, keep_extra) if fmt.eligible(spec.shape)}
    _check_not_quantized(plan)
    out: list[TensorSpec] = []
    quantized: set[str] = set()
    for spec in plan:
        if spec.key in candidates:
            base = spec.key[: -len(WEIGHT_SUFFIX)]
            out.extend(TensorSpec(base + suffix, dtype, shape) for suffix, dtype, shape in fmt.layer_specs(spec.shape))
            out.append(TensorSpec(base + DESCRIPTOR_SUFFIX, "U8", (len(fmt.descriptor_bytes),)))
            quantized.add(spec.key)
        else:
            out.append(spec)
    return out, quantized


def _quantized_tensors(
    tensors: Iterator[tuple[str, torch.Tensor]], quantized: set[str], fmt: QuantFormat
) -> Iterator[tuple[str, torch.Tensor]]:
    descriptor = torch.frombuffer(bytearray(fmt.descriptor_bytes), dtype=torch.uint8)
    for key, tensor in tensors:
        if key not in quantized:
            yield key, tensor
            continue
        parts = fmt.quantize(tensor)
        del tensor
        base = key[: -len(WEIGHT_SUFFIX)]
        while parts:
            suffix, part = parts.pop(0)
            yield base + suffix, part
            del part
        yield base + DESCRIPTOR_SUFFIX, descriptor.clone()


def quantize_stream(
    plan: Sequence[TensorSpec],
    tensors: Iterator[tuple[str, torch.Tensor]],
    family: str,
    keep_extra: Sequence[str] = (),
    fmt: QuantFormat | None = None,
) -> tuple[list[TensorSpec], Iterator[tuple[str, torch.Tensor]]]:
    fmt = fmt or get_format(DEFAULT_FORMAT)
    new_plan, quantized = quantize_plan(plan, family, keep_extra, fmt)
    return new_plan, _quantized_tensors(tensors, quantized, fmt)


HASH_CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: str | Path, is_cancelled: Callable[[], bool] | None = None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK_BYTES):
            if is_cancelled is not None and is_cancelled():
                raise WriteCancelledError(f"hashing {Path(path).name} was cancelled")
            digest.update(chunk)
    return digest.hexdigest()


def build_recipe(family: str, keep_extra: Sequence[str], plan: Sequence[TensorSpec], fmt: QuantFormat) -> dict:
    layers = sum(1 for s in plan if s.key.endswith(DESCRIPTOR_SUFFIX))
    return {
        "tool": "quantize",
        "format": fmt.descriptor["format"],
        "scaling": fmt.scaling,
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
    fmt: QuantFormat,
) -> dict[str, str]:
    meta = dict(source_metadata)
    meta["potionui.quantized_from"] = source_sha256
    meta["potionui.recipe"] = json.dumps(build_recipe(family, keep_extra, plan, fmt), sort_keys=True)
    return meta


def quantize_file(
    source: str | Path,
    dest: str | Path,
    *,
    family: str | None = None,
    keep_extra: Sequence[str] = (),
    quant_format: str = DEFAULT_FORMAT,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> WriteResult:
    source = Path(source)
    if Path(dest).resolve() == source.resolve():
        raise StreamMismatchError("the destination must differ from the source")
    fmt = get_format(quant_format)
    plan, tensors, metadata = read_stream(source)
    family = family or detect_family(plan, metadata)
    new_plan, new_tensors = quantize_stream(plan, tensors, family, keep_extra, fmt)
    digest = sha256_file(source)
    meta = quantized_metadata(metadata, digest, family, keep_extra, new_plan, fmt)
    return write_stream(dest, new_plan, new_tensors, meta, on_progress=on_progress, is_cancelled=is_cancelled)

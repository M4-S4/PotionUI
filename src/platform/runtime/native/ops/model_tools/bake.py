from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from ...lora.key_mapping import LoraDelta, map_lora_keys
from .delta import compute_delta
from .lora_io import (
    BakeError,
    LoraInput,
    is_text_encoder_stem,
    load_lora_state,
    refusal_for_unsupported_kind,
)
from .skeleton import Skeleton, SourcePart, build_skeleton
from .source import read_stream
from .spec import TensorSpec

FP8_DTYPES = ("F8_E4M3", "F8_E5M2")
FLOAT_DTYPES = ("BF16", "F16", "F32")
SCALE_SUFFIXES = (".weight_scale", ".scale_weight")
FP8_SIDECAR_SUFFIXES = (".weight_scale", ".scale_weight", ".input_scale", ".scale_input", ".comfy_quant")
FP8_FORMATS = ("float8_e4m3fn", "float8_e5m2")
DEQUANT_DTYPE = "BF16"


@dataclass
class BakeReport:
    family: str
    variant: str
    output_dtype: str | None
    dequantized_fp8: bool
    keys_touched: dict[str, list[str]] = field(default_factory=dict)
    unmatched: dict[str, list[str]] = field(default_factory=dict)
    dropped_keys: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class _Job:
    param: str
    part: SourcePart


def _scale_delta(d: LoraDelta, strength: float) -> LoraDelta:
    return LoraDelta(
        down=d.down, up=d.up, alpha=d.alpha, scale=d.scale * float(strength),
        target_slice=d.target_slice, kron=d.kron, kron_factors=d.kron_factors,
    )


def _check_delta_shape(d: LoraDelta, dest: tuple[int, ...], label: str) -> None:
    rows = d.target_slice[2] if d.target_slice is not None else dest[0]
    if d.kron:
        natural = (d.up.shape[0] * d.down.shape[0], d.up.shape[1] * d.down.shape[1])
    else:
        if d.up.ndim != 2 or d.down.ndim != 2 or d.up.shape[1] != d.down.shape[0]:
            raise BakeError(f"{label}: malformed LoRA factors up {tuple(d.up.shape)} down {tuple(d.down.shape)}")
        natural = (d.up.shape[0], d.down.shape[1])
    if natural != (rows, dest[1]):
        raise BakeError(f"{label}: shape mismatch, adapter delta is {natural} but the target is {(rows, dest[1])}")


def _collect_deltas(
    skeleton: Skeleton, loras: Sequence[LoraInput], allow_unmatched: bool
) -> tuple[dict[str, list[LoraDelta]], dict[str, list[str]], dict[str, list[str]]]:
    per_param: dict[str, list[LoraDelta]] = defaultdict(list)
    touched: dict[str, list[str]] = {}
    unmatched_report: dict[str, list[str]] = {}
    for lora in loras:
        name = Path(lora.path).name
        sd = load_lora_state(lora.path)
        refusal = refusal_for_unsupported_kind(name, {k: tuple(v.shape) for k, v in sd.items()})
        if refusal is not None:
            raise BakeError(refusal)
        mapped, unmatched = map_lora_keys(sd, skeleton.module)
        te = [u for u in unmatched if is_text_encoder_stem(u)]
        if te:
            raise BakeError(
                f"{name}: text-encoder LoRA keys are not supported when baking into a diffusion model "
                f"({len(te)} found, e.g. {te[0]}); remove them first",
                {lora.path: te},
            )
        if unmatched:
            sample = ", ".join(unmatched[:5])
            if not allow_unmatched:
                raise BakeError(
                    f"{name}: {len(unmatched)} LoRA key(s) do not match this {skeleton.family} model "
                    f"(shape mismatch or wrong family), e.g. {sample}",
                    {lora.path: list(unmatched)},
                )
            unmatched_report[lora.path] = list(unmatched)
        if not mapped:
            raise BakeError(f"{name}: no LoRA key matches this {skeleton.family} model")
        touched[lora.path] = []
        for param, deltas in mapped.items():
            shape = skeleton.param_shape(param)
            for d in deltas:
                _check_delta_shape(d, shape, f"{name}:{param}")
                per_param[param].append(_scale_delta(d, lora.strength))
            touched[lora.path].append(param)
    return per_param, touched, unmatched_report


def _fp8_scale_key(stem: str, sources: dict[str, TensorSpec]) -> str | None:
    for suffix in SCALE_SUFFIXES:
        if stem + suffix in sources:
            return stem + suffix
    return None


def _check_quant_descriptors(model_path: str, sources: dict[str, TensorSpec], fp8_stems: list[str]) -> None:
    keys = [s + ".comfy_quant" for s in fp8_stems if s + ".comfy_quant" in sources]
    if not keys:
        return
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in keys:
            blob = f.get_tensor(key)
            try:
                fmt = json.loads(bytes(blob.tolist()).decode("utf-8")).get("format")
            except ValueError:
                fmt = None
            if fmt not in FP8_FORMATS:
                raise BakeError(f"{key}: quantization format {fmt!r} cannot be dequantized (expected fp8)")


def _dequantize(weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    out = weight.to(torch.float32)
    if scale is None:
        return out
    scale = scale.to(torch.float32)
    if scale.ndim == 1 and scale.shape[0] == out.shape[0] and out.ndim == 2:
        scale = scale.reshape(-1, 1)
    return out * scale


def bake_stream(
    model_path: str | Path,
    loras: Sequence[LoraInput],
    family: str,
    *,
    allow_unmatched: bool = False,
    device: str | None = None,
) -> tuple[list[TensorSpec], Iterator[tuple[str, torch.Tensor]], BakeReport]:
    model_path = str(model_path)
    if not loras:
        raise BakeError("no LoRA was given")
    plan, source_tensors, metadata = read_stream(model_path)
    skeleton = build_skeleton(plan, metadata, family)
    sources = skeleton.sources
    per_param, touched, unmatched = _collect_deltas(skeleton, loras, allow_unmatched)

    jobs: dict[str, _Job] = {}
    parts_left: dict[str, int] = {}
    for param in per_param:
        parts = skeleton.locate(param)
        parts_left[param] = len(parts)
        for part in parts:
            spec = sources[part.file_key]
            if spec.dtype not in FLOAT_DTYPES + FP8_DTYPES:
                raise BakeError(
                    f"{part.file_key}: stored as {spec.dtype}, a quantized format that cannot take a baked LoRA"
                )
            jobs[part.file_key] = _Job(param, part)

    fp8_weights = [
        s.key for s in plan
        if s.key.startswith(skeleton.prefix) and s.key.endswith(".weight") and s.dtype in FP8_DTYPES
    ]
    fp8_stems = [k[: -len(".weight")] for k in fp8_weights]
    _check_quant_descriptors(model_path, sources, fp8_stems)
    dropped: set[str] = set()
    for stem in fp8_stems:
        dropped.update(stem + suffix for suffix in FP8_SIDECAR_SUFFIXES if stem + suffix in sources)
    if fp8_weights and "scaled_fp8" in sources:
        dropped.add("scaled_fp8")
    fp8_set = set(fp8_weights)

    out_plan: list[TensorSpec] = []
    touched_dtypes: list[str] = []
    for spec in plan:
        if spec.key in dropped:
            continue
        if spec.key in fp8_set:
            out_plan.append(TensorSpec(spec.key, DEQUANT_DTYPE, spec.shape))
            if spec.key in jobs:
                touched_dtypes.append(DEQUANT_DTYPE)
        else:
            out_plan.append(spec)
            if spec.key in jobs:
                touched_dtypes.append(spec.dtype)

    out_metadata = {k: v for k, v in metadata.items() if not (fp8_weights and k == "_quantization_metadata")}
    output_dtype = max(set(touched_dtypes), key=touched_dtypes.count) if touched_dtypes else None
    report = BakeReport(
        family=skeleton.family,
        variant=skeleton.variant,
        output_dtype=DEQUANT_DTYPE if fp8_weights else output_dtype,
        dequantized_fp8=bool(fp8_weights),
        keys_touched={
            path: [part.file_key for param in params for part in skeleton.locate(param)]
            for path, params in touched.items()
        },
        unmatched=unmatched,
        dropped_keys=sorted(dropped),
        metadata=out_metadata,
    )

    compute_device = device or "cpu"
    delta_cache: dict[str, torch.Tensor] = {}

    def delta_for(job: _Job) -> torch.Tensor:
        full = delta_cache.get(job.param)
        if full is None:
            full = compute_delta(per_param[job.param], skeleton.param_shape(job.param), compute_device)
            delta_cache[job.param] = full
        piece = full[job.part.row_start: job.part.row_start + job.part.rows].to("cpu")
        parts_left[job.param] -= 1
        if parts_left[job.param] == 0:
            del delta_cache[job.param]
        return piece

    def generate() -> Iterator[tuple[str, torch.Tensor]]:
        scale_file = safe_open(model_path, framework="pt", device="cpu") if fp8_weights else None
        try:
            for key, tensor in source_tensors:
                if key in dropped:
                    continue
                job = jobs.get(key)
                if key in fp8_set:
                    stem = key[: -len(".weight")]
                    scale_key = _fp8_scale_key(stem, sources)
                    scale = scale_file.get_tensor(scale_key) if scale_key else None
                    value = _dequantize(tensor, scale)
                    if job is not None:
                        value = value + delta_for(job).to(value.device)
                    yield key, value.to(torch.bfloat16).contiguous()
                elif job is not None:
                    value = tensor.to(torch.float32) + delta_for(job).to(tensor.device)
                    yield key, value.to(tensor.dtype).contiguous()
                else:
                    yield key, tensor
        finally:
            if scale_file is not None:
                scale_file.__exit__(None, None, None)

    return out_plan, generate(), report

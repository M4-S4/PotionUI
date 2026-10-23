import json
import math
import os
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.platform.runtime.native.ops.model_tools.bake import bake_stream
from src.platform.runtime.native.ops.model_tools.errors import WriteCancelledError
from src.platform.runtime.native.ops.model_tools.families import FAMILY_KEEP, detect_family
from src.platform.runtime.native.ops.model_tools.format import DESCRIPTOR_SUFFIX, WEIGHT_SUFFIX
from src.platform.runtime.native.ops.model_tools.formats import QuantFormat, get_format
from src.platform.runtime.native.ops.model_tools.lora_io import LoraInput
from src.platform.runtime.native.ops.model_tools.merge import merge_loras
from src.platform.runtime.native.ops.model_tools.quantize import (
    quantize_stream,
    quantized_metadata,
    sha256_file,
    shape_kept_layers,
)
from src.platform.runtime.native.ops.model_tools.source import read_header, read_stream
from src.platform.runtime.native.ops.model_tools.spec import TensorSpec
from src.platform.runtime.native.ops.model_tools.verify import LayerReport, VerifyReport, layer_metrics, verify_quantized
from src.platform.runtime.native.ops.model_tools.writer import WriteResult, write_stream

from src.pipelines.pipes.model_tools.errors import ModelToolsRefused

HASH_CHUNK_BYTES = 8 * 1024 * 1024
QUANTIZE_VERIFY_SAMPLE = 32
BAKE_VERIFY_SAMPLE = 3
WORST_LAYERS = 5
LICENCE_KEYS = ("license", "licence", "modelspec.license", "ss_license")
LICENCE_NOTE_LIMIT = 300
TOOL_TAG = "model_tools"

Notify = Callable[[str, int, int], None]


@dataclass(frozen=True)
class JobRequest:
    mode: str
    model: Path | None
    loras: tuple[LoraInput, ...]
    keep_extra: tuple[str, ...]
    merge_method: str
    rank: int | None
    min_energy: float
    dtype: str
    family: str | None
    quant_format: str


@dataclass
class Job:
    mode: str
    kind: str
    model_type: str
    plan: list[TensorSpec]
    stream: Iterator[tuple[str, torch.Tensor]]
    base_metadata: dict[str, str]
    sources: list[Path]
    format_label: str
    describe: Callable[[], dict[str, Any]]
    verify: Callable[[Path], dict[str, Any]]
    source_size_bytes: int
    source_metadatas: list[dict[str, str]] = field(default_factory=list)
    quant: tuple[str, tuple[str, ...], list[TensorSpec], QuantFormat] | None = None

    @property
    def planned_bytes(self) -> int:
        return sum(spec.nbytes for spec in self.plan)


@dataclass
class Published:
    path: Path
    size_bytes: int
    summary: dict[str, Any]


def licence_note(metadatas: Sequence[dict[str, str]]) -> str | None:
    found: list[str] = []
    for metadata in metadatas:
        for key, value in metadata.items():
            if key.lower() in LICENCE_KEYS and str(value).strip():
                text = str(value).strip()
                if text not in found:
                    found.append(text)
                break
    if not found:
        return None
    return "; ".join(found)[:LICENCE_NOTE_LIMIT]


def _layer_dicts(layers: Sequence[LayerReport]) -> list[dict[str, Any]]:
    worst = sorted(layers, key=lambda l: l.rel_l2, reverse=True)[:WORST_LAYERS]
    return [
        {"key": l.key, "rel_l2": l.rel_l2, "cosine": l.cosine, "snr_db": l.snr_db}
        for l in worst
    ]


def _quality(layers: Sequence[LayerReport]) -> dict[str, Any]:
    if not layers:
        return {}
    rels = [l.rel_l2 for l in layers]
    return {
        "mean_rel_l2": sum(rels) / len(rels),
        "max_rel_l2": max(rels),
        "min_cosine": min(l.cosine for l in layers),
        "min_snr_db": min(l.snr_db for l in layers),
        "worst_layers": _layer_dicts(layers),
    }


class QuantProbe:
    def __init__(self, quantized_weights: set[str], fmt: QuantFormat) -> None:
        self.quantized_weights = quantized_weights
        self.fmt = fmt
        self.layers: list[LayerReport] = []
        self._current: tuple[str, torch.Tensor] | None = None
        self._pending: tuple[str, dict[str, torch.Tensor]] | None = None

    def tap_input(self, tensors: Iterator[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        for key, tensor in tensors:
            self._current = (key, tensor)
            yield key, tensor

    def tap_output(self, tensors: Iterator[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        for key, tensor in tensors:
            current = self._current
            if key in self.quantized_weights and current is not None and current[0] == key:
                self._pending = (key, {WEIGHT_SUFFIX: tensor})
            elif self._pending is not None and current is not None:
                weight_key, parts = self._pending
                base = weight_key[: -len(WEIGHT_SUFFIX)]
                if key == base + DESCRIPTOR_SUFFIX:
                    restored = self.fmt.restore(parts, tuple(current[1].shape))
                    self.layers.append(layer_metrics(weight_key, current[1], restored))
                    self._pending = None
                elif key.startswith(base + "."):
                    parts[key[len(base):]] = tensor
            yield key, tensor
        self._current = None
        self._pending = None


def _quantized_weights(plan: Sequence[TensorSpec]) -> set[str]:
    return {
        s.key[: -len(DESCRIPTOR_SUFFIX)] + WEIGHT_SUFFIX for s in plan if s.key.endswith(DESCRIPTOR_SUFFIX)
    }


def _counts(source_plan: Sequence[TensorSpec], quantized: set[str]) -> tuple[int, int]:
    kept = sum(1 for s in source_plan if s.key not in quantized)
    return len(quantized), kept


def _verify_report_dict(report: VerifyReport) -> dict[str, Any]:
    if not report.ok:
        problems = []
        if report.missing:
            problems.append(f"{len(report.missing)} source tensor(s) missing from the output, e.g. {report.missing[0]}")
        if report.kept_mismatched:
            problems.append(
                f"{len(report.kept_mismatched)} kept tensor(s) differ from the source, e.g. {report.kept_mismatched[0]}"
            )
        if not report.layers:
            problems.append("no quantized layer could be checked")
        raise ModelToolsRefused("Verification failed: " + "; ".join(problems))
    if not math.isfinite(report.max_rel_l2):
        raise ModelToolsRefused("Verification failed: a quantized layer produced a non-finite error")
    return {
        "checked_layers": len(report.layers),
        "sampled": report.sampled,
        "kept_checked": report.kept_checked,
        "mean_rel_l2": report.mean_rel_l2,
        "max_rel_l2": report.max_rel_l2,
        "min_cosine": report.min_cosine,
        "min_snr_db": report.min_snr_db,
        "worst_layers": _layer_dicts(report.worst),
    }


def _quality_note(fmt: QuantFormat) -> dict[str, str]:
    return {"quality_note": fmt.quality_note} if fmt.quality_note else {}


def _resolve_quant_family(request: JobRequest, plan: Sequence[TensorSpec], metadata: dict[str, str]) -> str:
    return request.family or detect_family(plan, metadata)


def _prepare_quantize(request: JobRequest) -> Job:
    assert request.model is not None
    source = request.model
    fmt = get_format(request.quant_format)
    plan, tensors, metadata = read_stream(source)
    family = _resolve_quant_family(request, plan, metadata)
    new_plan, new_tensors = quantize_stream(plan, tensors, family, request.keep_extra, fmt)
    quantized = _quantized_weights(new_plan)
    layers, kept = _counts(plan, quantized)
    shape_kept = len(shape_kept_layers(plan, family, request.keep_extra, fmt))
    quant = (family, tuple(request.keep_extra), new_plan, fmt)

    def describe() -> dict[str, Any]:
        return {
            "family": family,
            "layers_quantized": layers,
            "layers_kept": kept,
            "layers_shape_kept": shape_kept,
            **_quality_note(fmt),
        }

    def verify(staged: Path) -> dict[str, Any]:
        return _verify_report_dict(verify_quantized(source, staged, sample_limit=QUANTIZE_VERIFY_SAMPLE))

    return Job(
        mode=request.mode, kind="model", model_type="", plan=new_plan, stream=new_tensors,
        base_metadata=metadata, sources=[source], format_label=fmt.label,
        describe=describe, verify=verify, source_size_bytes=source.stat().st_size,
        source_metadatas=[metadata], quant=quant,
    )


def _check_baked(source: Path, staged: Path, touched: Sequence[str]) -> dict[str, Any]:
    from safetensors import safe_open

    picks = _spread(sorted(set(touched)), BAKE_VERIFY_SAMPLE)
    deltas: list[float] = []
    with safe_open(str(source), framework="pt", device="cpu") as before, safe_open(
        str(staged), framework="pt", device="cpu"
    ) as after:
        for key in picks:
            a = before.get_tensor(key).to(torch.float32)
            b = after.get_tensor(key).to(torch.float32)
            if a.shape != b.shape:
                raise ModelToolsRefused(f"Verification failed: {key} changed shape")
            if not bool(torch.isfinite(b).all()):
                raise ModelToolsRefused(f"Verification failed: {key} holds non-finite values after the merge")
            norm = torch.linalg.vector_norm(a).item()
            change = torch.linalg.vector_norm(b - a).item()
            if change == 0.0:
                raise ModelToolsRefused(f"Verification failed: the LoRA delta did not change {key}")
            deltas.append(change / norm if norm > 0 else math.inf)
    return {
        "checked_layers": len(picks),
        "mean_rel_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "max_rel_delta": max(deltas) if deltas else 0.0,
    }


def _spread(items: list[str], limit: int) -> list[str]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def _bake(request: JobRequest) -> tuple[list[TensorSpec], Iterator[tuple[str, torch.Tensor]], Any]:
    assert request.model is not None
    return bake_stream(request.model, list(request.loras), request.family)


def _prepare_bake(request: JobRequest) -> Job:
    assert request.model is not None
    source = request.model
    plan, stream, report = _bake(request)
    touched = sorted({key for keys in report.keys_touched.values() for key in keys})
    metadata = dict(report.metadata)
    dtype = (report.output_dtype or "").lower()

    def describe() -> dict[str, Any]:
        return {
            "family": report.family,
            "layers_merged": len(touched),
            "dequantized_fp8": report.dequantized_fp8,
            "output_dtype": report.output_dtype,
        }

    def verify(staged: Path) -> dict[str, Any]:
        return _check_baked(source, staged, touched)

    return Job(
        mode=request.mode, kind="model", model_type="", plan=plan, stream=stream,
        base_metadata=metadata, sources=[source, *[Path(l.path) for l in request.loras]],
        format_label=dtype or "merged", describe=describe, verify=verify,
        source_size_bytes=source.stat().st_size,
        source_metadatas=[read_header(source)[1]],
    )


def _prepare_bake_quantize(request: JobRequest) -> Job:
    assert request.model is not None
    source = request.model
    plan, baked, report = _bake(request)
    family = request.family or (report.family if report.family in FAMILY_KEEP else detect_family(plan, report.metadata))
    fmt = get_format(request.quant_format)
    touched = sorted({key for keys in report.keys_touched.values() for key in keys})
    probe = QuantProbe(set(), fmt)
    new_plan, new_stream = quantize_stream(plan, probe.tap_input(baked), family, request.keep_extra, fmt)
    quantized = _quantized_weights(new_plan)
    probe.quantized_weights = quantized
    stream = probe.tap_output(new_stream)
    layers, kept = _counts(plan, quantized)
    shape_kept = len(shape_kept_layers(plan, family, request.keep_extra, fmt))
    quant = (family, tuple(request.keep_extra), new_plan, fmt)

    def describe() -> dict[str, Any]:
        return {
            "family": family,
            "layers_quantized": layers,
            "layers_kept": kept,
            "layers_shape_kept": shape_kept,
            **_quality_note(fmt),
            "layers_merged": len(touched),
            "dequantized_fp8": report.dequantized_fp8,
        }

    def verify(staged: Path) -> dict[str, Any]:
        staged_plan, _ = read_header(staged)
        if {s.key for s in staged_plan} != {s.key for s in new_plan}:
            raise ModelToolsRefused("Verification failed: the written tensors differ from the plan")
        if not probe.layers:
            raise ModelToolsRefused("Verification failed: no quantized layer could be checked")
        if any(not math.isfinite(l.rel_l2) for l in probe.layers):
            raise ModelToolsRefused("Verification failed: a quantized layer produced a non-finite error")
        return {"checked_layers": len(probe.layers), "reference": "merged fp32", **_quality(probe.layers)}

    return Job(
        mode=request.mode, kind="model", model_type="", plan=new_plan, stream=stream,
        base_metadata=dict(report.metadata), sources=[source, *[Path(l.path) for l in request.loras]],
        format_label=fmt.label, describe=describe, verify=verify,
        source_size_bytes=source.stat().st_size, source_metadatas=[read_header(source)[1]], quant=quant,
    )


def _prepare_merge_loras(request: JobRequest) -> Job:
    plan, stream, report = merge_loras(
        list(request.loras),
        request.merge_method,
        request.rank,
        request.min_energy,
        dtype=request.dtype,
        family=request.family,
    )
    sources = [Path(l.path) for l in request.loras]

    def describe() -> dict[str, Any]:
        return {
            "family": report.family,
            "rank": report.rank,
            "layers_merged": report.layers,
            "energy_kept": report.min_energy_kept,
            "method": report.mode,
            "output_dtype": report.output_dtype,
            "warnings": list(report.warnings),
        }

    def verify(staged: Path) -> dict[str, Any]:
        staged_plan, _ = read_header(staged)
        if {s.key for s in staged_plan} != {s.key for s in plan}:
            raise ModelToolsRefused("Verification failed: the written tensors differ from the plan")
        energy = report.min_energy_kept
        if energy is not None and not math.isfinite(energy):
            raise ModelToolsRefused("Verification failed: the kept energy is not finite")
        return {"checked_layers": report.layers, "min_energy_kept": energy}

    return Job(
        mode=request.mode, kind="lora", model_type="lora", plan=plan, stream=stream,
        base_metadata=dict(report.metadata), sources=sources, format_label=request.dtype.lower(),
        describe=describe, verify=verify, source_size_bytes=sum(p.stat().st_size for p in sources),
        source_metadatas=[read_header(p)[1] for p in sources],
    )


PREPARERS: dict[str, Callable[[JobRequest], Job]] = {
    "quantize": _prepare_quantize,
    "merge_into_model": _prepare_bake,
    "merge_and_quantize": _prepare_bake_quantize,
    "merge_loras": _prepare_merge_loras,
}


def prepare(request: JobRequest) -> Job:
    return PREPARERS[request.mode](request)


def stamp_metadata(job: Job, request: JobRequest, digests: dict[Path, str]) -> dict[str, str]:
    meta = dict(job.base_metadata)
    meta["potionui.tool"] = TOOL_TAG
    meta["potionui.mode"] = job.mode
    if request.model is not None:
        meta["potionui.source_model"] = json.dumps(
            {"name": request.model.name, "sha256": digests[request.model]}, sort_keys=True
        )
    if request.loras:
        meta["potionui.loras"] = json.dumps(
            [
                {"name": Path(l.path).name, "sha256": digests[Path(l.path)], "strength": l.strength}
                for l in request.loras
            ]
        )
    if job.mode in ("merge_loras", "merge_into_model", "merge_and_quantize"):
        meta["potionui.merge"] = json.dumps(
            {"method": request.merge_method if job.mode == "merge_loras" else "bake", "rank": request.rank},
            sort_keys=True,
        )
    if job.quant is not None and request.model is not None:
        family, keep_extra, plan, fmt = job.quant
        meta = quantized_metadata(meta, digests[request.model], family, keep_extra, plan, fmt)
    note = licence_note(job.source_metadatas)
    if note:
        meta["potionui.licence_note"] = note
    return meta


def hash_sources(paths: Sequence[Path], is_cancelled: Callable[[], bool] | None, notify: Notify) -> dict[Path, str]:
    digests: dict[Path, str] = {}
    total = sum(p.stat().st_size for p in paths) or 1
    done = 0
    for path in paths:
        digests[path] = sha256_file(path, is_cancelled)
        done += path.stat().st_size
        notify("hashing", done, total)
    return digests


def _sources_summary(request: JobRequest, digests: dict[Path, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if request.model is not None:
        out.append({"name": request.model.name, "sha256": digests[request.model], "strength": None})
    for lora in request.loras:
        out.append({"name": Path(lora.path).name, "sha256": digests[Path(lora.path)], "strength": lora.strength})
    return out


def _place(staged: Path, destination: Path, overwrite: bool) -> None:
    if overwrite:
        os.replace(staged, destination)
        return
    exists = ModelToolsRefused(
        f"{destination.name} appeared while the output was being written; it was left untouched"
    )
    try:
        os.link(staged, destination)
    except FileExistsError:
        raise exists from None
    except OSError:
        if os.path.lexists(destination):
            raise exists from None
        os.replace(staged, destination)
        return
    staged.unlink(missing_ok=True)


def staging_path(final: Path) -> Path:
    return final.with_name(f".{final.name}.{uuid.uuid4().hex[:8]}.staged")


def publish(
    job: Job,
    request: JobRequest,
    destination: Path,
    digests: dict[Path, str],
    *,
    notify: Notify,
    is_cancelled: Callable[[], bool] | None,
    overwrite: bool = False,
) -> Published:
    metadata = stamp_metadata(job, request, digests)
    staged = staging_path(destination)

    def on_progress(written: int, total: int) -> None:
        notify("writing", written, total)

    try:
        result: WriteResult = write_stream(
            staged, job.plan, job.stream, metadata, on_progress=on_progress, is_cancelled=is_cancelled
        )
        notify("verifying", 0, 1)
        verification = job.verify(staged)
        if is_cancelled is not None and is_cancelled():
            raise WriteCancelledError(f"publishing {destination.name} was cancelled")
        _place(staged, destination, overwrite)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    summary: dict[str, Any] = {
        "mode": job.mode,
        "sources": _sources_summary(request, digests),
        "licence_note": licence_note(job.source_metadatas),
        "verification": verification,
        **job.describe(),
    }
    for key in ("worst_layers", "mean_rel_l2", "max_rel_l2", "min_cosine"):
        if key in verification:
            summary[key] = verification[key]
    if "min_energy_kept" in verification and "energy_kept" not in summary:
        summary["energy_kept"] = verification["min_energy_kept"]
    return Published(path=destination, size_bytes=result.size_bytes, summary=_jsonable(summary))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

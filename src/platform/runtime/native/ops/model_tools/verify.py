from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from .format import DESCRIPTOR_SUFFIX, WEIGHT_SUFFIX
from .formats import format_for_descriptor, parse_descriptor
from .source import read_header

SNR_CAP_DB = 200.0
WORST_COUNT = 5


@dataclass(frozen=True, slots=True)
class LayerReport:
    key: str
    rel_l2: float
    cosine: float
    snr_db: float


@dataclass(slots=True)
class VerifyReport:
    layers: list[LayerReport] = field(default_factory=list)
    worst: list[LayerReport] = field(default_factory=list)
    mean_rel_l2: float = 0.0
    max_rel_l2: float = 0.0
    min_cosine: float = 1.0
    min_snr_db: float = SNR_CAP_DB
    kept_checked: int = 0
    kept_mismatched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    sampled: bool = False

    @property
    def ok(self) -> bool:
        return not self.kept_mismatched and not self.missing and bool(self.layers)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["ok"] = self.ok
        return data


def _pick(items: list[str], limit: int | None) -> list[str]:
    if limit is None or limit >= len(items):
        return items
    if limit <= 0:
        return []
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def layer_metrics(key: str, original: torch.Tensor, restored: torch.Tensor) -> LayerReport:
    a = original.to(torch.float32).reshape(-1)
    b = restored.to(torch.float32).reshape(-1)
    err = torch.linalg.vector_norm(a - b).item()
    norm_a = torch.linalg.vector_norm(a).item()
    norm_b = torch.linalg.vector_norm(b).item()
    rel = err / norm_a if norm_a > 0 else (0.0 if err == 0 else math.inf)
    denom = norm_a * norm_b
    cosine = float(torch.dot(a, b).item() / denom) if denom > 0 else (1.0 if err == 0 else 0.0)
    snr = SNR_CAP_DB if err == 0 else min(SNR_CAP_DB, 20.0 * math.log10(norm_a / err)) if norm_a > 0 else -SNR_CAP_DB
    return LayerReport(key=key, rel_l2=rel, cosine=cosine, snr_db=snr)


def verify_quantized(
    source_path: str | Path,
    quantized_path: str | Path,
    *,
    sample_limit: int | None = None,
) -> VerifyReport:
    source_plan, _ = read_header(source_path)
    shapes = {s.key: s.shape for s in source_plan}
    quant_plan, _ = read_header(quantized_path)
    quant_keys = {s.key for s in quant_plan}
    quantized = [
        s.key for s in source_plan
        if s.key.endswith(WEIGHT_SUFFIX)
        and s.key[: -len(WEIGHT_SUFFIX)] + DESCRIPTOR_SUFFIX in quant_keys
    ]
    quantized_set = set(quantized)
    kept = [s.key for s in source_plan if s.key not in quantized_set]
    report = VerifyReport(sampled=sample_limit is not None)
    report.missing = [s.key for s in source_plan if s.key not in quant_keys]
    chosen_q = _pick(quantized, sample_limit)
    chosen_k = [k for k in _pick(kept, sample_limit) if k in quant_keys]
    with safe_open(str(source_path), framework="pt", device="cpu") as src, safe_open(
        str(quantized_path), framework="pt", device="cpu"
    ) as dst:
        for key in chosen_q:
            base = key[: -len(WEIGHT_SUFFIX)]
            fmt = format_for_descriptor(parse_descriptor(dst.get_tensor(base + DESCRIPTOR_SUFFIX)))
            shape = shapes[key]
            parts = {suffix: dst.get_tensor(base + suffix) for suffix, _, _ in fmt.layer_specs(shape)}
            restored = fmt.restore(parts, shape)
            report.layers.append(layer_metrics(key, src.get_tensor(key), restored))
            del parts, restored
        for key in chosen_k:
            a = src.get_tensor(key)
            b = dst.get_tensor(key)
            report.kept_checked += 1
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(
                a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)
            ):
                report.kept_mismatched.append(key)
            del a, b
    if report.layers:
        rels = [l.rel_l2 for l in report.layers]
        report.mean_rel_l2 = sum(rels) / len(rels)
        report.max_rel_l2 = max(rels)
        report.min_cosine = min(l.cosine for l in report.layers)
        report.min_snr_db = min(l.snr_db for l in report.layers)
        report.worst = sorted(report.layers, key=lambda l: l.rel_l2, reverse=True)[:WORST_COUNT]
    return report

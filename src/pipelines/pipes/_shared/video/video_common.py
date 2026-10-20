from __future__ import annotations

from typing import Any, Dict, List
import torch
from src.pipelines.pipes._shared.generation.generator_base import emit_gallery


def emit_video_results(
        generation_outputs: callable,
        results: List[str],
        used_seeds: List[int],
        resolution: "tuple[int, int] | None" = None,
) -> None:
    emit_gallery(generation_outputs, images=[], seeds=used_seeds, videos=results, video_resolution=resolution)


def build_video_output(results: List[str]) -> Dict[str, Any]:
    return {"video": results}


def to_device(cond: dict, device, dtype) -> dict:
    return {
        k: (v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device=device))
        for k, v in cond.items()
    }

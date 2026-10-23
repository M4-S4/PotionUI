from __future__ import annotations

from typing import Any
from src.platform.observability.profiling import get_profiler
from src.pipelines.contracts import logger


def te_ram_gb(te_native_model: Any) -> float:
    module = getattr(te_native_model, "module", None)
    if module is None or not hasattr(module, "parameters"):
        return 0.0
    try:
        return sum(p.numel() * p.element_size() for p in module.parameters()) / (1 << 30)
    except Exception:
        return 0.0


def release_idle_te(bundle: Any, models: Any, log_prefix: str) -> float:
    key = getattr(bundle, "te_cache_key", None)
    if not key or models is None:
        return 0.0
    evict = getattr(models, "evict_dead_weight", None)
    if not callable(evict):
        return 0.0
    te_gb = te_ram_gb(getattr(bundle, "te", None))
    try:
        unloaded = evict(key)
    except Exception:
        logger.debug("[%s] TE eviction failed for key=%r", log_prefix, key, exc_info=True)
        return 0.0
    get_profiler().mark(
        "ltx_generator.te_evicted", pipe=log_prefix, key=key,
        unloaded=bool(unloaded), te_gb=round(te_gb, 2) if unloaded else 0.0,
    )
    if unloaded:
        logger.debug("[%s] evicted idle TE cache entry key=%r (~%.2fGB)", log_prefix, key, te_gb)
    return te_gb if unloaded else 0.0

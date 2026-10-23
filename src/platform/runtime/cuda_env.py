from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

CUDA_CACHE_MAXSIZE_BYTES = 4 * 1024**3
CUDA_CACHE_SUBDIR = "cuda_cache"


def apply_cuda_env_defaults(
    environ: MutableMapping[str, str] | None = None,
    platform: str | None = None,
) -> dict[str, str]:
    env = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    applied: dict[str, str] = {}

    def set_default(key: str, value: str) -> None:
        if key not in env:
            env[key] = value
            applied[key] = value

    set_default("CUDA_CACHE_MAXSIZE", str(CUDA_CACHE_MAXSIZE_BYTES))

    if "CUDA_CACHE_PATH" not in env:
        cache_dir = Path(env.get("POTIONUI_STORAGE_PATH") or "storage").resolve() / CUDA_CACHE_SUBDIR
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        else:
            set_default("CUDA_CACHE_PATH", str(cache_dir))

    if not platform.startswith("win"):
        set_default("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    return applied

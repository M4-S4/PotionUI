"""Hardware-derived defaults for the auto-provisioned native backend.

`NativeBackendConfig` used to hardcode device="cuda", dtype="float16",
gpu_max_vram=8 regardless of what the host actually has (or lacks). A GPU-less
box silently got a cuda/float16/8GB config that fails at generation time, and a
90GB-class card was capped at 8GB. This module inspects the host once and
derives values that actually match it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.platform.observability.logger import logger


@dataclass(frozen=True)
class NativeHardwareDefaults:
    """What a freshly auto-provisioned native backend should default to."""
    device: str
    dtype: str
    gpu_max_vram: int


@lru_cache(maxsize=1)
def detect_native_hardware_defaults() -> NativeHardwareDefaults:
    try:
        from src.platform.runtime.native.kernels import capabilities
    except ImportError:
        logger.warning("[NATIVE_HARDWARE] torch not importable; defaulting device=cpu dtype=float32")
        return NativeHardwareDefaults(device="cpu", dtype="float32", gpu_max_vram=0)

    snapshot = capabilities()
    if not snapshot.devices:
        logger.info("[NATIVE_HARDWARE] No CUDA device detected; defaulting device=cpu dtype=float32")
        return NativeHardwareDefaults(device="cpu", dtype="float32", gpu_max_vram=0)

    from src.platform.runtime.native.memory.residency import (
        minimum_inference_memory_gb,
        total_vram_gb,
    )

    gpu = snapshot.device(0)
    major, minor = gpu.capability
    dtype = "bfloat16" if gpu.at_least(8) else "float16"

    total_gb = total_vram_gb("cuda") or 0.0
    reserve_gb = minimum_inference_memory_gb()
    gpu_max_vram = max(1, int(total_gb - reserve_gb))

    logger.info(
        "[NATIVE_HARDWARE] Detected %s, %.1fGB VRAM, sm%d%d; defaulting "
        "device=cuda dtype=%s gpu_max_vram=%dGB (%.1fGB total - %.1fGB reserve)",
        gpu.name, total_gb, major, minor, dtype, gpu_max_vram, total_gb, reserve_gb,
    )
    return NativeHardwareDefaults(device="cuda", dtype=dtype, gpu_max_vram=gpu_max_vram)

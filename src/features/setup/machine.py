"""Host facts the setup wizard shows before it offers a starter model.

Every probe is independent and best-effort: a failed probe yields `None` for
its part, never an exception, so the endpoint works on a CPU-only box, in CI
and with every plugin off.
"""

import shutil
from typing import Any, Callable, Optional

from src.features.setup.dto import MachineGpu, MachineInfo, ModelsDisk
from src.platform.observability.logger import logger
from src.platform.runtime.gpu_profile import detect_gpu_profile
from src.platform.runtime.system_memory import get_system_memory

MIN_SUPPORTED_COMPUTE_CAPABILITY = (7, 5)
_BYTES_PER_GIB = 1024 ** 3


def _probe_gpu() -> Optional[MachineGpu]:
    try:
        profile = detect_gpu_profile()
        if not profile.has_gpu:
            return None
        capability = profile.compute_capability
        return MachineGpu(
            name=profile.name,
            vram_bytes=int(round(profile.vram_gb * _BYTES_PER_GIB)),
            compute_capability=f"{capability[0]}.{capability[1]}" if capability else None,
            supported=bool(capability) and tuple(capability) >= MIN_SUPPORTED_COMPUTE_CAPABILITY,
        )
    except Exception as exc:
        logger.warning("Setup machine probe: GPU detection failed (%s)", exc)
        return None


def _probe_ram() -> Optional[int]:
    try:
        total = int(get_system_memory().total)
        return total if total > 0 else None
    except Exception as exc:
        logger.warning("Setup machine probe: RAM detection failed (%s)", exc)
        return None


def _probe_models_disk(model_roots: Any) -> Optional[ModelsDisk]:
    try:
        home = model_roots.home_dir()
        usage = shutil.disk_usage(home)
        return ModelsDisk(path=str(home), free_bytes=int(usage.free), total_bytes=int(usage.total))
    except Exception as exc:
        logger.warning("Setup machine probe: models disk usage failed (%s)", exc)
        return None


def probe_machine(model_roots: Any) -> MachineInfo:
    return MachineInfo(
        gpu=_probe_gpu(),
        ram_total_bytes=_probe_ram(),
        models_disk=_probe_models_disk(model_roots),
    )

"""Cgroup-aware system RAM reading.

``psutil.virtual_memory()`` reports the HOST's RAM — even from inside a
memory-limited container (``docker run --memory``, a k8s
``resources.limits.memory``, ...). Every admission decision in this codebase
that budgets against "how much system RAM is there" (model-lifecycle LRU
eviction headroom, standalone-upscale RAM floor, ...) reads through
``psutil.virtual_memory()`` directly, so inside a memory-limited container
those decisions overshoot the real ceiling: they keep loading/caching until
the HOST looks full, at which point the container's cgroup — not the
model-lifecycle admission control — is what stops it, via the OOM killer.

This is exactly the failure mode the Docker rig-simulation harness
(``docker/``) exists to catch, and it is a real bug on the eventual 0.0.2
Docker distribution too, not just a test aid: any user running PotionUI in a
memory-capped container hits it.

``get_system_memory()`` is the one shared seam: it returns ``psutil``'s
reading clamped to the tightest memory limit of the process's own cgroup and
of its ancestors, read from ``/proc/self/cgroup`` (so a container sharing the
host's cgroup namespace, or a capped systemd slice, is seen too). Cgroup v2
(``memory.max`` / ``memory.current``) comes first; cgroup v1
(``memory.limit_in_bytes`` / ``memory.usage_in_bytes``) is the fallback for
hosts that still mount it. A cgroup's usage counts its page cache, so its
inactive file pages (``memory.stat``), which the kernel reclaims before any
OOM kill, count as free, as the kubelet's working set does. Without any limit
it is a plain passthrough, so call sites that route through it are unaffected
outside a memory-limited cgroup.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

DEFAULT_CGROUP_ROOT = Path("/sys/fs/cgroup")
DEFAULT_PROC_CGROUP_PATH = Path("/proc/self/cgroup")

_UNLIMITED = "max"

_V1_UNLIMITED_FLOOR = 1 << 60

_V2_FILES = ("memory.max", "memory.current", "inactive_file")
_V1_FILES = ("memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file")

# Logged once per process the first time a real cgroup limit is found, so a
# capped container run is visible in logs without spamming every call site.
_warned_once = False


@dataclass(frozen=True)
class SystemMemory:
    """Bytes, matching ``psutil.virtual_memory()``'s ``total``/``available``."""

    total: int
    available: int

    @property
    def total_gb(self) -> float:
        return self.total / (1024 ** 3)

    @property
    def available_gb(self) -> float:
        return self.available / (1024 ** 3)


def _read_int_file(path: Path) -> Optional[int]:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if raw == _UNLIMITED:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_stat(path: Path, key: str) -> int:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return 0
    for line in lines:
        name, _, value = line.partition(" ")
        if name == key:
            try:
                return max(0, int(value))
            except ValueError:
                return 0
    return 0


def _own_cgroups(proc_cgroup_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """The process's cgroup v2 path and its cgroup v1 memory path, from ``/proc/self/cgroup``."""
    try:
        lines = proc_cgroup_path.read_text().splitlines()
    except OSError:
        return None, None
    unified = memory = None
    for line in lines:
        hierarchy, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if hierarchy == "0" and controllers == "":
            unified = path or "/"
        elif "memory" in controllers.split(","):
            memory = path or "/"
    return unified, memory


def _levels(mount: Path, own: Optional[str]) -> List[Path]:
    """The cgroup folders from the process's own up to the mount root; the root alone when the own one is not visible there."""
    levels = [mount]
    parts = [part for part in (own or "/").split("/") if part not in ("", ".")]
    if ".." in parts:
        return levels
    for depth in range(1, len(parts) + 1):
        folder = mount.joinpath(*parts[:depth])
        if not folder.is_dir():
            break
        levels.append(folder)
    return levels[::-1]


def _tightest(levels: List[Path], files: Tuple[str, str, str], available: int) -> Optional[Tuple[int, int]]:
    """``(limit, headroom)`` of the level whose free memory is the smallest, or ``None`` when no level has a limit."""
    limit_name, usage_name, inactive_key = files
    found: Optional[Tuple[int, int]] = None
    for folder in levels:
        limit = _read_int_file(folder / limit_name)
        if limit is None or limit >= _V1_UNLIMITED_FLOOR:
            continue
        usage = _read_int_file(folder / usage_name)
        if usage is None:
            headroom = available
        else:
            headroom = max(0, limit - max(0, usage - _read_stat(folder / "memory.stat", inactive_key)))
        if found is None:
            found = (limit, headroom)
        else:
            found = (min(found[0], limit), min(found[1], headroom))
    return found


def get_system_memory(
    *,
    cgroup_root: Path = DEFAULT_CGROUP_ROOT,
    proc_cgroup_path: Path = DEFAULT_PROC_CGROUP_PATH,
) -> SystemMemory:
    """``(total, available)`` system RAM in bytes, cgroup-aware.

    - No cgroup limit visible (files missing/unreadable, or every limit is
      unlimited) -> returns ``psutil.virtual_memory()``'s
      ``total``/``available`` unchanged.
    - A numeric limit on the process's cgroup or on one of its ancestors
      (cgroup v2, else cgroup v1) -> ``total`` is clamped to the smallest
      limit, and ``available`` to the smallest ``limit - (usage - inactive
      file pages)`` of those cgroups, a level without a readable usage
      counting as psutil's own reading. Either way the result is never larger
      than what psutil itself reported.
    """
    vm = psutil.virtual_memory()
    total = vm.total
    available = vm.available

    unified, memory = _own_cgroups(proc_cgroup_path)
    version = "v2"
    found = _tightest(_levels(cgroup_root, unified), _V2_FILES, available)
    if found is None:
        version = "v1"
        found = _tightest(_levels(cgroup_root / "memory", memory), _V1_FILES, available)
    if found is None:
        return SystemMemory(total=total, available=available)

    limit, headroom = found
    capped_total = min(total, limit)
    capped_available = min(headroom, capped_total, available)

    global _warned_once
    if not _warned_once:
        logger.warning(
            "[SYSTEM_MEMORY] cgroup %s memory limit detected: %.2fGB "
            "(host RAM: %.2fGB) - RAM-budgeting decisions will use the "
            "container limit, not host RAM",
            version,
            limit / (1024 ** 3),
            total / (1024 ** 3),
        )
        _warned_once = True

    return SystemMemory(total=capped_total, available=capped_available)


def reset_warning_for_tests() -> None:
    """Clear the one-time-warning flag. Test-only."""
    global _warned_once
    _warned_once = False

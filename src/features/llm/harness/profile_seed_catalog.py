import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.features.llm.harness.profile import ProfileLayer
from src.features.llm.harness.profile_seed import (
    BASE_PATH,
    CONTENT_DIR,
    PLUGIN_ROOT_PREFIX,
    AppliedSeed,
    SeedKey,
    SeedRoot,
    SeedTable,
    local_root,
    marketplace_root,
)
from src.platform.plugins.manifest_roots import plugin_manifest_roots

RECHECK_SECONDS = 2.0

_Signature = Tuple[Tuple[str, str, Tuple[Tuple[str, int], ...]], ...]


def plugin_profile_roots(manifests) -> List[SeedRoot]:
    return [
        SeedRoot(f"{PLUGIN_ROOT_PREFIX}{root.plugin_id}", root.path)
        for root in plugin_manifest_roots(manifests, "llm_profiles")
    ]


def _stamp(path: Path) -> Tuple[str, int]:
    try:
        return path.name, path.stat().st_mtime_ns
    except OSError:
        return path.name, -1


def _signature(roots: List[SeedRoot]) -> _Signature:
    return tuple(
        (root.name, root.path.as_posix(), tuple(_stamp(path) for path in sorted(root.path.glob("*.yml"))))
        for root in roots
    )


class SeedCatalog:
    def __init__(
        self,
        content_dir: Path = CONTENT_DIR,
        plugin_registry=None,
        base_path: Path = BASE_PATH,
        *,
        clock: Callable[[], float] = time.monotonic,
        recheck_seconds: float = RECHECK_SECONDS,
    ) -> None:
        self._content_dir = Path(content_dir)
        self._plugin_registry = plugin_registry
        self._base_path = base_path
        self._clock = clock
        self._recheck_seconds = recheck_seconds
        self._checked_at = 0.0
        self._lock = threading.Lock()
        self._signature: Optional[_Signature] = None
        self._table: Optional[SeedTable] = None

    def roots(self) -> List[SeedRoot]:
        roots = [marketplace_root(self._content_dir)]
        if self._plugin_registry is not None:
            roots.extend(plugin_profile_roots(self._plugin_registry.get_enabled_plugins()))
        roots.append(local_root(self._content_dir))
        return roots

    def table(self) -> SeedTable:
        with self._lock:
            now = self._clock()
            if self._table is not None and now - self._checked_at < self._recheck_seconds:
                return self._table
            roots = self.roots()
            signature = _signature(roots)
            if self._table is None or signature != self._signature:
                self._table = SeedTable.scan(roots, self._base_path)
                self._signature = signature
            self._checked_at = now
            return self._table

    def layer(self, key: SeedKey) -> ProfileLayer:
        return self.table().layer(key)

    def applied(self, key: SeedKey) -> Tuple[AppliedSeed, ...]:
        return self.table().applied(key)

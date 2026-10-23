import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Set

from src.features.backends.backend_config import NATIVE_ENGINE, NATIVE_LOCAL_DRIVER
from src.features.models.read_throttle import ReadThrottle
from src.platform.runtime.thread_priority import ThreadPriority
from src.platform.settings.repository import SettingRepository

logger = logging.getLogger(__name__)

LEVEL_LOW = "low"
LEVEL_NORMAL = "normal"
LEVEL_MAX = "max"
DEFAULT_LEVEL = LEVEL_LOW

INDEXING_LEVEL_SETTING_KEY = "model_indexing_level"

_MIB = 1024 * 1024


@dataclass(frozen=True)
class IndexingProfile:
    level: str
    workers: int
    priority: ThreadPriority
    read_bytes_per_second: Optional[int]
    pauses_for_generation: bool
    evict_page_cache: bool


PROFILES = {
    LEVEL_LOW: IndexingProfile(
        LEVEL_LOW, workers=1, priority=ThreadPriority.BACKGROUND,
        read_bytes_per_second=96 * _MIB, pauses_for_generation=True, evict_page_cache=True,
    ),
    LEVEL_NORMAL: IndexingProfile(
        LEVEL_NORMAL, workers=2, priority=ThreadPriority.BELOW_NORMAL,
        read_bytes_per_second=384 * _MIB, pauses_for_generation=True, evict_page_cache=True,
    ),
    LEVEL_MAX: IndexingProfile(
        LEVEL_MAX, workers=4, priority=ThreadPriority.NORMAL,
        read_bytes_per_second=None, pauses_for_generation=False, evict_page_cache=False,
    ),
}


class InvalidIndexingLevel(ValueError):
    def __init__(self, value):
        super().__init__(f"'{value}' is not an indexing level; use one of: {', '.join(PROFILES)}")
        self.value = value


def normalize_level(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    return candidate if candidate in PROFILES else None


def local_native_backend_ids(backend_registry) -> Set[str]:
    if backend_registry is None:
        return set()
    try:
        backends = backend_registry.get_backends_for_engine(NATIVE_ENGINE)
    except Exception:
        return set()
    return {b.backend_id for b in backends if getattr(b.config, "driver", "") == NATIVE_LOCAL_DRIVER}


def local_generation_probe(
    status_tracker, local_backend_ids: Callable[[], Iterable[str]]
) -> Callable[[], bool]:
    def busy() -> bool:
        local = set(local_backend_ids())
        if not local:
            return False
        for record in status_tracker.list_active():
            state = getattr(record.state, "value", record.state)
            if state == "running" and record.backend_id in local:
                return True
        return False

    return busy


class IndexingGovernor:
    def __init__(
        self,
        setting_repository: Optional[SettingRepository] = None,
        generation_busy: Optional[Callable[[], bool]] = None,
        default_level: str = DEFAULT_LEVEL,
        throttle: Optional[ReadThrottle] = None,
    ):
        self._repository = setting_repository
        self._level = default_level
        self.throttle = throttle or ReadThrottle(busy=generation_busy)
        self.refresh()

    @classmethod
    def unmanaged(cls) -> "IndexingGovernor":
        return cls(default_level=LEVEL_MAX)

    def level(self) -> str:
        if self._repository is None:
            return self._level
        try:
            setting = self._repository.get_setting_by_key(INDEXING_LEVEL_SETTING_KEY)
            stored = normalize_level(setting.get_typed_value()) if setting is not None else None
        except Exception:
            logger.warning("Could not read the model indexing level; using the default", exc_info=True)
            stored = None
        return stored or self._level

    def profile(self) -> IndexingProfile:
        return PROFILES[self.level()]

    def refresh(self) -> IndexingProfile:
        profile = self.profile()
        self.throttle.configure(
            profile.read_bytes_per_second, profile.pauses_for_generation, profile.evict_page_cache
        )
        return profile

    def verification_throttle(self) -> ReadThrottle:
        profile = self.profile()
        return ReadThrottle(
            bytes_per_second=profile.read_bytes_per_second,
            evict_page_cache=profile.evict_page_cache,
        )

    def set_level(self, level) -> IndexingProfile:
        normalized = normalize_level(level)
        if normalized is None:
            raise InvalidIndexingLevel(level)
        if self._repository is None:
            self._level = normalized
        else:
            self._persist(normalized)
        return self.refresh()

    def _persist(self, level: str) -> None:
        if not self._repository.update_setting_value_by_key(INDEXING_LEVEL_SETTING_KEY, level):
            raise RuntimeError(f"The '{INDEXING_LEVEL_SETTING_KEY}' setting row is missing")

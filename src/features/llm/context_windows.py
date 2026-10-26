import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from src.features.llm import context_budget
from src.features.llm.model_listing import CONTEXT_WINDOW_PROBE_TIMEOUT_SECONDS, DiscoveredWindow
from src.features.llm.repository import LLMConfig
from src.features.llm.ttl_cache import TTLCache
from src.features.llm.types import LLMTypes
from src.platform.database.rows import now_utc

CONTEXT_WINDOW_HIT_TTL_SECONDS = 3600.0
CONTEXT_WINDOW_MISS_TTL_SECONDS = 300.0

WindowKey = Tuple[str, str, str]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowProbe:
    window: Optional[DiscoveredWindow]
    discovered_at: datetime


class ContextWindows:
    def __init__(self, types: LLMTypes) -> None:
        self._types = types
        self._hits: TTLCache[WindowKey, WindowProbe] = TTLCache(CONTEXT_WINDOW_HIT_TTL_SECONDS)
        self._misses: TTLCache[WindowKey, WindowProbe] = TTLCache(CONTEXT_WINDOW_MISS_TTL_SECONDS)

    @staticmethod
    def key(type_id: str, base_url: str, model: str) -> WindowKey:
        return (type_id, (base_url or "").strip().rstrip("/"), model)

    def supports(self, type_id: str) -> bool:
        entry = self._types.get(type_id)
        return entry is not None and callable(getattr(entry.adapter, "discover_context_window", None))

    async def refresh(self, type_id: str, base_url: str, api_key: Optional[str], model: str) -> WindowProbe:
        key = self.key(type_id, base_url, model)
        window: Optional[DiscoveredWindow] = None
        try:
            window = await asyncio.wait_for(
                self._types.require(type_id).adapter.discover_context_window(base_url, api_key, model),
                timeout=CONTEXT_WINDOW_PROBE_TIMEOUT_SECONDS + 1.0,
            )
        except Exception:
            logger.debug("[ContextWindows] context window probe failed for %s", type_id, exc_info=True)
        probe = WindowProbe(window=window, discovered_at=now_utc())
        self._hits.invalidate(key)
        self._misses.invalidate(key)
        (self._hits if window is not None else self._misses).set(key, probe)
        return probe

    async def ensure(self, config: LLMConfig) -> None:
        try:
            if context_budget.resolve_capacity(config).source == "config":
                return
            if not self.supports(config.type):
                return
            key = self.key(config.type, config.base_url, config.model)
            if self._hits.get(key) or self._misses.get(key):
                return
            await self.refresh(config.type, config.base_url, config.api_key, config.model)
        except Exception:
            logger.debug("[ContextWindows] ensure failed for '%s'", getattr(config, "id", None), exc_info=True)

    def discovered(self, config: LLMConfig) -> Optional[DiscoveredWindow]:
        try:
            probe = self._hits.get(self.key(config.type, config.base_url, config.model))
        except Exception:
            return None
        return probe.window if probe is not None else None

    def capacity(self, config: LLMConfig) -> context_budget.CapacityInfo:
        return context_budget.resolve_capacity(config, self.discovered(config))

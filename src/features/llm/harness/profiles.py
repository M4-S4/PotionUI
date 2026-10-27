from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Tuple

from src.features.llm.harness import codec
from src.features.llm.harness.adapter import Adapter
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource, Verification
from src.features.llm.harness.profile_schema import PROFILE_FIELDS, clean_overrides
from src.features.llm.harness.profile_seed import SeedKey, SeedTable, default_seed
from src.features.llm.model_listing import ModelListingError
from src.features.llm.ttl_cache import TTLCache
from src.platform.database.rows import dt_iso, now_utc

if TYPE_CHECKING:
    from src.features.llm.repository import LLMConfig

logger = logging.getLogger(__name__)

DISCOVERY_HIT_TTL_SECONDS = 3600.0
DISCOVERY_MISS_TTL_SECONDS = 300.0
DISCOVERY_TIMEOUT_SECONDS = 10.0
NOT_DISCOVERABLE = frozenset({"context_request"})

DiscoveryKey = Tuple[str, str, str]


@dataclass(frozen=True)
class Discovery:
    layer: ProfileLayer
    at: datetime
    error: Optional[str] = None


@dataclass(frozen=True)
class FieldReport:
    value: Any
    source: ProfileSource
    seed: Any
    discovered: Any
    override: Any


@dataclass(frozen=True)
class ProfileReport:
    config_id: str
    adapter: str
    model: str
    discovered_at: Optional[str]
    discovery_error: Optional[str]
    fields: Mapping[str, FieldReport]
    verified: Optional[Verification]


@lru_cache(maxsize=128)
def _cleaned_overrides(config_id: str, encoded: str) -> Mapping[str, Any]:
    cleaned, problems = clean_overrides(json.loads(encoded))
    for problem in problems:
        logger.warning("[LayeredProfileResolver] ignoring a stored override on '%s': %s", config_id, problem)
    return MappingProxyType(cleaned)


def _typed(name: str, raw: Mapping[str, Any]) -> Any:
    return codec.field_from_json(CapabilityProfile, name, raw[name]) if name in raw else None


def _failure_message(exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return "Discovery timed out."
    if isinstance(exc, ModelListingError):
        return str(exc)
    return "Discovery failed unexpectedly."


class LayeredProfileResolver:
    def __init__(
        self,
        adapter_for: Callable[[LLMConfig], Adapter],
        seed: Optional[SeedTable] = None,
        *,
        now: Callable[[], datetime] = now_utc,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._adapter_for = adapter_for
        self._seed = seed if seed is not None else default_seed()
        self._now = now
        self._hits: TTLCache[DiscoveryKey, Discovery] = TTLCache(DISCOVERY_HIT_TTL_SECONDS, clock=clock)
        self._misses: TTLCache[DiscoveryKey, Discovery] = TTLCache(DISCOVERY_MISS_TTL_SECONDS, clock=clock)

    def resolve(self, config: LLMConfig) -> CapabilityProfile:
        adapter_name = self._adapter_for(config).name
        return self._profile(config, adapter_name, self._cached(config, adapter_name))

    async def refresh(self, config: LLMConfig) -> CapabilityProfile:
        adapter = self._adapter_for(config)
        key = self._key(config, adapter.name)
        previous = self._hits.get(key)
        fields, error = await self._discover(adapter, config)
        if error is not None:
            self._record_failure(key, previous, error)
        else:
            self._misses.invalidate(key)
            self._hits.set(key, Discovery(ProfileLayer(ProfileSource.DISCOVERED, fields), self._now()))
        return self.resolve(config)

    async def ensure(self, config: LLMConfig) -> CapabilityProfile:
        if self._cached(config, self._adapter_for(config).name) is None:
            return await self.refresh(config)
        return self.resolve(config)

    def explain(self, config: LLMConfig) -> ProfileReport:
        adapter_name = self._adapter_for(config).name
        discovery = self._cached(config, adapter_name)
        profile = self._profile(config, adapter_name, discovery)
        discovered = discovery.layer.fields if discovery is not None else {}
        overrides = self._overrides(config)
        seed = self._seed_layer(config, adapter_name, discovered, overrides).fields
        fields = {
            spec.name: FieldReport(
                value=getattr(profile, spec.name),
                source=profile.sources.get(spec.name, ProfileSource.SEED),
                seed=_typed(spec.name, seed),
                discovered=_typed(spec.name, discovered),
                override=_typed(spec.name, overrides),
            )
            for spec in PROFILE_FIELDS
        }
        return ProfileReport(
            config_id=config.id,
            adapter=adapter_name,
            model=config.model,
            discovered_at=dt_iso(discovery.at) if discovery is not None else None,
            discovery_error=discovery.error if discovery is not None else None,
            fields=fields,
            verified=profile.verified,
        )

    async def _discover(self, adapter: Adapter, config: LLMConfig) -> Tuple[Dict[str, Any], Optional[str]]:
        try:
            layer = await asyncio.wait_for(adapter.discover(config), timeout=DISCOVERY_TIMEOUT_SECONDS)
        except Exception as exc:
            logger.debug("[LayeredProfileResolver] discovery failed for '%s'", config.id, exc_info=True)
            return {}, _failure_message(exc)
        fields = dict(layer.fields)
        forbidden = sorted(NOT_DISCOVERABLE & set(fields))
        if forbidden:
            logger.warning("[LayeredProfileResolver] '%s' tried to discover %s", adapter.name, forbidden)
            return {}, f"The {adapter.name} adapter returned an invalid profile: discovery cannot set {forbidden}"
        try:
            CapabilityProfile(adapter=adapter.name, model=config.model).layered(
                [ProfileLayer(ProfileSource.DISCOVERED, fields)]
            )
        except (TypeError, ValueError) as exc:
            logger.warning("[LayeredProfileResolver] '%s' returned an invalid profile layer: %s", adapter.name, exc)
            return {}, f"The {adapter.name} adapter returned an invalid profile: {exc}"
        return fields, None

    def _profile(self, config: LLMConfig, adapter_name: str, discovery: Optional[Discovery]) -> CapabilityProfile:
        overrides = self._overrides(config)
        discovered = discovery.layer.fields if discovery is not None else {}
        layers = [self._seed_layer(config, adapter_name, discovered, overrides)]
        if discovered:
            layers.append(discovery.layer)
        if overrides:
            layers.append(ProfileLayer(ProfileSource.OVERRIDE, overrides))
        return CapabilityProfile(adapter=adapter_name, model=config.model).layered(layers)

    def _seed_layer(
        self, config: LLMConfig, adapter_name: str, discovered: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> ProfileLayer:
        server = overrides.get("server") or discovered.get("server")
        return self._seed.layer(SeedKey(adapter=adapter_name, model=config.model, server=server))

    @staticmethod
    def _overrides(config: LLMConfig) -> Mapping[str, Any]:
        stored = getattr(config, "profile_overrides", None)
        if not stored:
            return {}
        return _cleaned_overrides(config.id, json.dumps(stored, sort_keys=True))

    def _record_failure(self, key: DiscoveryKey, previous: Optional[Discovery], message: str) -> None:
        if previous is not None:
            self._hits.set(key, Discovery(previous.layer, previous.at, message))
            return
        self._misses.set(key, Discovery(ProfileLayer(ProfileSource.DISCOVERED, {}), self._now(), message))

    def _cached(self, config: LLMConfig, adapter_name: str) -> Optional[Discovery]:
        key = self._key(config, adapter_name)
        return self._hits.get(key) or self._misses.get(key)

    @staticmethod
    def _key(config: LLMConfig, adapter_name: str) -> DiscoveryKey:
        return (adapter_name, (config.base_url or "").strip().rstrip("/"), config.model)

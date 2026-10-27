import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, FrozenSet, List, Mapping, Optional, Pattern, Tuple

from src.features.llm.harness.profile import LAYERED_FIELDS, CapabilityProfile, ProfileLayer, ProfileSource
from src.platform.filesystem.yaml_io import load_yaml_file

SEED_PATH = Path(__file__).resolve().parents[3] / "std" / "llm" / "profiles.yml"
SEED_VERSION = 1

_TOP_LEVEL = frozenset({"version", "sources", "base", "rows", "catalog"})
_ROW_KEYS = frozenset({"id", "source", "adapters", "models", "servers", "fields"})
_CATALOG_KEYS = frozenset({"adapter", "model", "fields"})
_CACHE_SIZE = 512


class SeedError(ValueError):
    pass


@dataclass(frozen=True)
class SeedKey:
    adapter: str
    model: str
    server: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", self.model.strip().lower())
        if self.server is not None:
            object.__setattr__(self, "server", self.server.strip().lower() or None)


@dataclass(frozen=True)
class SeedRow:
    id: str
    sources: Tuple[str, ...]
    adapters: FrozenSet[str]
    models: Tuple[Pattern[str], ...]
    servers: FrozenSet[str]
    fields: Mapping[str, Any]

    def matches(self, key: SeedKey) -> bool:
        if self.adapters and key.adapter not in self.adapters:
            return False
        if self.servers and key.server not in self.servers:
            return False
        if self.models and not any(pattern.search(key.model) for pattern in self.models):
            return False
        return True


@dataclass(frozen=True)
class CatalogEntry:
    adapter: str
    model: str
    fields: Mapping[str, Any]


def _validated_fields(owner: str, fields: Any) -> Mapping[str, Any]:
    if not isinstance(fields, dict) or not fields:
        raise SeedError(f"{owner}: fields must be a non-empty mapping")
    if owner != "base" and "server" in fields:
        raise SeedError(f"{owner}: the server is what selects rows, so a row cannot set it")
    try:
        CapabilityProfile(adapter="seed", model="seed").layered([ProfileLayer(ProfileSource.SEED, fields)])
    except (TypeError, ValueError) as exc:
        raise SeedError(f"{owner}: {exc}") from exc
    return MappingProxyType(dict(fields))


def _strings(owner: str, name: str, value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise SeedError(f"{owner}: {name} must be a list of non-empty strings")
    return tuple(value)


def _unknown_keys(owner: str, data: Mapping[str, Any], allowed: FrozenSet[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SeedError(f"{owner}: unknown keys {unknown}")


def _row(data: Any, sources: Mapping[str, str]) -> SeedRow:
    if not isinstance(data, dict) or not isinstance(data.get("id"), str) or not data["id"]:
        raise SeedError("every row needs an id")
    owner = f"row {data['id']!r}"
    _unknown_keys(owner, data, _ROW_KEYS)
    cited = _strings(owner, "source", data.get("source"))
    if not cited:
        raise SeedError(f"{owner}: cite at least one source")
    missing = [name for name in cited if name not in sources]
    if missing:
        raise SeedError(f"{owner}: unknown sources {missing}")
    patterns = []
    for text in _strings(owner, "models", data.get("models")):
        try:
            patterns.append(re.compile(text, re.IGNORECASE))
        except re.error as exc:
            raise SeedError(f"{owner}: bad model pattern {text!r}: {exc}") from exc
    return SeedRow(
        id=data["id"],
        sources=cited,
        adapters=frozenset(_strings(owner, "adapters", data.get("adapters"))),
        models=tuple(patterns),
        servers=frozenset(item.lower() for item in _strings(owner, "servers", data.get("servers"))),
        fields=_validated_fields(owner, data.get("fields")),
    )


def _catalog_entry(data: Any) -> CatalogEntry:
    if not isinstance(data, dict):
        raise SeedError("every catalog entry must be a mapping")
    _unknown_keys("catalog entry", data, _CATALOG_KEYS)
    adapter, model = data.get("adapter"), data.get("model")
    if not isinstance(adapter, str) or not adapter or not isinstance(model, str) or not model:
        raise SeedError("every catalog entry needs an adapter and a model")
    return CatalogEntry(
        adapter=adapter,
        model=model.lower(),
        fields=_validated_fields(f"catalog {adapter}/{model}", data.get("fields")),
    )


class SeedTable:
    def __init__(
        self,
        sources: Mapping[str, str],
        base: Mapping[str, Any],
        rows: Tuple[SeedRow, ...],
        catalog: Tuple[CatalogEntry, ...],
    ) -> None:
        self.sources = MappingProxyType(dict(sources))
        self.base = base
        self.rows = rows
        self.catalog = catalog
        self._catalog_index = {(entry.adapter, entry.model): entry for entry in catalog}
        self._layers = lru_cache(maxsize=_CACHE_SIZE)(self._compute)

    @classmethod
    def from_mapping(cls, data: Any) -> "SeedTable":
        if not isinstance(data, dict):
            raise SeedError("the seed table must be a mapping")
        _unknown_keys("seed table", data, _TOP_LEVEL)
        if data.get("version") != SEED_VERSION:
            raise SeedError(f"unsupported seed version {data.get('version')!r}")
        sources = data.get("sources")
        if not isinstance(sources, dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in sources.items()):
            raise SeedError("sources must map a name to a url")
        base_section = data.get("base")
        base = _validated_fields("base", base_section.get("fields") if isinstance(base_section, dict) else None)
        missing = sorted(LAYERED_FIELDS - set(base))
        if missing:
            raise SeedError(f"base must set every profile field, missing {missing}")
        rows = tuple(_row(item, sources) for item in data.get("rows") or [])
        duplicate_ids = sorted(name for name, count in Counter(row.id for row in rows).items() if count > 1)
        if duplicate_ids:
            raise SeedError(f"duplicate row ids {duplicate_ids}")
        catalog = tuple(_catalog_entry(item) for item in data.get("catalog") or [])
        duplicate_models = sorted(
            pair for pair, count in Counter((entry.adapter, entry.model) for entry in catalog).items() if count > 1
        )
        if duplicate_models:
            raise SeedError(f"duplicate catalog entries {duplicate_models}")
        return cls(sources, base, rows, catalog)

    @classmethod
    def load(cls, path: Path = SEED_PATH) -> "SeedTable":
        return cls.from_mapping(load_yaml_file(path))

    def layer(self, key: SeedKey) -> ProfileLayer:
        return self._layers(key)

    def applied(self, key: SeedKey) -> Tuple[str, ...]:
        applied: List[str] = ["base"]
        if (key.adapter, key.model) in self._catalog_index:
            applied.append(f"catalog:{key.model}")
        applied.extend(row.id for row in self.rows if row.matches(key))
        return tuple(applied)

    def _compute(self, key: SeedKey) -> ProfileLayer:
        fields = dict(self.base)
        entry = self._catalog_index.get((key.adapter, key.model))
        if entry is not None:
            fields.update(entry.fields)
        for row in self.rows:
            if row.matches(key):
                fields.update(row.fields)
        return ProfileLayer(ProfileSource.SEED, MappingProxyType(fields))


@lru_cache(maxsize=1)
def default_seed() -> SeedTable:
    return SeedTable.load()

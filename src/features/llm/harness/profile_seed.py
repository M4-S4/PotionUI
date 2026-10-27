import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Pattern, Tuple

import yaml

from src.features.llm.harness.profile import LAYERED_FIELDS, CapabilityProfile, ProfileLayer, ProfileSource
from src.platform.filesystem.yaml_io import load_yaml_file

logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).resolve().parents[3] / "std" / "llm" / "profile_base.yml"
CONTENT_DIR = Path(__file__).resolve().parents[4] / "content" / "llm-profiles"
SEED_VERSION = 1
CORE_ROOT = "core"
MARKETPLACE_ROOT = "marketplace"
LOCAL_ROOT = "local"
PLUGIN_ROOT_PREFIX = "plugin:"

_FILE_KEYS = frozenset({"version", "sources", "models_dev", "rows", "catalog"})
_TABLE_KEYS = _FILE_KEYS | {"base"}
_ROW_KEYS = frozenset({"id", "source", "adapters", "models", "servers", "fields"})
_CATALOG_KEYS = frozenset({"adapter", "model", "fields"})
_CACHE_SIZE = 512


class SeedError(ValueError):
    pass


@dataclass(frozen=True)
class SeedOrigin:
    root: str
    file: str

    def __str__(self) -> str:
        return f"{self.root}/{self.file}"

_NO_ORIGIN = SeedOrigin("", "")

@dataclass(frozen=True)
class SeedRoot:
    name: str
    path: Path

@dataclass(frozen=True)
class AppliedSeed:
    id: str
    origin: SeedOrigin

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
    origin: SeedOrigin = _NO_ORIGIN

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
    origin: SeedOrigin = _NO_ORIGIN


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


def _row(data: Any, sources: Mapping[str, str], origin: SeedOrigin) -> SeedRow:
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
        origin=origin,
    )


def _catalog_entry(data: Any, origin: SeedOrigin) -> CatalogEntry:
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
        origin=origin,
    )




@dataclass(frozen=True)
class SeedFile:
    origin: SeedOrigin
    sources: Mapping[str, str]
    models_dev: Mapping[str, str]
    rows: Tuple[SeedRow, ...]
    catalog: Tuple[CatalogEntry, ...]
    problems: Tuple[str, ...] = ()


def _sources(data: Any) -> Mapping[str, str]:
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in data.items()):
        raise SeedError("sources must map a name to a url")
    return data


def _models_dev(data: Any) -> Mapping[str, str]:
    if data is None:
        return {}
    if not isinstance(data, dict) or not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in data.items()):
        raise SeedError("models_dev must map a models.dev provider to an adapter")
    return dict(data)


def _check_version(data: Mapping[str, Any]) -> None:
    if data.get("version") != SEED_VERSION:
        raise SeedError(f"unsupported seed version {data.get('version')!r}")


def parse_seed_file(data: Any, origin: SeedOrigin, *, strict: bool = False, allowed: FrozenSet[str] = _FILE_KEYS) -> SeedFile:
    if not isinstance(data, dict):
        raise SeedError("the seed table must be a mapping")
    _unknown_keys("seed table", data, allowed)
    _check_version(data)
    sources = _sources(data.get("sources"))
    models_dev = _models_dev(data.get("models_dev"))
    problems: List[str] = []
    rows: Dict[str, SeedRow] = {}
    for item in data.get("rows") or []:
        try:
            row = _row(item, sources, origin)
            if row.id in rows:
                raise SeedError(f"duplicate row ids {[row.id]}")
            rows[row.id] = row
        except SeedError as exc:
            if strict:
                raise
            problems.append(str(exc))
    catalog: Dict[Tuple[str, str], CatalogEntry] = {}
    for item in data.get("catalog") or []:
        try:
            entry = _catalog_entry(item, origin)
            if (entry.adapter, entry.model) in catalog:
                raise SeedError(f"duplicate catalog entries {[(entry.adapter, entry.model)]}")
            catalog[(entry.adapter, entry.model)] = entry
        except SeedError as exc:
            if strict:
                raise
            problems.append(str(exc))
    return SeedFile(origin, sources, models_dev, tuple(rows.values()), tuple(catalog.values()), tuple(problems))


def _base_fields(section: Any) -> Mapping[str, Any]:
    base = _validated_fields("base", section.get("fields") if isinstance(section, dict) else None)
    missing = sorted(LAYERED_FIELDS - set(base))
    if missing:
        raise SeedError(f"base must set every profile field, missing {missing}")
    return base


def _replacing(existing: List[Any], incoming: Any, same: Any) -> None:
    for index, current in enumerate(existing):
        if same(current, incoming):
            existing[index] = incoming
            return
    existing.append(incoming)


class SeedTable:
    def __init__(
        self,
        sources: Mapping[str, str],
        base: Mapping[str, Any],
        rows: Tuple[SeedRow, ...],
        catalog: Tuple[CatalogEntry, ...],
        problems: Tuple[str, ...] = (),
    ) -> None:
        self.sources = MappingProxyType(dict(sources))
        self.base = base
        self.rows = rows
        self.catalog = catalog
        self.problems = problems
        self._catalog_index = {(entry.adapter, entry.model): entry for entry in catalog}
        self._layers = lru_cache(maxsize=_CACHE_SIZE)(self._compute)

    @classmethod
    def assemble(cls, base: Mapping[str, Any], files: Iterable[SeedFile]) -> "SeedTable":
        sources: Dict[str, str] = {}
        rows: List[SeedRow] = []
        catalog: List[CatalogEntry] = []
        problems: List[str] = []
        for seed_file in files:
            sources.update(seed_file.sources)
            for row in seed_file.rows:
                _replacing(rows, row, lambda current, incoming: current.id == incoming.id)
            for entry in seed_file.catalog:
                _replacing(
                    catalog, entry, lambda current, incoming: (current.adapter, current.model) == (incoming.adapter, incoming.model)
                )
            problems.extend(seed_file.problems)
        ordered = [row for row in rows if not row.servers] + [row for row in rows if row.servers]
        return cls(sources, base, tuple(ordered), tuple(catalog), tuple(problems))

    @classmethod
    def from_mapping(cls, data: Any) -> "SeedTable":
        seed_file = parse_seed_file(data, _NO_ORIGIN, strict=True, allowed=_TABLE_KEYS)
        return cls.assemble(_base_fields(data.get("base")), [seed_file])

    @classmethod
    def load_base(cls, path: Path = BASE_PATH) -> Mapping[str, Any]:
        data = load_yaml_file(path)
        if not isinstance(data, dict):
            raise SeedError("the base table must be a mapping")
        _unknown_keys("base table", data, frozenset({"version", "base"}))
        _check_version(data)
        return _base_fields(data.get("base"))

    @classmethod
    def scan(cls, roots: Iterable[SeedRoot], base_path: Path = BASE_PATH) -> "SeedTable":
        files: List[SeedFile] = []
        for root in roots:
            if not root.path.is_dir():
                continue
            for path in sorted(root.path.glob("*.yml")):
                location = path.as_posix()
                try:
                    seed_file = parse_seed_file(load_yaml_file(path), SeedOrigin(root.name, path.name))
                except (SeedError, yaml.YAMLError, OSError, ValueError) as exc:
                    logger.warning("[ProfileSeed] skipping %s: %s", location, exc)
                    files.append(SeedFile(SeedOrigin(root.name, path.name), {}, {}, (), (), (f"{location}: {exc}",)))
                    continue
                for problem in seed_file.problems:
                    logger.warning("[ProfileSeed] rejected a row in %s: %s", location, problem)
                files.append(
                    SeedFile(
                        seed_file.origin,
                        seed_file.sources,
                        seed_file.models_dev,
                        seed_file.rows,
                        seed_file.catalog,
                        tuple(f"{location}: {problem}" for problem in seed_file.problems),
                    )
                )
        return cls.assemble(cls.load_base(base_path), files)

    def layer(self, key: SeedKey) -> ProfileLayer:
        return self._layers(key)

    def applied(self, key: SeedKey) -> Tuple[AppliedSeed, ...]:
        applied: List[AppliedSeed] = [AppliedSeed("base", SeedOrigin(CORE_ROOT, BASE_PATH.name))]
        entry = self._catalog_index.get((key.adapter, key.model))
        if entry is not None:
            applied.append(AppliedSeed(f"catalog:{key.model}", entry.origin))
        applied.extend(AppliedSeed(row.id, row.origin) for row in self.rows if row.matches(key))
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


def marketplace_root(content_dir: Path = CONTENT_DIR) -> SeedRoot:
    return SeedRoot(MARKETPLACE_ROOT, content_dir / MARKETPLACE_ROOT)


def local_root(content_dir: Path = CONTENT_DIR) -> SeedRoot:
    return SeedRoot(LOCAL_ROOT, content_dir / LOCAL_ROOT)


@lru_cache(maxsize=1)
def default_seed() -> SeedTable:
    return SeedTable.scan([marketplace_root()])

from __future__ import annotations

import threading
from typing import Optional

from src.platform.media_tools.params import PARAM_TYPES
from src.platform.media_tools.types import (
    CARDINALITIES,
    CORE_SOURCE,
    MEDIA_KINDS,
    ToolSpec,
)


def validate_spec(spec: ToolSpec) -> None:
    if not spec.id or not spec.label:
        raise ValueError("A media tool needs an id and a label")
    if spec.source == CORE_SOURCE:
        if "." in spec.id:
            raise ValueError(f"Core media tool id '{spec.id}' must not contain a dot")
    elif not spec.id.startswith(f"{spec.source}."):
        raise ValueError(f"Media tool id '{spec.id}' must start with '{spec.source}.'")
    if not spec.accepts or any(kind not in MEDIA_KINDS for kind in spec.accepts):
        raise ValueError(f"Media tool '{spec.id}' must accept one or more of {', '.join(MEDIA_KINDS)}")
    if spec.produces != "same" and spec.produces not in MEDIA_KINDS:
        raise ValueError(f"Media tool '{spec.id}' has an unknown 'produces' value '{spec.produces}'")
    if spec.cardinality not in CARDINALITIES:
        raise ValueError(f"Media tool '{spec.id}' has an unknown cardinality '{spec.cardinality}'")
    if spec.impl is None and not spec.component:
        raise ValueError(f"Media tool '{spec.id}' needs an implementation or a component")
    names: set[str] = set()
    for param in spec.params:
        name = param.get("name")
        if not name or name in names:
            raise ValueError(f"Media tool '{spec.id}' has a missing or duplicate parameter name")
        names.add(name)
        if param.get("type") not in PARAM_TYPES:
            raise ValueError(f"Media tool '{spec.id}' parameter '{name}' has an unsupported type '{param.get('type')}'")


class MediaToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._lock = threading.RLock()

    def register(self, spec: ToolSpec) -> None:
        validate_spec(spec)
        with self._lock:
            existing = self._tools.get(spec.id)
            if existing is not None and existing.source != spec.source:
                raise ValueError(f"Media tool '{spec.id}' is already registered by '{existing.source}'")
            self._tools[spec.id] = spec

    def unregister_source(self, source: str) -> int:
        with self._lock:
            doomed = [tool_id for tool_id, spec in self._tools.items() if spec.source == source]
            for tool_id in doomed:
                del self._tools[tool_id]
            return len(doomed)

    def get(self, tool_id: str) -> Optional[ToolSpec]:
        with self._lock:
            return self._tools.get(tool_id)

    def list(self, kind: Optional[str] = None, scope: Optional[str] = None) -> list[ToolSpec]:
        with self._lock:
            specs = list(self._tools.values())
        if kind:
            specs = [spec for spec in specs if kind in spec.accepts]
        if scope:
            specs = [spec for spec in specs if scope in spec.scopes]
        return sorted(specs, key=lambda spec: spec.source != CORE_SOURCE)

    def sources(self) -> set[str]:
        with self._lock:
            return {spec.source for spec in self._tools.values()}

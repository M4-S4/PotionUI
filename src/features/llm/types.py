from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.platform.plugins.llm_providers import LLMProviderRegistry


@dataclass(frozen=True)
class LLMType:
    type: str
    adapter: Any
    label: Optional[str] = None
    default_base_url: Optional[str] = None
    source: str = "core"


class LLMTypes:
    def __init__(self, core: Sequence[LLMType], provider_registry: Optional[LLMProviderRegistry] = None) -> None:
        self._core: Dict[str, LLMType] = {entry.type: entry for entry in core}
        self.provider_registry = provider_registry if provider_registry is not None else LLMProviderRegistry()

    def get(self, type_id: str) -> Optional[LLMType]:
        if type_id in self._core:
            return self._core[type_id]
        registration = self.provider_registry.get(type_id)
        if registration is None:
            return None
        return LLMType(
            type=registration.type_name,
            adapter=registration.adapter,
            label=registration.label,
            default_base_url=registration.default_base_url or None,
            source=registration.source,
        )

    def adapter_for(self, config: Any) -> Any:
        return self.require(config.type).adapter

    def require(self, type_id: str) -> LLMType:
        entry = self.get(type_id)
        if entry is None:
            raise ValueError(f"Unsupported LLM type: {type_id}")
        return entry

    def all(self) -> List[LLMType]:
        plugin_entries = [
            self.get(registration.type_name)
            for registration in self.provider_registry.all()
            if registration.type_name not in self._core
        ]
        return [*self._core.values(), *(entry for entry in plugin_entries if entry is not None)]

    def describe(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": entry.type,
                "label": entry.label,
                "adapter": getattr(entry.adapter, "name", entry.type),
                "default_base_url": entry.default_base_url,
                "supports_model_listing": callable(getattr(entry.adapter, "list_models", None)),
                "source": entry.source,
            }
            for entry in self.all()
        ]

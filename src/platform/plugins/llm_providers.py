from dataclasses import dataclass
from typing import Any

from src.platform.plugins.sourced_registry import SourcedRegistry


@dataclass(frozen=True)
class LLMProviderRegistration:
    type_name: str
    label: str
    client: Any
    default_base_url: str = ""
    source: str = "core"


class LLMProviderRegistry(SourcedRegistry[LLMProviderRegistration]):
    key_attr = "type_name"
    label = "LLM provider"


llm_provider_registry = LLMProviderRegistry()

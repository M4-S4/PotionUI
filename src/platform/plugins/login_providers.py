from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.platform.plugins.sourced_registry import DuplicateRegistrationError, SourcedRegistry


PLUGIN_ROUTE_PREFIX = "/api/plugins/"


class DuplicateLoginProviderError(DuplicateRegistrationError):
    pass


class InvalidLoginProviderError(ValueError):
    pass


def source_from_start_path(start_path: str) -> Optional[str]:
    if not start_path.startswith(PLUGIN_ROUTE_PREFIX):
        return None
    remainder = start_path[len(PLUGIN_ROUTE_PREFIX):]
    plugin_id = remainder.split("/", 1)[0]
    return plugin_id or None


@dataclass(frozen=True)
class LoginProviderDefinition:
    id: str
    label: str
    start_path: str
    source: str = "core"

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label, "start_path": self.start_path}


class LoginProviderRegistry(SourcedRegistry[LoginProviderDefinition]):
    key_attr = "id"
    label = "Login provider"
    duplicate_error = DuplicateLoginProviderError

    def register(self, definition: LoginProviderDefinition) -> None:
        if not definition.id:
            raise InvalidLoginProviderError("Login provider id is required")
        if not definition.label:
            raise InvalidLoginProviderError(
                f"Login provider '{definition.id}' is missing a label"
            )
        if not definition.start_path.startswith("/"):
            raise InvalidLoginProviderError(
                f"Login provider '{definition.id}' needs an absolute start_path"
            )
        super().register(definition)

    def unregister(self, provider_id: str) -> None:
        self._entries.pop(provider_id, None)

    def manifest(self) -> List[Dict[str, Any]]:
        return [defn.to_dict() for defn in self.all()]


login_provider_registry = LoginProviderRegistry()

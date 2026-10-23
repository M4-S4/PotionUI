"""
Prompt importer registry.

This module provides the single declaration point for a prompt-library import
source: its id, the label shown in the Import menu, the plugin frontend
component that renders its modal, and the backend object that runs the
import. There are no builtin importers - the registry starts empty, and the
frontend's Import button is only shown once a plugin has registered one.

Plugins extend the import menu by registering additional
`PromptImporterDefinition`s on the shared `prompt_importer_registry` singleton
when they are enabled, and by having them removed via `unregister_source` when
disabled.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from src.platform.plugins.sourced_registry import DuplicateRegistrationError, SourcedRegistry


class DuplicatePromptImporterError(DuplicateRegistrationError):
    """Raised when registering an importer id that is already registered."""


@dataclass(frozen=True)
class PromptImporterDefinition:
    """Declaration for a single prompt importer."""

    importer_id: str
    label: str
    frontend_component: str
    # A `src.plugin_api.prompts.PromptImporter` instance - typed `Any` here
    # because `src.platform` must not import `src.plugin_api`/`src.features`.
    backend: Any
    source: str = "core"


class PromptImporterRegistry(SourcedRegistry[PromptImporterDefinition]):
    """Registry mapping importer id -> `PromptImporterDefinition`."""

    key_attr = "importer_id"
    label = "Prompt importer"
    duplicate_error = DuplicatePromptImporterError

    def frontend_manifest(self) -> List[Dict[str, Any]]:
        """Serialize the registry for the `GET /api/prompts/importers` endpoint."""
        return [
            {"id": defn.importer_id, "label": defn.label, "component": defn.frontend_component}
            for defn in self.all()
        ]


# Module-level singleton shared by the plugin enable/disable path and the
# prompt import routes - mirrors `field_type_registry`.
prompt_importer_registry = PromptImporterRegistry()

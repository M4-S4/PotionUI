"""Preset requirement checker registry.

A preset declares what it needs to run (`requirements:` in `preset.yml` - see
docs/presets.md "Requirements"): a binary on PATH, a Python package, a
specific model present in the depot, a VRAM floor, a supported OS. A
"requirement checker" is the code that evaluates one `type:` of entry against
this instance. Core ships five (`src.features.presets.requirements.builtin`);
a plugin can add engine-specific ones (e.g. a ComfyUI custom-node check)
without touching core, by declaring `requirement_checkers:` in its
`manifest.yml`.

This module holds only the registration bookkeeping - the checker contract
itself (`RequirementChecker`, `RequirementResult`, `RequirementContext`) lives
in `src.features.presets.requirements.contracts` (re-exported by
`src.plugin_api.presets`), because it carries feature-shaped context (the
models catalog, the resolved backend) that `src.platform` must not import.
Checker objects are stored here `Any`-typed for the same reason - mirrors
`FieldTypeRegistry`/`PromptImporterRegistry`.

Plugins extend the system by registering additional checkers on the shared
`requirement_checker_registry` singleton when they are enabled, and by having
them removed via `unregister_source` when disabled.
"""

from dataclasses import dataclass
from typing import Any

from src.platform.plugins.sourced_registry import DuplicateRegistrationError, SourcedRegistry


class DuplicateRequirementCheckerError(DuplicateRegistrationError):
    """Raised when registering a requirement type name that is already registered."""


@dataclass(frozen=True)
class RequirementCheckerRegistration:
    """One registered requirement checker."""

    type_name: str
    # A `src.plugin_api.presets.RequirementChecker` instance - typed `Any`
    # here because `src.platform` must not import `src.features`/`src.plugin_api`.
    checker: Any
    source: str = "core"


class RequirementCheckerRegistry(SourcedRegistry[RequirementCheckerRegistration]):
    """Registry mapping requirement `type:` name -> `RequirementCheckerRegistration`."""

    key_attr = "type_name"
    label = "Requirement checker"
    duplicate_error = DuplicateRequirementCheckerError



# Module-level singleton shared by the plugin enable/disable path and preset
# requirements evaluation - mirrors `field_type_registry`.
requirement_checker_registry = RequirementCheckerRegistry()

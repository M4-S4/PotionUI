"""Keeping data.

`db` is the application's database. A plugin may create and own its own tables -
give them a name that starts with the plugin id so it is obvious whose they are -
and `generate_ulid()` mints sortable primary keys for them.

`PluginRepository` reads the settings an admin filled in against your manifest's
`settings:` section, including your credentials; `SettingRepository` and
`Settings` read application-wide settings. Do not read another plugin's
settings.
"""

from typing import Any, Dict, Optional

from src.features.plugins.manifest_settings import resolve_plugin_settings
from src.features.plugins.repository import PluginRepository
from src.platform.plugins.registry import PluginRegistry
from src.platform.settings.repository import SettingRepository
from src.platform.settings.settings import Settings
from src.platform.util.ids import generate_ulid

__all__ = [
    "PluginRepository",
    "SettingRepository",
    "Settings",
    "db",
    "generate_ulid",
    "plugin_settings",
]


def plugin_settings(
    plugin_id: str,
    repository: Optional[PluginRepository] = None,
    registry: Optional[PluginRegistry] = None,
) -> Dict[str, Any]:
    if registry is None:
        from src.platform.plugins.runtime_registries import get_global_plugin_registry
        registry = get_global_plugin_registry()
    return resolve_plugin_settings(repository or PluginRepository(), registry, plugin_id)



def __getattr__(name):
    """`db` is resolved on access, not bound here at import time - a plugin
    importing this module before a test patches the process-default
    `Database` singleton would otherwise keep the pre-patch reference for the
    rest of the process."""
    if name == "db":
        from src.platform.database.database import db
        return db
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

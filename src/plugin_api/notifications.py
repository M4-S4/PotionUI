"""Notifying users from a plugin.

Declare the notification types once, from a `notification_type.register` hook
handler (the registry arrives in `context.data["registry"]`), so each shows up
in every user's preferences and can be turned off; `notify(...)` then raises
one, persisted, pushed to the user's open clients and shown as a toast unless
the user disabled that type. A notification without a registered `type` is
always delivered.
"""

from typing import Any, Dict, Iterable, List, Optional

from src.features.notifications.types import (
    DuplicateNotificationTypeError,
    NotificationTypeRegistry,
    NotificationTypeSpec,
)
from src.platform.plugins.runtime_registries import get_container

__all__ = [
    "DuplicateNotificationTypeError",
    "NotificationTypeRegistry",
    "NotificationTypeSpec",
    "notify",
    "register_notification_types",
]


def register_notification_types(registry: NotificationTypeRegistry, specs: Iterable[NotificationTypeSpec]) -> List[str]:
    """Register the types not registered yet; returns the keys it added."""
    added: List[str] = []
    for spec in specs:
        if registry.has(spec.key):
            continue
        try:
            registry.register(spec)
        except DuplicateNotificationTypeError:
            continue
        added.append(spec.key)
    return added


def notify(
    *,
    source: str,
    title: str,
    message: str = "",
    user_id: Optional[str] = None,
    type: str = "",
    level: str = "info",
    category: str = "plugin",
    metadata: Optional[Dict[str, Any]] = None,
    transient: bool = False,
    show_toast: bool = True,
) -> int:
    """Raise a notification; returns how many users received it. `source` is the
    plugin id; `user_id` None broadcasts; `type` is one of the registered keys;
    `level` is "info", "success", "warning" or "error"; `transient` skips
    persistence."""
    rows = get_container().notification_manager(
        level=level,
        title=title,
        message=message,
        category=category,
        user_id=user_id,
        source=source,
        transient=transient,
        show_toast=show_toast,
        metadata=dict(metadata) if metadata else None,
        type=type,
    )
    return len(rows or [])

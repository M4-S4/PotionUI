"""Tests for the `src.plugin_api.notifications` surface."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.features.notifications.types import NotificationTypeRegistry, NotificationTypeSpec
from src.plugin_api.notifications import notify, register_notification_types


def test_notify_raises_the_notification_through_the_application_and_counts_recipients():
    manager = Mock(return_value=[object(), object()])
    container = SimpleNamespace(notification_manager=manager)

    with patch("src.plugin_api.notifications.get_container", return_value=container):
        delivered = notify(
            source="my-plugin", title="Done", message="m.safetensors is in", user_id="user-1",
            type="my-plugin.done", level="success", metadata={"model_id": "m1"},
        )

    assert delivered == 2
    manager.assert_called_once_with(
        level="success", title="Done", message="m.safetensors is in", category="plugin", user_id="user-1",
        source="my-plugin", transient=False, show_toast=True, metadata={"model_id": "m1"}, type="my-plugin.done",
    )


def test_notify_filtered_out_for_every_target_counts_zero():
    container = SimpleNamespace(notification_manager=Mock(return_value=[]))

    with patch("src.plugin_api.notifications.get_container", return_value=container):
        assert notify(source="my-plugin", title="Quiet") == 0


def test_register_notification_types_adds_the_missing_ones_only():
    registry = NotificationTypeRegistry()
    registry.register(NotificationTypeSpec(key="my-plugin.old", label="Old", category="plugin"))

    added = register_notification_types(registry, [
        NotificationTypeSpec(key="my-plugin.old", label="Old again", category="plugin"),
        NotificationTypeSpec(key="my-plugin.new", label="New", category="plugin"),
    ])

    assert added == ["my-plugin.new"]
    assert registry.get("my-plugin.old").label == "Old"
    assert registry.has("my-plugin.new")

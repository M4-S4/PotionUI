"""Tests for `limit_status` in the `src.plugin_api.limits` surface."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.plugin_api.limits import LimitStatus, limit_status


def _guard(resolution, *, exempt=False, measured=None):
    settings = SimpleNamespace(exempt_admins=True, day_timezone="Europe/Paris", contact_line="Ask the admin")
    return Mock(
        resolution=Mock(return_value=resolution),
        settings=Mock(read=Mock(return_value=settings)),
        is_exempt=Mock(return_value=exempt),
        measure=Mock(return_value=measured),
    )


def _resolution(limit):
    return SimpleNamespace(subject=SimpleNamespace(user_id="user-1"), value=Mock(return_value=limit))


def _container(guard, kind=None):
    return SimpleNamespace(
        plans=SimpleNamespace(guard=guard),
        limit_kind_registry=Mock(get=Mock(return_value=kind)),
    )


def test_user_without_a_plan_is_unlimited():
    guard = _guard(None)

    with patch("src.plugin_api.limits.get_container", return_value=_container(guard)):
        status = limit_status("user-1", "my-plugin.downloads")

    assert status == LimitStatus(kind="my-plugin.downloads")
    assert not status.limited
    assert status.remaining is None
    guard.measure.assert_not_called()


def test_plan_without_a_limit_for_the_kind_is_unlimited():
    guard = _guard(_resolution(None))

    with patch("src.plugin_api.limits.get_container", return_value=_container(guard)):
        assert limit_status("user-1", "my-plugin.downloads").limit is None


def test_exempt_admin_keeps_the_limit_but_is_not_measured():
    guard = _guard(_resolution(5.0), exempt=True)

    with patch("src.plugin_api.limits.get_container", return_value=_container(guard)):
        status = limit_status("admin-1", "my-plugin.downloads")

    assert status.limit == 5.0
    assert status.exempt is True
    assert status.used is None
    assert status.contact_line == "Ask the admin"
    assert not status.limited
    guard.measure.assert_not_called()


def test_limited_user_is_measured_in_the_current_window():
    resets_at = datetime(2026, 10, 13, tzinfo=timezone.utc)
    guard = _guard(_resolution(5.0), measured=SimpleNamespace(used=3.0, window_start=None, resets_at=resets_at))
    kind = SimpleNamespace(key="my-plugin.downloads")

    with patch("src.plugin_api.limits.get_container", return_value=_container(guard, kind)):
        status = limit_status("user-1", "my-plugin.downloads")

    assert status == LimitStatus(
        kind="my-plugin.downloads", limit=5.0, used=3.0, exempt=False, resets_at=resets_at, contact_line="Ask the admin",
    )
    assert status.limited
    assert status.remaining == 2.0
    guard.measure.assert_called_once_with(kind, "user-1", "Europe/Paris")


def test_unregistered_kind_is_unlimited():
    guard = _guard(_resolution(5.0))

    with patch("src.plugin_api.limits.get_container", return_value=_container(guard, kind=None)):
        assert limit_status("user-1", "my-plugin.gone").limit is None

    guard.measure.assert_not_called()


def test_remaining_never_goes_negative():
    assert LimitStatus(kind="k", limit=2.0, used=9.0).remaining == 0.0

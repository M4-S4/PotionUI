from dataclasses import replace
from unittest.mock import patch

import pytest

import backend.api as api


@pytest.fixture
def registered(make_settings):
    captured = {}

    def fake_register(provider_id, label, start_path, auto_redirect=None):
        captured["check"] = auto_redirect

    with patch.object(api, "register_login_provider", fake_register), patch.object(
        api, "load_settings", lambda: make_settings()
    ):
        api.enable()
    return captured


def _advertised(registered):
    return {"auto_redirect": bool(registered["check"]())}


def test_auto_redirect_is_off_by_default(make_settings, registered):
    with patch.object(api, "load_settings", lambda: make_settings()):
        assert _advertised(registered)["auto_redirect"] is False


def test_auto_redirect_follows_the_setting_without_re_registering(make_settings, registered):
    current = {"settings": make_settings()}
    with patch.object(api, "load_settings", lambda: current["settings"]):
        assert _advertised(registered)["auto_redirect"] is False
        current["settings"] = replace(make_settings(), auto_redirect=True)
        assert _advertised(registered)["auto_redirect"] is True
        current["settings"] = replace(make_settings(), auto_redirect=False)
        assert _advertised(registered)["auto_redirect"] is False


def test_auto_redirect_is_never_advertised_while_unconfigured(make_settings, registered):
    unconfigured = replace(make_settings(client_secret=""), auto_redirect=True)
    with patch.object(api, "load_settings", lambda: unconfigured):
        assert _advertised(registered)["auto_redirect"] is False


def test_manifest_declares_a_boolean_off_by_default():
    from pathlib import Path

    import yaml

    manifest = yaml.safe_load(
        (Path(api.__file__).resolve().parents[1] / "manifest.yml").read_text(encoding="utf-8")
    )
    spec = next(entry for entry in manifest["settings"] if entry["name"] == "auto_redirect")
    assert spec["type"] == "boolean"
    assert spec["default"] is False

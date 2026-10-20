from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.settings import (
    CORE_USER_NSFW_SETTING,
    get_setting,
    user_nsfw_mode,
)

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "https://civitai.com/api/v1"


class _StoredRows:
    rows = {}

    def get_plugin_settings(self, plugin_id):
        assert plugin_id == "model-browser"
        return [
            SimpleNamespace(setting_key=key, setting_value=value)
            for key, value in self.rows.items()
        ]


@pytest.fixture
def stored(monkeypatch):
    rows = {}
    monkeypatch.setattr(_StoredRows, "rows", rows)
    monkeypatch.setattr("src.plugin_api.storage.PluginRepository", _StoredRows)
    return rows


def test_every_manifest_setting_reads_as_its_default_when_unset(stored):
    manifest = yaml.safe_load((PLUGIN_ROOT / "manifest.yml").read_text(encoding="utf-8"))
    keys = [spec["name"] for spec in manifest["settings"] if spec["type"] != "info"]

    assert sorted(keys) == sorted(
        ["nsfw_ceiling", "nsfw_max_level", "hide_poi", "civitai_api_base", "cache_ttl_minutes"]
    )
    assert {key: get_setting(key) for key in keys} == {
        "nsfw_ceiling": "inherit",
        "nsfw_max_level": 2,
        "hide_poi": False,
        "civitai_api_base": DEFAULT_BASE,
        "cache_ttl_minutes": 10,
    }


@pytest.mark.parametrize("raw", ["", "   "])
@pytest.mark.parametrize("key", ["nsfw_ceiling", "nsfw_max_level", "hide_poi", "civitai_api_base", "cache_ttl_minutes"])
def test_a_blank_setting_is_its_default(stored, key, raw):
    stored[key] = raw

    assert get_setting(key) == {
        "nsfw_ceiling": "inherit",
        "nsfw_max_level": 2,
        "hide_poi": False,
        "civitai_api_base": DEFAULT_BASE,
        "cache_ttl_minutes": 10,
    }[key]


@pytest.mark.parametrize(
    "key, raw, value",
    [
        ("nsfw_ceiling", "inherit", "inherit"),
        ("nsfw_ceiling", "blur", "blur"),
        ("nsfw_ceiling", "blocked", "blocked"),
        ("nsfw_ceiling", " Blocked ", "blocked"),
        ("nsfw_ceiling", "allowed", "inherit"),
        ("nsfw_ceiling", "hide", "inherit"),
        ("nsfw_max_level", "1", 1),
        ("nsfw_max_level", "4", 4),
        ("nsfw_max_level", " 8 ", 8),
        ("nsfw_max_level", "3", 2),
        ("nsfw_max_level", "16", 2),
        ("nsfw_max_level", "0", 2),
        ("nsfw_max_level", "-1", 2),
        ("nsfw_max_level", "R", 2),
        ("nsfw_max_level", "4.0", 2),
        ("hide_poi", "True", True),
        ("hide_poi", "true", True),
        ("hide_poi", "1", True),
        ("hide_poi", "yes", True),
        ("hide_poi", " on ", True),
        ("hide_poi", "False", False),
        ("hide_poi", "0", False),
        ("hide_poi", "off", False),
        ("hide_poi", "maybe", False),
        ("civitai_api_base", "https://civitai.red/api/v1", "https://civitai.red/api/v1"),
        ("civitai_api_base", "https://civitai.red/api/v1/", "https://civitai.red/api/v1"),
        ("civitai_api_base", "  https://civitai.com/api/v1//  ", "https://civitai.com/api/v1"),
        ("civitai_api_base", "http://civitai.com/api/v1", DEFAULT_BASE),
        ("civitai_api_base", "civitai.com/api/v1", DEFAULT_BASE),
        ("civitai_api_base", "ftp://civitai.com/api/v1", DEFAULT_BASE),
        ("civitai_api_base", "https://", DEFAULT_BASE),
        ("civitai_api_base", "https:///api/v1", DEFAULT_BASE),
        ("civitai_api_base", "https://[::1/api/v1", DEFAULT_BASE),
        ("cache_ttl_minutes", "30", 30),
        ("cache_ttl_minutes", "1", 1),
        ("cache_ttl_minutes", "1440", 1440),
        ("cache_ttl_minutes", "0", 1),
        ("cache_ttl_minutes", "-5", 1),
        ("cache_ttl_minutes", "5000", 1440),
        ("cache_ttl_minutes", "ten", 10),
        ("cache_ttl_minutes", "7.5", 10),
    ],
)
def test_get_setting(stored, key, raw, value):
    stored[key] = raw

    assert get_setting(key) == value


def test_get_setting_refuses_a_key_the_plugin_does_not_have(stored):
    with pytest.raises(KeyError):
        get_setting("nsfw_filter")


class _Repository:
    def __init__(self, answer):
        self.answer = answer
        self.asked = []

    def __call__(self):
        return self

    def get_user_setting_by_key(self, user_id, key):
        self.asked.append((user_id, key))
        return self.answer(key)

    def get_plugin_setting(self, plugin_id, key):
        self.asked.append((plugin_id, key))
        return self.answer(key)


@pytest.mark.parametrize(
    "stored, mode",
    [
        ("show", "show"),
        ("blur", "blur"),
        ("hide", "hide"),
        (" SHOW ", "show"),
        (None, "blur"),
        ("", "blur"),
        ("explicit", "blur"),
    ],
)
def test_user_nsfw_mode_reads_the_cores_user_setting(monkeypatch, stored, mode):
    repository = _Repository(lambda key: stored)
    monkeypatch.setattr("src.plugin_api.SettingRepository", repository)

    assert user_nsfw_mode("user-1") == mode
    assert repository.asked == [("user-1", CORE_USER_NSFW_SETTING)]
    assert CORE_USER_NSFW_SETTING == "media_nsfw_filter_mode"

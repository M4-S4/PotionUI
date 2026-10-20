import logging
from types import SimpleNamespace

import pytest

from src.features.plugins.manifest_settings import (
    resolve_plugin_settings,
    secret_setting_keys,
    settings_json_schema,
)
from src.platform.plugins.manifest import SettingSpec

SETTINGS = [
    {"name": "token", "type": "string", "is_secret": True},
    {"name": "host", "type": "string", "default": "localhost"},
    {"name": "cuda", "type": "string", "default": "13.0", "allow_empty": True},
    {"name": "port", "type": "number", "default": 8188, "min": 1, "max": 65535},
    {"name": "delay", "type": "number", "default": 1.5, "min": 0.5, "max": 10.0},
    {"name": "secure", "type": "boolean", "default": False},
    {
        "name": "level",
        "type": "select",
        "default": "2",
        "options": [{"value": "1", "label": "a"}, {"value": "2", "label": "b"}],
    },
    {"name": "notice", "type": "info"},
]


class Repo:
    def __init__(self, **stored):
        self.stored = stored

    def get_plugin_settings(self, plugin_id):
        assert plugin_id == "demo"
        return [
            SimpleNamespace(setting_key=key, setting_value=value)
            for key, value in self.stored.items()
        ]


class Registry:
    def __init__(self, settings=SETTINGS):
        self.manifest = SimpleNamespace(settings=settings)

    def get_plugin(self, plugin_id):
        return self.manifest if plugin_id == "demo" else None


def resolve(**stored):
    return resolve_plugin_settings(Repo(**stored), Registry(), "demo")


def test_nothing_stored_yields_every_manifest_default():
    assert resolve() == {
        "token": None,
        "host": "localhost",
        "cuda": "13.0",
        "port": 8188,
        "delay": 1.5,
        "secure": False,
        "level": "2",
    }


def test_stored_values_are_coerced_by_declared_type():
    values = resolve(token="sk-1", host="example.org", port=" 9000 ", delay="2", secure=" Yes ", level=" 1 ")

    assert values["token"] == "sk-1"
    assert values["host"] == "example.org"
    assert values["port"] == 9000 and isinstance(values["port"], int)
    assert values["delay"] == 2.0 and isinstance(values["delay"], float)
    assert values["secure"] is True
    assert values["level"] == "1"


def test_a_string_keeps_its_empty_value_only_without_a_default_or_when_allowed():
    values = resolve(token="", host="", cuda="")

    assert values["token"] == ""
    assert values["host"] == "localhost"
    assert values["cuda"] == ""


@pytest.mark.parametrize("raw", ["ten", "7.5", "", "  "])
def test_an_unparsable_number_falls_back_to_the_default(raw):
    assert resolve(port=raw)["port"] == 8188


def test_a_bad_select_and_boolean_garbage_fall_back():
    values = resolve(level="9", secure="maybe")

    assert values["level"] == "2"
    assert values["secure"] is False


@pytest.mark.parametrize(
    "key, raw, expected",
    [("port", "0", 1), ("port", "70000", 65535), ("delay", "0.1", 0.5), ("delay", "99", 10.0)],
)
def test_out_of_range_numbers_are_clamped_and_logged(caplog, key, raw, expected):
    with caplog.at_level(logging.WARNING):
        value = resolve(**{key: raw})[key]

    assert value == expected
    assert type(value) is type(resolve()[key])
    assert any("demo" in record.message and key in record.message for record in caplog.records)


def test_an_in_range_number_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        resolve(port="9000")

    assert caplog.records == []


def test_rows_the_manifest_does_not_declare_are_ignored():
    assert "worker_token:p1" not in resolve(**{"worker_token:p1": "x"})


def test_an_unknown_plugin_has_no_settings():
    assert resolve_plugin_settings(Repo(), Registry(), "other") == {}


def test_secret_keys_come_from_the_manifest_in_order():
    assert secret_setting_keys(Registry(), "demo") == ["token"]
    assert secret_setting_keys(Registry(), "other") is None
    typed = Registry([SettingSpec(name="a", type="string", is_secret=True)])
    assert secret_setting_keys(typed, "demo") == ["a"]


def test_the_json_schema_is_derived_from_the_manifest():
    schema = settings_json_schema(Registry(), "demo")

    assert set(schema["properties"]) == {"token", "host", "cuda", "port", "delay", "secure", "level"}
    assert schema["properties"]["token"]["format"] == "password"
    assert schema["properties"]["port"] == {"type": "number", "default": 8188, "minimum": 1, "maximum": 65535}
    assert schema["properties"]["level"]["enum"] == ["1", "2"]
    assert settings_json_schema(Registry(), "other") is None

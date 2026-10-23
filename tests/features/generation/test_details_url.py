from unittest.mock import Mock

from src.features.generation.details_url import generation_details_url, resolve_public_base_url


def _settings(value):
    settings = Mock()
    settings.get_setting.return_value = value
    return settings


def test_configured_base_wins_over_the_request_origin():
    url = generation_details_url("gen-1", _settings("https://potion.example.com/"), "http://127.0.0.1:26730")

    assert url == "https://potion.example.com/history?generation=gen-1"


def test_request_origin_is_used_when_nothing_is_configured():
    assert generation_details_url("gen-1", _settings(""), "http://192.168.1.5:26730") == (
        "http://192.168.1.5:26730/history?generation=gen-1"
    )


def test_relative_path_when_no_base_is_known():
    assert generation_details_url("gen-1", None, None) == "/history?generation=gen-1"


def test_id_is_percent_encoded():
    assert generation_details_url("a b/c", None, None) == "/history?generation=a%20b%2Fc"


def test_failing_settings_lookup_falls_back_to_origin():
    settings = Mock()
    settings.get_setting.side_effect = RuntimeError("db down")

    assert resolve_public_base_url(settings, "https://x.test") == "https://x.test"


def test_whitespace_only_base_falls_back_to_the_request_origin():
    assert generation_details_url("gen-1", _settings("   "), "https://x.test") == (
        "https://x.test/history?generation=gen-1"
    )

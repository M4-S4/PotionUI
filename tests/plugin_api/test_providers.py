import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import aiohttp
import pytest

from src.plugin_api import HookContext
from src.plugin_api.providers import (
    HttpMarketplaceProvider,
    ProviderConnectionError,
    ProviderMetadata,
    ProviderRateLimitError,
    register_provider,
    retry_after_from_headers,
)


def test_seconds_value_is_returned_as_a_float():
    assert retry_after_from_headers({"Retry-After": "12"}) == 12.0


def test_http_date_value_is_converted_to_seconds_from_now():
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=90), usegmt=True)

    assert 60 < retry_after_from_headers({"Retry-After": when}) <= 90


def test_missing_or_unreadable_value_falls_back_to_the_default():
    assert retry_after_from_headers({}) == 60.0
    assert retry_after_from_headers({"Retry-After": "soon"}) == 60.0
    assert retry_after_from_headers({"Retry-After": "soon"}, default=5.0) == 5.0


class _Answer:
    def __init__(self, status=200, data=None, headers=None, text="body"):
        self.status = status
        self.headers = headers or {}
        self._data = data
        self._text = text

    async def json(self):
        return self._data

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _Session:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class _Provider(HttpMarketplaceProvider):
    BASE_URL = "https://example.test/api"
    DOWNLOAD_HOST = "example.test"
    DEFAULT_RATE_LIMIT_DELAY = 0.0

    def get_metadata(self):
        return ProviderMetadata(id="example", name="Example", description="", website="https://example.test")

    async def get_model_by_hash(self, sha256):
        return None


def _provider(outcome):
    provider = _Provider()
    provider._initialized = True
    provider._session = _Session(outcome)
    provider._session.closed = False
    return provider


async def test_get_json_returns_the_payload_of_a_200():
    provider = _provider(_Answer(data={"ok": True}))

    assert await provider._get_json("https://example.test/x", what="fetching x") == (200, {"ok": True})


async def test_get_json_passes_only_the_options_it_was_given():
    provider = _provider(_Answer(data={}))

    await provider._get_json("https://example.test/x", what="fetching x")
    await provider._get_json("https://example.test/x", what="fetching x", params={"a": 1}, headers={"h": "v"})

    bare, full = provider._session.calls
    assert set(bare[1]) == {"timeout"}
    assert set(full[1]) == {"timeout", "params", "headers"}


async def test_get_json_returns_the_status_and_body_text_of_other_answers():
    provider = _provider(_Answer(status=503, text="down"))

    assert await provider._get_json("https://example.test/x", what="fetching x") == (503, "down")


async def test_get_json_maps_429_to_a_rate_limit_error_with_the_header_seconds():
    provider = _provider(_Answer(status=429, headers={"Retry-After": "12"}))

    with pytest.raises(ProviderRateLimitError) as caught:
        await provider._get_json("https://example.test/x", what="fetching x")

    assert caught.value.retry_after == 12.0
    assert "Example" in str(caught.value)


async def test_get_json_maps_a_timeout_and_a_client_error_to_connection_errors():
    with pytest.raises(ProviderConnectionError, match="Timeout fetching x"):
        await _provider(asyncio.TimeoutError())._get_json("https://example.test/x", what="fetching x")
    with pytest.raises(ProviderConnectionError, match="Connection error: boom"):
        await _provider(aiohttp.ClientError("boom"))._get_json("https://example.test/x", what="fetching x")


async def test_initialize_reads_settings_and_builds_headers():
    provider = _Provider()

    assert await provider.initialize({"api_key": "k", "rate_limit_delay": "oops"}) is True

    assert provider._rate_limit_delay == 0.0
    assert provider._headers["Authorization"] == "Bearer k"
    assert provider._initialized is True


def test_matches_download_url_uses_the_download_host():
    provider = _Provider()

    assert provider.matches_download_url("https://example.test/file")
    assert not provider.matches_download_url("https://other.test/file")


def test_register_provider_adds_the_class_to_the_providers_dict():
    context = HookContext(hook_name="provider.register", plugin_id="example", data={})

    register_provider(context, "example", "Example", lambda: _Provider)

    assert context.data["providers"] == {"example": _Provider}


def test_register_provider_swallows_a_failing_import():
    context = HookContext(hook_name="provider.register", plugin_id="example", data={})

    def broken():
        raise ImportError("nope")

    assert register_provider(context, "example", "Example", broken) is context
    assert "providers" not in context.data


def test_register_provider_swallows_any_failure_from_load():
    context = HookContext(hook_name="provider.register", plugin_id="example", data={})

    def broken():
        raise RuntimeError("boom")

    assert register_provider(context, "example", "Example", broken) is context
    assert "providers" not in context.data


def test_register_provider_keeps_existing_entries():
    existing = object()
    context = HookContext(hook_name="provider.register", plugin_id="example", data={"providers": {"other": existing}})

    register_provider(context, "example", "Example", lambda: _Provider)

    assert context.data["providers"] == {"other": existing, "example": _Provider}

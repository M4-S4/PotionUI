import httpx
import pytest

from src.features.llm.harness.profile import ProfileSource
from src.features.llm.model_listing import ModelListingError
from tests.features.llm.harness.adapters.openai_responses_support import ADAPTER, make_config
from tests.features.llm.harness.wire_peer import install_peer


def json_reply(payload, status=200):
    return lambda request: httpx.Response(status, json=payload)


class TestListing:
    async def test_models_are_listed_sorted_with_their_owner(self, monkeypatch):
        peer = install_peer(monkeypatch, json_reply({"data": [
            {"id": "o-mini", "owned_by": "system"},
            {"id": "gpt-5.6", "owned_by": "openai"},
            {"id": "plain"},
            {"nope": 1},
        ]}))

        models = await ADAPTER.list_models("https://api.openai.com/v1/", "test-key-a")

        assert str(peer.requests[0].url) == "https://api.openai.com/v1/models"
        assert peer.requests[0].headers["authorization"] == "Bearer test-key-a"
        assert [model.id for model in models] == ["gpt-5.6", "o-mini", "plain"]
        assert models[0].details == {"owned_by": "openai"}
        assert models[2].details is None

    async def test_listing_without_a_key_sends_no_authorization(self, monkeypatch):
        peer = install_peer(monkeypatch, json_reply({"data": []}))

        await ADAPTER.list_models("https://api.openai.com/v1")

        assert "authorization" not in peer.requests[0].headers

    async def test_a_reply_without_a_list_is_reported_under_the_openai_name(self, monkeypatch):
        install_peer(monkeypatch, json_reply({"object": "error"}))

        with pytest.raises(ModelListingError, match="OpenAI returned no model list"):
            await ADAPTER.list_models("https://api.openai.com/v1", "test-key-a")

    async def test_rejected_credentials_are_reported(self, monkeypatch):
        install_peer(monkeypatch, json_reply({}, status=401))

        with pytest.raises(ModelListingError, match="rejected the credentials"):
            await ADAPTER.list_models("https://api.openai.com/v1", "bad-key")


class TestDiscovery:
    async def test_discovery_returns_a_discovered_layer(self, monkeypatch):
        install_peer(monkeypatch, json_reply({"data": [{"id": "gpt-5.6", "owned_by": "openai"}]}))

        layer = await ADAPTER.discover(make_config())

        assert layer.source is ProfileSource.DISCOVERED
        assert dict(layer.fields) == {}

    async def test_a_context_window_the_server_reports_is_picked_up(self, monkeypatch):
        install_peer(monkeypatch, json_reply({"data": [
            {"id": "other", "context_window": 1000},
            {"id": "gpt-5.6", "context_length": 400000},
        ]}))

        layer = await ADAPTER.discover(make_config())

        assert dict(layer.fields) == {"context_window": 400000}

    async def test_a_failed_probe_leaves_the_layer_empty(self, monkeypatch):
        install_peer(monkeypatch, json_reply({}, status=500))

        layer = await ADAPTER.discover(make_config())

        assert dict(layer.fields) == {}


import httpx
import pytest

from src.features.llm.harness.profile import CapabilityProfile, ProfileSource, ReasoningKind
from src.features.llm.model_listing import ModelListingError
from tests.features.llm.harness.adapters.openrouter_support import ADAPTER, BASE_URL, MODEL, make_config
from tests.features.llm.harness.wire_peer import install_peer

CATALOG = {"data": [
    {
        "id": MODEL,
        "name": "Anthropic: Claude Sonnet 5.5",
        "context_length": 400000,
        "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
        "pricing": {
            "prompt": "0.000003",
            "completion": "0.000015",
            "input_cache_read": "0.0000003",
            "input_cache_write": "0.00000375",
            "internal_reasoning": "0",
        },
        "top_provider": {"context_length": 400000, "max_completion_tokens": 64000},
        "supported_parameters": ["tools", "tool_choice", "parallel_tool_calls", "reasoning", "include_reasoning", "max_tokens"],
    },
    {
        "id": "meta/plain-8b",
        "name": "Plain 8B",
        "context_length": 8192,
        "architecture": {"input_modalities": ["text"]},
        "pricing": {"prompt": "0", "completion": "0"},
        "supported_parameters": ["max_tokens"],
    },
    {"id": "no-window"},
    {"name": "no id"},
]}


def json_reply(payload, status=200):
    return lambda request: httpx.Response(status, json=payload)


class TestListing:
    async def test_models_are_listed_sorted_with_their_context_length(self, monkeypatch):
        peer = install_peer(monkeypatch, json_reply(CATALOG))

        models = await ADAPTER.list_models(BASE_URL + "/", "or-key-a")

        assert str(peer.requests[0].url) == f"{BASE_URL}/models"
        assert peer.requests[0].headers["authorization"] == "Bearer or-key-a"
        assert [model.id for model in models] == [MODEL, "meta/plain-8b", "no-window"]
        assert models[0].label == "Anthropic: Claude Sonnet 5.5"
        assert models[0].details == {"context_length": 400000}
        assert models[2].details is None

    async def test_listing_without_a_key_sends_no_authorization(self, monkeypatch):
        peer = install_peer(monkeypatch, json_reply({"data": []}))

        await ADAPTER.list_models(BASE_URL)

        assert "authorization" not in peer.requests[0].headers

    async def test_a_reply_without_a_list_is_reported_under_the_adapter_name(self, monkeypatch):
        install_peer(monkeypatch, json_reply({"error": {}}))

        with pytest.raises(ModelListingError, match="OpenRouter returned no model list"):
            await ADAPTER.list_models(BASE_URL, "or-key-a")

    async def test_rejected_credentials_are_reported(self, monkeypatch):
        install_peer(monkeypatch, json_reply({}, status=401))

        with pytest.raises(ModelListingError, match="rejected the credentials"):
            await ADAPTER.list_models(BASE_URL, "bad-key")


class TestDiscovery:
    async def test_a_capable_model_fills_the_profile_layer(self, monkeypatch):
        install_peer(monkeypatch, json_reply(CATALOG))

        layer = await ADAPTER.discover(make_config())
        profile = CapabilityProfile(adapter="openrouter", model=MODEL).layered([layer])

        assert layer.source is ProfileSource.DISCOVERED
        assert profile.context_window == 400000
        assert profile.max_output_tokens == 64000
        assert profile.tools is True
        assert profile.parallel_tools is True
        assert profile.tool_choice == frozenset({"auto", "none", "named"})
        assert profile.reasoning is ReasoningKind.EFFORT
        assert profile.vision is True
        assert (profile.prices.input, profile.prices.output) == (3.0, 15.0)
        assert profile.prices.cached_input == pytest.approx(0.3)
        assert profile.prices.cache_write == pytest.approx(3.75)
        assert profile.prices.reasoning is None

    async def test_a_plain_model_gets_no_tools_or_reasoning(self, monkeypatch):
        install_peer(monkeypatch, json_reply(CATALOG))

        layer = await ADAPTER.discover(make_config(model="meta/plain-8b"))
        profile = CapabilityProfile(adapter="openrouter", model="meta/plain-8b").layered([layer])

        assert profile.context_window == 8192
        assert profile.tools is False
        assert profile.reasoning is ReasoningKind.NONE
        assert profile.vision is False
        assert profile.prices.input == 0.0

    async def test_an_unlisted_model_gives_an_empty_layer(self, monkeypatch):
        install_peer(monkeypatch, json_reply(CATALOG))

        layer = await ADAPTER.discover(make_config(model="who/knows"))

        assert dict(layer.fields) == {}

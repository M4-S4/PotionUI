import httpx
import pytest

from src.features.llm.harness.adapters.ollama_chat import OllamaChatAdapter
from src.features.llm.harness.adapters.openai_chat import OpenAIChatAdapter
from src.features.llm.harness.profile import CapabilityProfile, ProfileSource, ReasoningKind
from src.features.llm.model_listing import ModelListingError
from tests.features.llm.harness.history import make_config
from tests.features.llm.harness.wire_peer import install_peer


def reply(payload, status=200):
    return lambda request: httpx.Response(status, json=payload)


async def test_openai_models_are_listed_sorted_with_their_owner(monkeypatch):
    peer = install_peer(monkeypatch, reply({"data": [{"id": "b-model", "owned_by": "me"}, {"id": "A-model"}, {"nope": 1}]}))

    models = await OpenAIChatAdapter().list_models("http://srv.local:8000/v1/", "sk-1")

    assert str(peer.requests[0].url) == "http://srv.local:8000/v1/models"
    assert peer.requests[0].headers["authorization"] == "Bearer sk-1"
    assert [m.id for m in models] == ["A-model", "b-model"]
    assert models[1].details == {"owned_by": "me"} and models[0].details is None


async def test_openai_models_without_a_key_send_no_authorization(monkeypatch):
    peer = install_peer(monkeypatch, reply({"data": []}))

    await OpenAIChatAdapter().list_models("http://srv.local:8000/v1")

    assert "authorization" not in peer.requests[0].headers


async def test_openai_model_list_without_data_is_an_error(monkeypatch):
    install_peer(monkeypatch, reply({"models": []}))

    with pytest.raises(ModelListingError, match="returned no model list"):
        await OpenAIChatAdapter().list_models("http://srv.local:8000/v1")


@pytest.mark.parametrize("key", ["context_length", "max_model_len", "context_window"])
async def test_openai_discovery_reads_the_window_the_server_reports(monkeypatch, key):
    install_peer(monkeypatch, reply({"data": [{"id": "other", key: 1}, {"id": "golden-model", key: 131072}]}))

    layer = await OpenAIChatAdapter().discover(make_config("openai"))

    assert layer.source is ProfileSource.DISCOVERED
    assert dict(layer.fields) == {"context_window": 131072}


async def test_discovered_facts_are_accepted_by_the_profile_layering(monkeypatch):
    install_peer(monkeypatch, reply({"data": [{"id": "golden-model", "max_model_len": 4096}]}))
    layer = await OpenAIChatAdapter().discover(make_config("openai"))

    profile = CapabilityProfile(adapter="openai-chat", model="golden-model").layered([layer])

    assert profile.context_window == 4096
    assert profile.sources["context_window"] is ProfileSource.DISCOVERED


async def test_openai_discovery_reports_nothing_for_an_unlisted_model(monkeypatch):
    install_peer(monkeypatch, reply({"data": [{"id": "other", "context_length": 5}]}))

    assert dict((await OpenAIChatAdapter().discover(make_config("openai"))).fields) == {}


async def test_openai_discovery_never_raises(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    install_peer(monkeypatch, refuse)

    assert dict((await OpenAIChatAdapter().discover(make_config("openai"))).fields) == {}


async def test_ollama_models_carry_size_and_details(monkeypatch):
    peer = install_peer(monkeypatch, reply({"models": [
        {
            "name": "qwen3:8b", "size": 5200000000, "modified_at": "2026-09-01T10:00:00Z",
            "details": {"parameter_size": "8.2B", "quantization_level": "Q4_K_M", "family": "qwen3"},
        },
        {"name": "Llama3.2:latest", "size": 2000000000, "details": {}},
    ]}))

    models = await OllamaChatAdapter().list_models("http://ollama.local:11434/", "k")

    assert str(peer.requests[0].url) == "http://ollama.local:11434/api/tags"
    assert peer.requests[0].headers["authorization"] == "Bearer k"
    assert [m.id for m in models] == ["Llama3.2:latest", "qwen3:8b"]
    assert models[1].details == {"parameter_size": "8.2B", "quantization": "Q4_K_M", "family": "qwen3"}
    assert models[0].details is None


async def test_ollama_model_list_without_models_is_an_error(monkeypatch):
    install_peer(monkeypatch, reply({"data": []}))

    with pytest.raises(ModelListingError):
        await OllamaChatAdapter().list_models("http://ollama.local:11434")


async def discover_ollama(monkeypatch, payload, status=200):
    peer = install_peer(monkeypatch, reply(payload, status))
    layer = await OllamaChatAdapter().discover(make_config("ollama"))
    return dict(layer.fields), peer


async def test_ollama_discovery_posts_the_model_to_api_show(monkeypatch):
    _, peer = await discover_ollama(monkeypatch, {})

    assert str(peer.requests[0].url) == "http://peer.invalid:1234/api/show"
    assert peer.body == {"model": "golden-model"}


async def test_an_explicit_num_ctx_in_the_modelfile_wins_over_the_trained_length(monkeypatch):
    fields, _ = await discover_ollama(
        monkeypatch,
        {"parameters": "stop <|im_end|>\nnum_ctx                        16384\n", "model_info": {"qwen3.context_length": 262144}},
    )

    assert fields == {"context_window": 16384}


async def test_a_trained_window_is_capped_so_a_huge_default_never_becomes_num_ctx(monkeypatch):
    fields, _ = await discover_ollama(monkeypatch, {"model_info": {"llama.context_length": 131072}})

    assert fields == {"context_window": 32768}


async def test_a_small_trained_window_is_kept(monkeypatch):
    fields, _ = await discover_ollama(monkeypatch, {"model_info": {"llama.context_length": 8192}})

    assert fields == {"context_window": 8192}


async def test_capabilities_become_profile_facts_when_the_server_reports_them(monkeypatch):
    fields, _ = await discover_ollama(monkeypatch, {"capabilities": ["completion", "tools", "thinking", "vision"]})

    assert fields == {"tools": True, "vision": True, "reasoning": "toggle"}


async def test_ollama_facts_are_accepted_by_the_profile_layering(monkeypatch):
    install_peer(monkeypatch, reply({"capabilities": ["tools", "thinking"], "model_info": {"qwen3.context_length": 8192}}))
    layer = await OllamaChatAdapter().discover(make_config("ollama"))

    profile = CapabilityProfile(adapter="ollama-chat", model="golden-model").layered([layer])

    assert (profile.tools, profile.vision, profile.reasoning, profile.context_window) == (True, False, ReasoningKind.TOGGLE, 8192)


async def test_a_model_without_those_capabilities_is_reported_as_lacking_them(monkeypatch):
    fields, _ = await discover_ollama(monkeypatch, {"capabilities": ["completion"]})

    assert fields == {"tools": False, "vision": False, "reasoning": "none"}


async def test_an_older_server_without_capabilities_claims_nothing(monkeypatch):
    fields, _ = await discover_ollama(monkeypatch, {"details": {"family": "llama"}})

    assert fields == {}


async def test_ollama_discovery_never_raises(monkeypatch):
    fields, _ = await discover_ollama(monkeypatch, {"error": "nope"}, status=404)

    assert fields == {}

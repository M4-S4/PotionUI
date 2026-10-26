import httpx
import pytest

from src.features.llm.harness.adapters.ollama_chat import OllamaChatAdapter
from src.features.llm.harness.adapters.openai_chat import OpenAIChatAdapter
from src.features.llm.harness.profile import CapabilityProfile, ProfileSource, ReasoningKind
from src.features.llm.model_listing import DiscoveredWindow, ModelListingError
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


async def test_openai_discovery_names_a_vllm_server_from_the_listing_owner(monkeypatch):
    install_peer(monkeypatch, reply({"data": [{"id": "golden-model", "owned_by": "vllm", "max_model_len": 4096}]}))

    layer = await OpenAIChatAdapter().discover(make_config("openai"))

    assert dict(layer.fields) == {"context_window": 4096, "server": "vllm"}


async def test_openai_discovery_names_the_server_even_when_the_model_is_not_listed_under_that_id(monkeypatch):
    install_peer(monkeypatch, reply({"data": [{"id": "alias", "owned_by": "vllm"}]}))

    layer = await OpenAIChatAdapter().discover(make_config("openai"))

    assert dict(layer.fields) == {"server": "vllm"}


@pytest.mark.parametrize("owner", ["openai", "organization-owner", None, 5])
async def test_openai_discovery_names_no_server_for_other_owners(monkeypatch, owner):
    install_peer(monkeypatch, reply({"data": [{"id": "golden-model", "owned_by": owner}]}))

    assert "server" not in (await OpenAIChatAdapter().discover(make_config("openai"))).fields


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


async def test_openai_context_window_probe_returns_the_reported_window_as_a_runtime_window(monkeypatch):
    install_peer(monkeypatch, reply({"data": [{"id": "golden-model", "max_model_len": 4096}]}))

    window = await OpenAIChatAdapter().discover_context_window("http://srv.local:8000/v1", "k", "golden-model")

    assert window == DiscoveredWindow(tokens=4096, kind="runtime")


async def test_openai_context_window_probe_is_none_for_an_unlisted_model_or_a_dead_server(monkeypatch):
    install_peer(monkeypatch, reply({"data": [{"id": "other", "context_length": 5}]}))
    assert await OpenAIChatAdapter().discover_context_window("http://srv.local:8000/v1", None, "golden-model") is None

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    install_peer(monkeypatch, refuse)
    assert await OpenAIChatAdapter().discover_context_window("http://srv.local:8000/v1", None, "golden-model") is None


async def test_ollama_context_window_probe_keeps_the_trained_length_uncapped_and_labelled(monkeypatch):
    peer = install_peer(monkeypatch, reply({"model_info": {"llama.context_length": 131072}}))

    window = await OllamaChatAdapter().discover_context_window("http://ollama.local:11434", "k", "m")

    assert window == DiscoveredWindow(tokens=131072, kind="trained")
    assert peer.body == {"model": "m"} and peer.requests[0].headers["authorization"] == "Bearer k"


async def test_ollama_context_window_probe_prefers_the_modelfile_num_ctx_and_never_raises(monkeypatch):
    install_peer(monkeypatch, reply({"parameters": "num_ctx 16384", "model_info": {"x.context_length": 262144}}))
    assert await OllamaChatAdapter().discover_context_window("http://o", None, "m") == DiscoveredWindow(tokens=16384, kind="runtime")

    install_peer(monkeypatch, reply({"error": "nope"}, status=500))
    assert await OllamaChatAdapter().discover_context_window("http://o", None, "m") is None


async def test_release_unloads_the_model_with_a_zero_keep_alive(monkeypatch):
    peer = install_peer(monkeypatch, reply({"done": True}))

    released = await OllamaChatAdapter().release(make_config("ollama", api_key="k"))

    assert released is True
    assert (peer.requests[0].method, str(peer.requests[0].url)) == ("POST", "http://peer.invalid:1234/api/chat")
    assert peer.body == {"model": "golden-model", "keep_alive": 0}
    assert peer.requests[0].headers["authorization"] == "Bearer k"


async def test_release_swallows_every_failure_and_reports_it_as_false(monkeypatch):
    install_peer(monkeypatch, reply({"error": "busy"}, status=500))
    assert await OllamaChatAdapter().release(make_config("ollama")) is False

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    install_peer(monkeypatch, refuse)
    assert await OllamaChatAdapter().release(make_config("ollama")) is False

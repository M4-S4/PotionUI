import json
from types import SimpleNamespace

import pytest

from src.features.llm.clients.openai import OpenAIClient
from src.features.llm.gateway import LLMGateway
from src.platform.plugins.llm_providers import LLMProviderRegistration, LLMProviderRegistry
from tests.features.llm.wire_capture import (
    OPENAI_REPLY,
    collect,
    install_wire_capture,
    json_response,
    make_config,
    sse_response,
    tool_history,
    tool_schemas,
)

TYPE_NAME = "example-llm"
LABEL = "Example LLM"
BASE_URL = "https://llm.example.org/v1"
PLUGIN_ID = "example-plugin"
SSE_EVENTS = [json.dumps({"choices": [{"delta": {"content": "ok"}}]}), "[DONE]"]


class RenamingClient(OpenAIClient):
    def adapt_request(self, request, config):
        request.payload["max_tokens"] = request.payload.pop("max_completion_tokens")
        return request


def registry_with(type_name):
    registry = LLMProviderRegistry()
    registry.register(LLMProviderRegistration(
        type_name=type_name, label=LABEL, client=RenamingClient(), default_base_url=BASE_URL, source=PLUGIN_ID,
    ))
    return registry


def config_of(type_name):
    return make_config(type_name, base_url=BASE_URL, provider_options={"context_window": 8192})


def gateway_for(registry, config=None):
    return LLMGateway(SimpleNamespace(get_configuration=lambda config_id: config), provider_registry=registry)


def test_provider_types_lists_the_builtins_then_the_contributed_type():
    types = gateway_for(registry_with(TYPE_NAME)).provider_types()

    assert [entry["type"] for entry in types] == ["ollama", "openai", "native", TYPE_NAME]
    assert all(entry["label"] is None and entry["default_base_url"] is None for entry in types[:-1])
    assert types[-1] == {
        "type": TYPE_NAME, "label": LABEL, "default_base_url": BASE_URL, "supports_model_listing": True,
    }


@pytest.mark.asyncio
async def test_a_contributed_type_named_like_a_builtin_is_ignored(monkeypatch):
    config = config_of("openai")
    gateway = gateway_for(registry_with("openai"), config)
    capture = install_wire_capture(monkeypatch, [json_response(OPENAI_REPLY)])

    await gateway.generate_with_tools(tool_history(), config.id, tool_schemas())

    assert [entry["type"] for entry in gateway.provider_types()].count("openai") == 1
    assert "max_completion_tokens" in capture.body and "max_tokens" not in capture.body


@pytest.mark.asyncio
async def test_the_contributed_client_serves_its_type(monkeypatch):
    config = config_of(TYPE_NAME)
    capture = install_wire_capture(monkeypatch, [json_response(OPENAI_REPLY)])

    response = await gateway_for(registry_with(TYPE_NAME), config).generate_with_tools(
        tool_history(), config.id, tool_schemas(),
    )

    assert response.content == "canned reply"
    assert capture.url == f"{BASE_URL}/chat/completions"
    assert capture.body["max_tokens"] == config.max_tokens
    assert "max_completion_tokens" not in capture.body


@pytest.mark.asyncio
async def test_the_contributed_type_lists_its_models(monkeypatch):
    capture = install_wire_capture(monkeypatch, [json_response({"data": [{"id": "example-model"}]})])

    models = await gateway_for(registry_with(TYPE_NAME)).list_models(TYPE_NAME, BASE_URL, "sk-1")

    assert [model.id for model in models] == ["example-model"]
    assert capture.url == f"{BASE_URL}/models"


@pytest.mark.asyncio
async def test_disabling_the_plugin_withdraws_the_type():
    config = config_of(TYPE_NAME)
    registry = registry_with(TYPE_NAME)
    gateway = gateway_for(registry, config)

    registry.unregister_source(PLUGIN_ID)

    assert TYPE_NAME not in [entry["type"] for entry in gateway.provider_types()]
    with pytest.raises(ValueError, match=f"Unsupported LLM type: {TYPE_NAME}"):
        await gateway.generate_with_tools(tool_history(), config.id, tool_schemas())


@pytest.mark.asyncio
@pytest.mark.parametrize("call, reply", [
    (lambda client, config: client.generate("hello", config, "SYS"), json_response(OPENAI_REPLY)),
    (
        lambda client, config: client.generate_with_tools(tool_history(), config, "SYS", tool_schemas()),
        json_response(OPENAI_REPLY),
    ),
    (
        lambda client, config: collect(client.stream_with_tools(tool_history(), config, "SYS", tool_schemas())),
        sse_response(SSE_EVENTS),
    ),
])
async def test_every_send_path_puts_the_adapted_request_on_the_wire(monkeypatch, call, reply):
    capture = install_wire_capture(monkeypatch, [reply])

    await call(RenamingClient(), config_of(TYPE_NAME))

    assert capture.body["max_tokens"] == config_of(TYPE_NAME).max_tokens
    assert "max_completion_tokens" not in capture.body

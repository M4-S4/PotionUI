import json
from types import SimpleNamespace

import pytest

from src.features.llm import trace_collector
from src.features.llm.context_windows import ContextWindows
from src.features.llm.gateway import LLMGateway
from src.features.llm.harness.adapter import ToolSpec
from src.features.llm.harness.adapters.ollama_chat import OllamaChatAdapter
from src.features.llm.harness.adapters.openai_chat import OpenAIChatAdapter
from src.features.llm.harness.profiles import LayeredProfileResolver
from src.features.llm.harness.build import user_message
from src.features.llm.types import LLMType, LLMTypes
from src.platform.plugins.llm_providers import LLMProviderRegistration, LLMProviderRegistry
from tests.features.llm.wire_capture import install_wire_capture, json_response, make_config, sse_response, tool_schemas

TYPE_NAME = "example-llm"
LABEL = "Example LLM"
BASE_URL = "https://llm.example.org/v1"
PLUGIN_ID = "example-plugin"
SSE_EVENTS = [json.dumps({"choices": [{"delta": {"content": "canned reply"}}]}), "[DONE]"]


class RenamingAdapter(OpenAIChatAdapter):
    name = "renaming-chat"

    def build_body(self, request, profile):
        body = super().build_body(request, profile)
        body["max_tokens"] = body.pop("max_completion_tokens")
        return body


def registry_with(type_name):
    registry = LLMProviderRegistry()
    registry.register(LLMProviderRegistration(
        type_name=type_name, label=LABEL, adapter=RenamingAdapter(), default_base_url=BASE_URL, source=PLUGIN_ID,
    ))
    return registry


def config_of(type_name):
    return make_config(type_name, base_url=BASE_URL, provider_options={"context_window": 8192})


def gateway_for(registry, config=None):
    types = LLMTypes(
        core=(
            LLMType(type="ollama", adapter=OllamaChatAdapter()),
            LLMType(type="openai-compatible", adapter=OpenAIChatAdapter()),
        ),
        provider_registry=registry,
    )
    windows = ContextWindows(types)
    repository = SimpleNamespace(get_configuration=lambda config_id: config)
    return LLMGateway(repository, types, LayeredProfileResolver(types.adapter_for), windows)


async def send(gateway, config):
    return await gateway.complete(
        config.id, [user_message("hi")], tools=[ToolSpec.from_schema(schema) for schema in tool_schemas()],
    )


def test_provider_types_lists_the_builtins_then_the_contributed_type():
    types = gateway_for(registry_with(TYPE_NAME)).provider_types()

    assert [entry["type"] for entry in types] == ["ollama", "openai-compatible", TYPE_NAME]
    assert all(entry["label"] is None and entry["default_base_url"] is None for entry in types[:-1])
    assert types[-1] == {
        "type": TYPE_NAME, "label": LABEL, "adapter": "renaming-chat", "default_base_url": BASE_URL,
        "supports_model_listing": True, "source": PLUGIN_ID,
    }


@pytest.mark.asyncio
async def test_a_contributed_type_named_like_a_builtin_is_ignored(monkeypatch):
    config = config_of("openai-compatible")
    gateway = gateway_for(registry_with("openai-compatible"), config)
    capture = install_wire_capture(monkeypatch, [sse_response(SSE_EVENTS)])

    await send(gateway, config)

    assert [entry["type"] for entry in gateway.provider_types()].count("openai-compatible") == 1
    assert "max_completion_tokens" in capture.body and "max_tokens" not in capture.body


@pytest.mark.asyncio
async def test_the_contributed_adapter_serves_its_type(monkeypatch):
    config = config_of(TYPE_NAME)
    capture = install_wire_capture(monkeypatch, [sse_response(SSE_EVENTS)])

    completion = await send(gateway_for(registry_with(TYPE_NAME), config), config)

    assert completion.text == "canned reply"
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
        await send(gateway, config)


@pytest.mark.asyncio
async def test_the_trace_names_the_type_and_records_the_params_that_went_out(monkeypatch):
    config = config_of(TYPE_NAME)
    capture = install_wire_capture(monkeypatch, [sse_response(SSE_EVENTS)])

    with trace_collector.activate("session-1", "user-1", purpose="chat"):
        await send(gateway_for(registry_with(TYPE_NAME), config), config)

    assert capture.trace["provider"] == TYPE_NAME
    assert capture.trace["adapter"] == "renaming-chat"
    assert capture.trace["request_params"]["max_tokens"] == config.max_tokens
    assert "max_completion_tokens" not in capture.trace["request_params"]

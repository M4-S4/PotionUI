import contextlib

import httpx
import pytest

from src.features.llm.harness.adapter import RenderRequest
from src.features.llm.harness.adapters.anthropic_messages import AnthropicMessagesAdapter
from src.features.llm.harness.adapters.gemini_interactions import GeminiInteractionsAdapter
from src.features.llm.harness.adapters.ollama_chat import OllamaChatAdapter
from src.features.llm.harness.adapters.openai_chat import OpenAIChatAdapter
from src.features.llm.harness.adapters.openai_responses import OpenAIResponsesAdapter
from src.features.llm.harness.adapters.openrouter import OpenRouterAdapter
from src.features.llm.harness.build import user_message
from src.features.llm.harness.profile import CapabilityProfile
from src.platform.http.tls import client_ssl_context
from tests.features.llm.wire_capture import install_wire_capture, json_response, make_config

ADAPTERS = [
    (OllamaChatAdapter, "ollama"),
    (OpenAIChatAdapter, "openai-compatible"),
    (OpenAIResponsesAdapter, "openai"),
    (OpenRouterAdapter, "openrouter"),
    (AnthropicMessagesAdapter, "anthropic"),
    (GeminiInteractionsAdapter, "gemini"),
]


@pytest.fixture
def constructed(monkeypatch):
    install_wire_capture(monkeypatch, [json_response({})])
    patched = httpx.AsyncClient
    seen = []

    def recording(*args, **kwargs):
        seen.append(kwargs)
        return patched(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", recording)
    return seen


async def _send(adapter, config):
    request = RenderRequest(config=config, system="sys", messages=(user_message("hi"),))
    wire = adapter.render(request, CapabilityProfile(adapter=adapter.name, model=config.model))
    async for _ in adapter.send(wire, request):
        pass


CALLS = {
    "send": _send,
    "list_models": lambda adapter, config: adapter.list_models("http://peer.invalid:1234", "sk-1"),
    "discover": lambda adapter, config: adapter.discover(config),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cls,provider", ADAPTERS)
@pytest.mark.parametrize("call", sorted(CALLS))
async def test_every_llm_http_adapter_uses_the_shared_tls_context(constructed, adapter_cls, provider, call):
    with contextlib.suppress(Exception):
        await CALLS[call](adapter_cls(), make_config(provider, api_key="sk-1"))

    assert constructed
    assert all(kwargs.get("verify") is client_ssl_context() for kwargs in constructed)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cls,provider", ADAPTERS)
async def test_a_streamed_call_keeps_an_unbounded_read_timeout(constructed, adapter_cls, provider):
    with contextlib.suppress(Exception):
        await _send(adapter_cls(), make_config(provider, api_key="sk-1"))

    assert constructed[0]["timeout"].read is None

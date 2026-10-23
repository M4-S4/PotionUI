import contextlib

import httpx
import pytest

from src.features.llm.clients.ollama import OllamaClient
from src.features.llm.clients.openai import OpenAIClient
from src.platform.http.tls import client_ssl_context
from tests.features.llm.wire_capture import json_response, install_wire_capture, make_config


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


async def _drain(stream):
    async for _ in stream:
        pass


CALLS = {
    "generate": lambda c, cfg: c.generate("hi", cfg, "sys"),
    "generate_with_tools": lambda c, cfg: c.generate_with_tools(
        [{"role": "user", "content": "hi"}], cfg, "sys"
    ),
    "stream_with_tools": lambda c, cfg: _drain(
        c.stream_with_tools([{"role": "user", "content": "hi"}], cfg, "sys")
    ),
    "list_models": lambda c, cfg: c.list_models("http://peer.invalid:1234"),
    "discover_context_window": lambda c, cfg: c.discover_context_window(
        "http://peer.invalid:1234", None, cfg.model
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("client_cls,provider", [(OpenAIClient, "openai"), (OllamaClient, "ollama")])
@pytest.mark.parametrize("call", sorted(CALLS))
async def test_every_llm_http_client_uses_the_shared_tls_context(constructed, client_cls, provider, call):
    config = make_config(provider)
    with contextlib.suppress(Exception):
        await CALLS[call](client_cls(), config)

    assert constructed
    assert all(kwargs.get("verify") is client_ssl_context() for kwargs in constructed)


@pytest.mark.asyncio
async def test_streaming_client_keeps_unbounded_read_timeout(constructed):
    config = make_config("openai")
    with contextlib.suppress(Exception):
        await CALLS["stream_with_tools"](OpenAIClient(), config)

    assert constructed[0]["timeout"].read is None

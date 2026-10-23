from unittest.mock import Mock

import pytest

from src.features.llm.clients.openai import OpenAIClient
from src.features.llm.gateway import LLMGateway
from tests.features.llm.wire_capture import OLLAMA_REPLY, install_wire_capture, json_response, make_config

HISTORY = [{"role": "user", "content": "hello"}]
EMPTY = {"message": {"content": ""}}


def _gateway(config):
    gateway = LLMGateway(llm_repository=Mock())
    gateway.repository.get_configuration.return_value = config
    return gateway


@pytest.mark.asyncio
async def test_ollama_history_retries_empty_completions_until_content(monkeypatch):
    capture = install_wire_capture(
        monkeypatch, [json_response(EMPTY), json_response(EMPTY), json_response(OLLAMA_REPLY)]
    )
    config = make_config("ollama", provider_options={"context_window": 8192})

    response = await _gateway(config).generate_with_history(HISTORY, "cfg-1")

    assert len(capture.requests) == 3
    assert response.content == "canned reply"


@pytest.mark.asyncio
async def test_ollama_history_gives_up_after_three_empty_completions(monkeypatch):
    capture = install_wire_capture(monkeypatch, [json_response(EMPTY)] * 3)
    config = make_config("ollama", provider_options={"context_window": 8192})

    response = await _gateway(config).generate_with_history(HISTORY, "cfg-1")

    assert len(capture.requests) == 3
    assert response.content == ""


@pytest.mark.asyncio
async def test_openai_null_content_without_tool_calls_is_an_empty_string(monkeypatch):
    install_wire_capture(
        monkeypatch, [json_response({"choices": [{"message": {"content": None}, "finish_reason": "stop"}]})]
    )

    response = await OpenAIClient().generate_with_tools(HISTORY, make_config("openai"), "SYS", tools=None)

    assert response.content == ""
    assert response.tool_calls is None

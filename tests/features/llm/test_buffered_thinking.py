import pytest

from src.features.llm.clients.ollama import OllamaClient
from src.features.llm.clients.openai import OpenAIClient
from tests.features.llm.wire_capture import install_wire_capture, json_response, make_config

HISTORY = [{"role": "user", "content": "hello"}]


def _ollama_reply(content="the answer", **message):
    return {
        "message": {"role": "assistant", "content": content, **message},
        "done": True,
        "done_reason": "stop",
    }


def _openai_reply(content="the answer", **message):
    return {
        "choices": [{"message": {"role": "assistant", "content": content, **message}, "finish_reason": "stop"}],
        "usage": {},
    }


async def _ollama(monkeypatch, reply, **config):
    install_wire_capture(monkeypatch, [json_response(reply)])
    return await OllamaClient().generate_with_tools(HISTORY, make_config("ollama", **config), "SYS")


async def _openai(monkeypatch, reply):
    install_wire_capture(monkeypatch, [json_response(reply)])
    return await OpenAIClient().generate_with_tools(HISTORY, make_config("openai"), "SYS")


async def test_ollama_reports_the_reasoning_beside_the_answer(monkeypatch):
    response = await _ollama(monkeypatch, _ollama_reply(thinking="let me think"))

    assert response.content == "the answer"
    assert response.thinking == "let me think"


async def test_ollama_without_reasoning_reports_none(monkeypatch):
    response = await _ollama(monkeypatch, _ollama_reply())

    assert response.thinking is None


async def test_ollama_with_only_reasoning_keeps_its_fallback_content_while_thinking_is_not_shown(monkeypatch):
    response = await _ollama(monkeypatch, _ollama_reply(content="", thinking="only thoughts"))

    assert response.content == "only thoughts"
    assert response.thinking == "only thoughts"


async def test_ollama_with_only_reasoning_does_not_duplicate_it_into_the_answer_when_shown(monkeypatch):
    response = await _ollama(
        monkeypatch, _ollama_reply(content="", thinking="only thoughts"), show_thinking=True,
    )

    assert response.content == ""
    assert response.thinking == "only thoughts"


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
async def test_openai_reports_the_reasoning_beside_the_answer(monkeypatch, key):
    response = await _openai(monkeypatch, _openai_reply(**{key: "let me think"}))

    assert response.content == "the answer"
    assert response.thinking == "let me think"


async def test_openai_without_reasoning_reports_none(monkeypatch):
    response = await _openai(monkeypatch, _openai_reply())

    assert response.thinking is None


async def test_openai_structured_reasoning_is_ignored(monkeypatch):
    response = await _openai(monkeypatch, _openai_reply(reasoning=[{"text": "x"}]))

    assert response.thinking is None

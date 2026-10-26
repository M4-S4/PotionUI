import json
from pathlib import Path

import pytest

from src.features.llm.clients.native import NativeLLMClient
from src.features.llm.clients.ollama import OllamaClient
from src.features.llm.clients.ollama_wire import build_prompt_tools_text
from src.features.llm.clients.openai import OpenAIClient
from src.features.llm.harness import prompt_tools
from tests.features.llm.harness import ollama_chat_cases, openai_chat_cases
from tests.features.llm.harness.history import make_config, tool_schemas
from tests.features.llm.harness.test_prompt_tools import COMPLEX, specs
from tests.features.llm.wire_capture import (
    OLLAMA_REPLY,
    collect,
    install_wire_capture,
    json_response,
    sse_response,
)


@pytest.mark.parametrize("schemas", [tool_schemas(), COMPLEX, COMPLEX[1:]], ids=["simple", "complex", "unsynthesisable"])
def test_the_system_tool_text_matches_the_legacy_ollama_text(schemas):
    assert prompt_tools.render_tools_text(specs(schemas)) == build_prompt_tools_text(schemas)


@pytest.mark.parametrize("schemas", [tool_schemas(), COMPLEX, COMPLEX[1:]], ids=["simple", "complex", "unsynthesisable"])
def test_the_inline_tool_text_matches_the_legacy_native_text(schemas):
    assert prompt_tools.render_inline_tools_text(specs(schemas)) == NativeLLMClient._inject_tools_into_system_message("", schemas)


GOLDENS = Path(__file__).resolve().parents[3] / "fixtures" / "llm_goldens"
OPENAI_GOLDEN = json.loads((GOLDENS / "openai_chat.json").read_text(encoding="utf-8"))
OLLAMA_GOLDEN = json.loads((GOLDENS / "ollama_chat.json").read_text(encoding="utf-8"))
SSE = sse_response([json.dumps({"choices": [{"delta": {"content": "x"}}]}), "[DONE]"])


@pytest.mark.parametrize("case", openai_chat_cases.CASES, ids=[case.id for case in openai_chat_cases.CASES])
async def test_the_openai_goldens_are_what_the_legacy_stream_sent(monkeypatch, case):
    capture = install_wire_capture(monkeypatch, [SSE])

    await collect(OpenAIClient().stream_with_tools(
        case.history, make_config("openai", **case.config), case.system,
        tools=case.tools, image_data=case.image, options_override=case.override,
    ))

    expected = OPENAI_GOLDEN[case.id]
    assert capture.url == expected["url"]
    assert capture.headers.get("authorization") == expected["authorization"]
    assert capture.requests[-1].content == json.dumps(
        expected["body"], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


@pytest.mark.parametrize("case", ollama_chat_cases.CASES, ids=[case.id for case in ollama_chat_cases.CASES])
async def test_the_ollama_goldens_are_what_the_legacy_client_sent_with_streaming_switched_on(monkeypatch, case):
    capture = install_wire_capture(monkeypatch, [json_response(OLLAMA_REPLY)])
    options = dict(case.config)
    if case.profile.get("prompt_tools"):
        options["provider_options"] = {**(options.get("provider_options") or {}), "force_prompt_tools": True}

    await OllamaClient().generate_with_tools(
        case.history, make_config("ollama", **options), case.system,
        tools=case.tools, image_data=case.image, options_override=case.override,
    )

    sent = capture.body
    sent["stream"] = True
    expected = OLLAMA_GOLDEN[case.id]
    assert capture.url == expected["url"]
    assert json.dumps(sent, ensure_ascii=False, separators=(",", ":")) == json.dumps(
        expected["body"], ensure_ascii=False, separators=(",", ":")
    )

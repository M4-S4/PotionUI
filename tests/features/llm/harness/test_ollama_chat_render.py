import json
from pathlib import Path

import pytest

from src.features.llm.harness.adapter import Adapter, RenderRequest, Settings, ToolSpec
from src.features.llm.harness.adapters.ollama_chat import OllamaChatAdapter
from src.features.llm.harness.errors import InvalidRequest
from src.features.llm.harness.messages import (
    ContextPart,
    ImagePart,
    Message,
    Origin,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from tests.features.llm.harness.cassettes import request_bytes
from tests.features.llm.harness.history import (
    IMAGE_B64,
    make_config,
    messages_from_legacy,
    profile_with,
    tool_schemas,
)
from tests.features.llm.harness.ollama_chat_cases import CASES, HISTORY

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[3] / "fixtures" / "llm_goldens" / "ollama_chat.json").read_text(encoding="utf-8")
)
ORIGIN = Origin(adapter="ollama-chat", provider="ollama-chat", model="golden-model")


def request_for(case) -> RenderRequest:
    return RenderRequest(
        config=make_config("ollama", **case.config),
        system=case.system,
        messages=messages_from_legacy(case.history, ORIGIN, case.image),
        tools=tuple(ToolSpec.from_schema(schema) for schema in case.tools or []),
        settings=Settings.from_options(case.override),
    )


def simple(history=HISTORY, **kwargs) -> RenderRequest:
    return RenderRequest(config=make_config("ollama"), system="SYS", messages=messages_from_legacy(history, ORIGIN), **kwargs)


def render(request: RenderRequest, **profile):
    return OllamaChatAdapter().render(request, profile_with("ollama-chat", **profile))


def test_the_adapter_satisfies_the_adapter_protocol():
    assert isinstance(OllamaChatAdapter(), Adapter)


def test_every_golden_case_is_covered_by_a_twin():
    assert set(GOLDEN) == {case.id for case in CASES}


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_the_new_adapter_renders_the_legacy_request_byte_for_byte(case):
    expected = GOLDEN[case.id]

    wire = OllamaChatAdapter().render(request_for(case), profile_with("ollama-chat", **case.profile))

    assert wire.url == expected["url"]
    assert wire.headers.get("Authorization") == expected["authorization"]
    assert wire.content() == request_bytes(expected["body"])


def test_the_request_always_streams_to_the_chat_endpoint():
    wire = render(simple())

    assert wire.url == "http://peer.invalid:1234/api/chat"
    assert wire.body["stream"] is True and wire.stream


def test_an_api_key_goes_out_as_a_bearer_header():
    request = RenderRequest(config=make_config("ollama", api_key="sk-cloud"), system="", messages=())

    assert render(request).headers["Authorization"] == "Bearer sk-cloud"


def test_num_ctx_comes_from_the_profile_window_and_a_saved_num_ctx_is_not_a_second_source():
    request = RenderRequest(
        config=make_config("ollama", provider_options={"num_ctx": 4096}), system="", messages=()
    )

    assert "num_ctx" not in render(request).body["options"]
    assert render(request, context_window=65536).body["options"]["num_ctx"] == 65536


def test_num_ctx_sits_right_after_the_two_base_options():
    request = RenderRequest(config=make_config("ollama", provider_options={"top_k": 20}), system="", messages=())

    assert list(render(request, context_window=65536).body["options"]) == ["temperature", "num_predict", "num_ctx", "top_k"]


@pytest.mark.parametrize(
    "configured, loaded, expected",
    [
        (None, False, 0),
        ("5m", False, "5m"),
        (0, False, 0),
        (None, True, "5m"),
        ("30m", True, "30m"),
        (0, True, "5m"),
        ("0", True, "5m"),
    ],
)
def test_keep_alive_follows_the_config_and_the_keep_loaded_setting(configured, loaded, expected):
    options = {} if configured is None else {"keep_alive": configured}
    request = RenderRequest(
        config=make_config("ollama", provider_options=options),
        system="",
        messages=(),
        settings=Settings(keep_model_loaded=loaded),
    )

    assert render(request).body["keep_alive"] == expected


def test_a_call_without_a_server_id_goes_back_with_its_internal_id():
    call = ToolCallPart(call_id="call_ab_0", name="t", args={"a": 1})
    result = ToolResultPart(call_id="call_ab_0", name="t", content=(TextPart("done"),))
    messages = (
        Message("u", Role.USER, (TextPart("go"),)),
        Message("a", Role.ASSISTANT, (call,), origin=ORIGIN),
        Message("t", Role.TOOL, (result,)),
    )
    request = RenderRequest(config=make_config("ollama"), system="", messages=messages)

    sent = render(request).body["messages"]

    assert sent[1]["tool_calls"][0]["id"] == "call_ab_0"
    assert sent[2] == {"role": "tool", "content": "done", "tool_call_id": "call_ab_0"}


def test_plain_assistant_text_with_reasoning_carries_its_thinking():
    assistant = Message("a", Role.ASSISTANT, (ReasoningPart(text="because"), TextPart("hi")), origin=ORIGIN)
    request = RenderRequest(config=make_config("ollama"), system="", messages=(assistant,))

    assert render(request).body["messages"][0] == {"role": "assistant", "content": "hi", "thinking": "because"}


def test_an_unresolved_image_reference_is_refused_rather_than_dropped():
    message = Message("u", Role.USER, (TextPart("look"), ImagePart(media_type="image/png", ref="/tmp/x.png")))
    request = RenderRequest(config=make_config("ollama"), system="", messages=(message,))

    with pytest.raises(InvalidRequest):
        render(request)


def test_images_are_sent_as_bare_base64_next_to_the_text():
    message = Message("u", Role.USER, (TextPart("q"), ImagePart(media_type="image/png", data=IMAGE_B64)))
    request = RenderRequest(config=make_config("ollama"), system="", messages=(message,))

    assert render(request).body["messages"][0] == {"role": "user", "content": "q", "images": [IMAGE_B64]}


def test_context_parts_render_the_way_the_legacy_context_block_did():
    message = Message("u", Role.USER, (TextPart("question"), ContextPart(kind="memory", text="notes")))
    request = RenderRequest(config=make_config("ollama"), system="", messages=(message,))

    assert render(request).body["messages"][0]["content"] == "question\n\n<context>\nnotes\n</context>"


def test_tools_are_left_out_when_the_profile_says_the_model_has_none_and_prompt_mode_is_off():
    request = simple(tools=tuple(ToolSpec.from_schema(schema) for schema in tool_schemas()))

    assert "tools" not in render(request, tools=False).body
    assert "tools" in render(request).body


def test_prompt_tools_mode_puts_the_tool_text_in_the_system_turn_and_sends_no_tools_key():
    request = simple(tools=tuple(ToolSpec.from_schema(schema) for schema in tool_schemas()))

    body = render(request, prompt_tools=True).body

    assert "tools" not in body
    assert body["messages"][0]["content"].startswith("SYS\n\n## Available Tools")


def test_prompt_tools_mode_with_no_system_prompt_still_carries_the_tool_text():
    request = RenderRequest(
        config=make_config("ollama"),
        system="",
        messages=messages_from_legacy(HISTORY, ORIGIN),
        tools=tuple(ToolSpec.from_schema(schema) for schema in tool_schemas()),
    )

    assert render(request, prompt_tools=True).body["messages"][0]["content"].startswith("\n\n## Available Tools")


def test_settings_replace_the_configured_sampling_in_place():
    request = simple(settings=Settings(temperature=0.2, max_tokens=99, top_k=7, min_p=0.1))

    options = render(request).body["options"]

    assert options == {"temperature": 0.2, "num_predict": 99, "top_k": 7, "min_p": 0.1}

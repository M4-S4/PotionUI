import copy
import dataclasses
import json
from typing import Any, Callable, Dict, List

import pytest

from src.features.llm.harness import codec
from src.features.llm.harness.adapter import RenderRequest, Settings, ToolChoice, ToolSpec
from src.features.llm.harness.adapters.anthropic_messages import AnthropicMessagesAdapter, digest, target_origin
from src.features.llm.harness.assembler import StepAssembler
from src.features.llm.harness.errors import HarnessError, OpaqueViolation
from src.features.llm.harness.messages import (
    Message,
    ReasoningPart,
    TextPart,
    ToolResultPart,
    message,
)
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.repository import LLMConfig
from tests.features.llm.harness.anthropic_helpers import (
    chunks_of,
    make_profile,
    read_json,
    step_dirs,
    stored_history,
)
from tests.fixtures.llm_harness import decode_chunks

SCENARIOS = ["thinking_parallel_tools", "approval_pause", "in_band_error"]
ADAPTER = AnthropicMessagesAdapter()


def scrubbed(wire) -> Dict[str, Any]:
    headers = {name: "[scrubbed]" if name.lower() == "x-api-key" else value for name, value in wire.headers.items()}
    return {"method": wire.method, "url": wire.url, "headers": headers, "body": wire.body}


def run_scenario(name: str) -> None:
    history: List[Message] = []
    carried: Dict[str, Any] = {}
    for directory in step_dirs(name):
        meta = read_json(directory / "meta.json")
        step_input = read_json(directory / "input.json")
        carried = {**carried, **{key: value for key, value in step_input.items() if key != "messages"}}
        history += codec.from_json(List[Message], step_input["messages"])
        profile = CapabilityProfile(adapter=meta["adapter"], model=meta["model"]).layered(
            [ProfileLayer(ProfileSource.OVERRIDE, meta["profile_overrides"])]
        )
        config = LLMConfig(**carried["config"])
        request = RenderRequest(
            config=config,
            system=carried["system"],
            messages=tuple(history),
            tools=tuple(codec.from_json(List[ToolSpec], carried["tools"])),
            settings=codec.from_json(Settings, carried["settings"]),
        )
        golden = read_json(directory / "request.json")
        wire = ADAPTER.render(request, profile)
        assert scrubbed(wire) == golden, directory
        assert wire.content() == json.dumps(golden["body"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        events = read_json(directory / "events.json")
        decoder = ADAPTER.decoder(request, profile)
        if "error" in events:
            expected = HarnessError.from_json(events["error"])
            with pytest.raises(type(expected)) as caught:
                decode_chunks(decoder, chunks_of(directory))
            assert caught.value.to_json() == expected.to_json()
            continue
        decoded = decode_chunks(decoder, chunks_of(directory))
        assert codec.to_json(decoded) == events, directory
        step = read_json(directory / "step.json")
        stored = codec.from_json(Message, step["message"])
        assert stored.origin == target_origin(config)
        assembler = StepAssembler(stored.id, stored.origin)
        assembler.feed_all(decoded)
        assert {
            "message": codec.to_json(assembler.message()),
            "usage": codec.to_json(assembler.usage),
            "finish": codec.to_json(assembler.finish),
        } == step, directory
        history.append(stored)


@pytest.mark.parametrize("name", SCENARIOS)
def test_cassette_scenario_replays_through_the_adapter(name):
    run_scenario(name)


@pytest.mark.parametrize("name", SCENARIOS)
def test_hand_built_cassettes_cite_the_provider_docs(name):
    for directory in step_dirs(name):
        meta = read_json(directory / "meta.json")
        assert meta["source"] == "hand-built"
        assert meta["doc_url"].startswith("https://platform.claude.com/docs/")
        assert meta["adapter"] == "anthropic-messages" and meta["provider"] == "anthropic"


def test_cassette_chunks_cut_through_multibyte_characters():
    cut = 0
    for directory in step_dirs("thinking_parallel_tools"):
        for chunk in chunks_of(directory):
            try:
                chunk.decode("utf-8")
            except UnicodeDecodeError:
                cut += 1
    assert cut >= 2


def test_the_pause_scenario_returns_results_in_emission_order_with_the_skipped_call_marked():
    directory = step_dirs("approval_pause")[1]
    messages = read_json(directory / "request.json")["body"]["messages"]
    results = messages[-1]["content"]
    assert [block["tool_use_id"] for block in results] == ["toolu_01HandBuiltRead", "toolu_01HandBuiltDelete"]
    assert "is_error" not in results[0] and results[1]["is_error"] is True
    assert [block["type"] for block in messages[1]["content"]] == ["thinking", "text", "tool_use", "tool_use"]


def final_request(follow_up: bool = True, **overrides: Any):
    history, request = stored_history("thinking_parallel_tools")
    if follow_up:
        history = history + [message("u3", "user", TextPart("And what about Berlin?"))]
    return history, dataclasses.replace(request, messages=tuple(history), **overrides)


def render(history: List[Message], base: RenderRequest, profile: CapabilityProfile = None):
    return ADAPTER.render(dataclasses.replace(base, messages=tuple(history)), profile or make_profile())


def test_stored_assistant_messages_replay_their_blocks_verbatim_and_in_order():
    history, request = final_request()
    messages = render(history, request).body["messages"]
    replayed = [item["content"] for item in messages if item["role"] == "assistant"]
    stored = [item.raw["content"] for item in history if item.role.value == "assistant"]
    assert replayed == stored
    assert [block["type"] for block in replayed[0]] == ["thinking", "redacted_thinking", "text", "tool_use", "tool_use"]
    assert replayed[0][0]["signature"] == "ErUBCkYIARgCIkB-hand-built-signature-1"
    assert replayed[2][0] == {"type": "thinking", "thinking": "", "signature": "ErUBCkYIARgCIkB-hand-built-signature-3"}


def test_replay_survives_a_trip_through_the_codec():
    history, request = final_request()
    revived = codec.loads(List[Message], codec.dumps(history))
    assert render(revived, request).content() == render(history, request).content()


def test_switching_models_drops_reasoning_and_switching_back_restores_it():
    history, request = final_request()
    original = render(history, request).content()
    switched = ADAPTER.render(dataclasses.replace(request, config=request.config.model_copy(update={"model": "claude-opus-5-5"})), make_profile())
    blocks = [block["type"] for item in switched.body["messages"] for block in item["content"]]
    assert "thinking" not in blocks and "redacted_thinking" not in blocks and "tool_use" in blocks
    assert render(history, request).content() == original


def test_rotating_the_api_key_drops_reasoning_instead_of_replaying_it():
    history, request = final_request()
    rotated = dataclasses.replace(request, config=request.config.model_copy(update={"api_key": "another-key"}))
    blocks = [block["type"] for item in ADAPTER.render(rotated, make_profile()).body["messages"] for block in item["content"]]
    assert "thinking" not in blocks and "redacted_thinking" not in blocks


def test_another_provider_type_with_the_same_model_and_key_does_not_replay_reasoning():
    history, request = final_request()
    renamed = dataclasses.replace(request, config=request.config.model_copy(update={"type": "anthropic-proxy"}))
    blocks = [block["type"] for item in ADAPTER.render(renamed, make_profile()).body["messages"] for block in item["content"]]
    assert "thinking" not in blocks and "redacted_thinking" not in blocks


def test_appending_messages_and_changing_effort_or_tool_choice_keep_the_replay_valid():
    history, request = final_request()
    more = history + [
        message("a4", "assistant", TextPart("Berlin is rainy."), origin=history[-2].origin),
        message("u4", "user", TextPart("Thanks.")),
    ]
    assert render(more, dataclasses.replace(request, settings=Settings(max_tokens=100, reasoning="high", tool_choice=ToolChoice("none"))))


def with_raw(source: Message, edit: Callable[[List[Dict[str, Any]]], None], reseal: bool) -> Message:
    raw = copy.deepcopy(source.raw)
    edit(raw["content"])
    if reseal:
        raw["sha256"] = digest(raw["content"])
    return dataclasses.replace(source, raw=raw)


def with_parts(source: Message, edit: Callable[[List[Any]], None]) -> Message:
    parts = list(source.parts)
    edit(parts)
    return dataclasses.replace(source, parts=tuple(parts))


def replaced(history: List[Message], index: int, new: Message) -> List[Message]:
    return history[:index] + [new] + history[index + 1:]


def drop_signature(content):
    del content[0]["signature"]


def blank_signature(content):
    content[0]["signature"] = ""


def drop_redacted(content):
    del content[1]


def swap_first_two(content):
    content[0], content[1] = content[1], content[0]


def edit_args(content):
    content[3]["input"] = {"city": "Rome"}


def edit_thinking(content):
    content[0]["thinking"] = "Something else entirely."


def drop_thinking(content):
    del content[0]


def add_field(content):
    content[3]["caller"] = {"type": "direct"}


def add_block(content):
    content.append({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}})


A1, T1, A2, U2, A3 = 1, 2, 3, 5, 6


def edited_text(source: Message, text: str) -> Message:
    return dataclasses.replace(source, parts=(TextPart(text),))


MUTATIONS: Dict[str, Callable[[List[Message]], List[Message]]] = {
    "drop_signature_in_raw": lambda h: replaced(h, A1, with_raw(h[A1], drop_signature, reseal=False)),
    "drop_signature_in_raw_resealed": lambda h: replaced(h, A1, with_raw(h[A1], drop_signature, reseal=True)),
    "blank_signature_in_raw_resealed": lambda h: replaced(h, A1, with_raw(h[A1], blank_signature, reseal=True)),
    "drop_signature_in_part": lambda h: replaced(h, A1, with_parts(h[A1], lambda p: p.__setitem__(0, ReasoningPart(summary=p[0].summary)))),
    "drop_signature_of_the_omitted_block": lambda h: replaced(h, A3, with_parts(h[A3], lambda p: p.__setitem__(0, ReasoningPart()))),
    "drop_redacted_block_in_raw": lambda h: replaced(h, A1, with_raw(h[A1], drop_redacted, reseal=False)),
    "drop_redacted_block_in_raw_resealed": lambda h: replaced(h, A1, with_raw(h[A1], drop_redacted, reseal=True)),
    "drop_redacted_part": lambda h: replaced(h, A1, with_parts(h[A1], lambda p: p.pop(1))),
    "drop_the_last_part": lambda h: replaced(h, A1, with_parts(h[A1], lambda p: p.pop())),
    "append_a_part": lambda h: replaced(h, A1, with_parts(h[A1], lambda p: p.append(TextPart("extra")))),
    "add_a_field_to_a_block_in_raw": lambda h: replaced(h, A1, with_raw(h[A1], add_field, reseal=False)),
    "add_an_unknown_block_in_raw": lambda h: replaced(h, A1, with_raw(h[A1], add_block, reseal=False)),
    "reorder_blocks_in_raw": lambda h: replaced(h, A1, with_raw(h[A1], swap_first_two, reseal=False)),
    "reorder_blocks_in_raw_resealed": lambda h: replaced(h, A1, with_raw(h[A1], swap_first_two, reseal=True)),
    "reorder_parts": lambda h: replaced(h, A1, with_parts(h[A1], lambda p: p.insert(0, p.pop(1)))),
    "reorder_tool_calls_in_parts": lambda h: replaced(h, A1, with_parts(h[A1], lambda p: p.insert(3, p.pop(4)))),
    "edit_tool_args_in_raw": lambda h: replaced(h, A1, with_raw(h[A1], edit_args, reseal=False)),
    "edit_tool_args_in_raw_resealed": lambda h: replaced(h, A1, with_raw(h[A1], edit_args, reseal=True)),
    "edit_tool_args_in_part": lambda h: replaced(
        h, A1, with_parts(h[A1], lambda p: p.__setitem__(3, dataclasses.replace(p[3], args={"city": "Rome"})))
    ),
    "edit_thinking_text_in_raw_resealed": lambda h: replaced(h, A1, with_raw(h[A1], edit_thinking, reseal=True)),
    "edit_earlier_tool_result": lambda h: replaced(
        h, T1, message("t1", "tool", ToolResultPart("toolu_01HandBuiltParis", "get_weather", (TextPart("sunny"),)), *h[T1].parts[1:])
    ),
    "flag_earlier_tool_result_as_error": lambda h: replaced(
        h, T1, message("t1", "tool", dataclasses.replace(h[T1].parts[0], is_error=True), *h[T1].parts[1:])
    ),
    "edit_earlier_user_message": lambda h: replaced(h, 0, edited_text(h[0], "Something else.")),
    "edit_earlier_context_note": lambda h: replaced(h, U2, message("u2", "user", TextPart("Wrap up now."))),
    "strip_reasoning_from_an_earlier_message_consistently": lambda h: replaced(
        h, A1, with_parts(with_raw(h[A1], drop_thinking, reseal=True), lambda p: p.pop(0))
    ),
    "delete_an_earlier_tool_message": lambda h: h[:T1] + h[T1 + 1:],
}


def test_the_untouched_history_renders():
    history, request = final_request()
    assert render(history, request)


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_a_tampered_history_is_refused_before_anything_is_sent(name):
    history, request = final_request()
    tampered = MUTATIONS[name](history)
    assert tampered != history
    with pytest.raises(OpaqueViolation) as caught:
        render(tampered, request)
    assert caught.value.retryable is False
    assert not isinstance(caught.value, (ConnectionError, TimeoutError))


def test_violations_name_the_message_and_the_reason():
    history, request = final_request()
    tampered = replaced(history, A1, with_raw(history[A1], drop_redacted, reseal=True))
    with pytest.raises(OpaqueViolation) as caught:
        render(tampered, request)
    assert caught.value.detail["message_id"] == "a1"
    assert "no longer match" in caught.value.detail["reason"]
    assert "a1" in caught.value.message

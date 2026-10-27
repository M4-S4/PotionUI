import copy
import dataclasses
import logging
from typing import Any, Callable, Dict, List, Tuple

import pytest

from src.features.llm.harness import codec
from src.features.llm.harness.adapter import RenderRequest, Settings, ToolChoice
from src.features.llm.harness.adapters.anthropic_messages import AnthropicMessagesAdapter, digest, wire_messages
from src.features.llm.harness.errors import OpaqueViolation
from src.features.llm.harness.messages import (
    Message,
    ReasoningPart,
    TextPart,
    ToolResultPart,
    message,
)
from src.features.llm.harness.profile import CapabilityProfile
from tests.features.llm.harness import cassettes
from tests.features.llm.harness.anthropic_helpers import (
    ADAPTER_DIR,
    chunks_of,
    make_profile,
    read_json,
    reply_to,
    scenario_path,
    step_dirs,
    stream,
    synthetic_history,
    text_block,
    thinking_block,
    tool_block,
    trip_request,
)

PLANNED = {"thinking_parallel_tools", "approval_pause", "in_band_error"}
PRESENT = sorted(path.name for path in cassettes.scenario_dirs(ADAPTER_DIR))
ADAPTER = AnthropicMessagesAdapter()


def require_hand_built(name):
    sources = {read_json(directory / "meta.json")["source"] for directory in step_dirs(name)}
    if sources != {"hand-built"}:
        pytest.skip(f"{name} has been replaced by a recording")


def test_every_planned_scenario_has_a_cassette():
    assert PLANNED <= set(PRESENT)


@pytest.mark.parametrize("name", PRESENT)
def test_cassette_scenario_replays_through_the_adapter(name):
    cassettes.run_scenario(ADAPTER, scenario_path(name))


@pytest.mark.parametrize("name", PRESENT)
def test_cassettes_declare_where_they_came_from(name):
    for directory in step_dirs(name):
        meta = read_json(directory / "meta.json")
        assert meta["source"] in {"hand-built", "recorded"}
        assert meta["adapter"] == "anthropic-messages" and meta["provider"] == "anthropic"
        if meta["source"] == "hand-built":
            assert meta["doc_url"].startswith("https://platform.claude.com/docs/")


def test_hand_built_cassette_chunks_cut_through_multibyte_characters():
    require_hand_built("thinking_parallel_tools")
    cut = 0
    for directory in step_dirs("thinking_parallel_tools"):
        for chunk in chunks_of(directory):
            try:
                chunk.decode("utf-8")
            except UnicodeDecodeError:
                cut += 1
    assert cut >= 2


def test_the_pause_scenario_returns_results_in_emission_order_with_the_skipped_call_marked():
    require_hand_built("approval_pause")
    directory = step_dirs("approval_pause")[1]
    messages = read_json(directory / "request.json")["body"]["messages"]
    results = messages[-1]["content"]
    assert [block["tool_use_id"] for block in results] == ["toolu_01HandBuiltRead", "toolu_01HandBuiltDelete"]
    assert "is_error" not in results[0] and results[1]["is_error"] is True
    assert [block["type"] for block in messages[1]["content"]] == ["thinking", "text", "tool_use", "tool_use"]


def final_request(follow_up: bool = True, **overrides: Any):
    history, request = synthetic_history()
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
    assert replayed[0][0]["signature"] == "sig-1"
    assert replayed[2][0] == {"type": "thinking", "thinking": "", "signature": "sig-3"}


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
        h, T1, message("t1", "tool", ToolResultPart("toolu_paris", "get_weather", (TextPart("sunny"),)), *h[T1].parts[1:])
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

DEGRADING: Dict[str, List[Tuple[str, int]]] = {
    "edit_earlier_tool_result": [("a2", A2), ("a3", A3)],
    "flag_earlier_tool_result_as_error": [("a2", A2), ("a3", A3)],
    "edit_earlier_user_message": [("a1", A1), ("a2", A2), ("a3", A3)],
    "edit_earlier_context_note": [("a3", A3)],
    "strip_reasoning_from_an_earlier_message_consistently": [("a2", A2), ("a3", A3)],
}
BROKEN_PAIRING = {"delete_an_earlier_tool_message"}


def test_the_untouched_history_renders_every_reply_with_its_thinking():
    history, request = final_request()
    wire = wire_messages(dataclasses.replace(request, messages=tuple(history)))
    assert wire.degraded == []
    replies = [item["content"] for item in wire.messages if item["role"] == "assistant"]
    assert [[block["type"] for block in reply][0] for reply in replies] == ["thinking", "thinking", "thinking"]


CONTENT_MUTATIONS = sorted(name for name in MUTATIONS if name not in DEGRADING and name not in BROKEN_PAIRING)


@pytest.mark.parametrize("name", CONTENT_MUTATIONS)
def test_tampered_blocks_are_refused_with_the_content_check_naming_the_reply(name):
    history, request = final_request()
    tampered = MUTATIONS[name](history)
    assert tampered != history
    with pytest.raises(OpaqueViolation) as caught:
        render(tampered, request)
    detail = caught.value.detail
    assert detail["check"] == "content"
    assert detail["index"] in (A1, A3) and tampered[detail["index"]].id == detail["message_id"]
    assert f"reply {detail['message_id']} (message {detail['index']})" in caught.value.message
    assert "content check failed" in caught.value.message


@pytest.mark.parametrize("name", sorted(BROKEN_PAIRING))
def test_a_history_with_unanswered_tool_calls_is_refused(name):
    history, request = final_request()
    tampered = MUTATIONS[name](history)
    assert tampered != history
    with pytest.raises(OpaqueViolation, match="tool_result"):
        render(tampered, request)


def clear_raw(history: List[Message], ids: List[str]) -> List[Message]:
    return [dataclasses.replace(item, raw=None) if item.id in ids else item for item in history]


@pytest.mark.parametrize("name", sorted(DEGRADING))
def test_edits_before_a_reply_send_that_reply_without_its_thinking_and_nothing_else_changes(name):
    history, request = final_request()
    tampered = MUTATIONS[name](history)
    assert tampered != history
    expected_ids = [item_id for item_id, _ in DEGRADING[name]]
    wire = wire_messages(dataclasses.replace(request, messages=tuple(tampered)))
    assert wire.degraded == DEGRADING[name]
    assert all(tampered[index].id == item_id for item_id, index in wire.degraded)
    assert render(tampered, request).content() == render(clear_raw(tampered, expected_ids), request).content()
    replies = [item.id for item in tampered if item.role.value == "assistant"]
    sent = [item["content"] for item in wire.messages if item["role"] == "assistant"]
    for reply_id, blocks in zip(replies, sent):
        kinds = [block["type"] for block in blocks]
        if reply_id in expected_ids:
            assert "thinking" not in kinds and "redacted_thinking" not in kinds
        else:
            assert blocks == next(item for item in tampered if item.id == reply_id).raw["content"]
    assert [block["id"] for blocks in sent for block in blocks if block["type"] == "tool_use"] == [
        "toolu_paris", "toolu_tokyo", "toolu_time",
    ]


def test_degraded_replies_are_logged_once_per_render(caplog):
    history, request = final_request()
    with caplog.at_level(logging.INFO):
        render(MUTATIONS["edit_earlier_user_message"](history), request)
        render(history, request)
    messages = [record.getMessage() for record in caplog.records if record.name == "root"]
    assert messages == ["[anthropic-messages] sent 3 stored repl(ies) without their thinking blocks"]


def test_a_trimmed_window_degrades_the_old_replies_and_the_chat_carries_on():
    history, _ = synthetic_history()
    history = history + [message("u3", "user", TextPart("And Berlin?"))]
    first = reply_to(history, stream(thinking_block(0, "Berlin weather.", "sig-4"), tool_block(1, "toolu_berlin", "get_weather", '{"city": "Berlin"}'), stop="tool_use"), "a4")
    history = history + [first, message("t3", "tool", ToolResultPart("toolu_berlin", "get_weather", (TextPart("rain"),)))]
    second = reply_to(history, stream(thinking_block(0, "Done.", "sig-5"), text_block(1, "Berlin is rainy.")), "a5")
    full = history + [second, message("u4", "user", TextPart("Thanks, and tomorrow?"))]
    assert wire_messages(trip_request(full)).degraded == []

    trimmed = full[7:]
    assert [item.id for item in trimmed] == ["u3", "a4", "t3", "a5", "u4"]
    wire = wire_messages(trip_request(trimmed))
    assert wire.degraded == [("a4", 1), ("a5", 3)]
    sent = [block["type"] for item in wire.messages if item["role"] == "assistant" for block in item["content"]]
    assert sent == ["tool_use", "text"]

    third = reply_to(trimmed, stream(thinking_block(0, "Tomorrow.", "sig-6"), tool_block(1, "toolu_next", "get_weather", '{"city": "Berlin"}'), stop="tool_use"), "a6")
    carried = trimmed + [third, message("t4", "tool", ToolResultPart("toolu_next", "get_weather", (TextPart("sun"),)))]
    again = wire_messages(trip_request(carried))
    assert again.degraded == [("a4", 1), ("a5", 3)]
    last = [item for item in again.messages if item["role"] == "assistant"][-1]["content"]
    assert [block["type"] for block in last] == ["thinking", "tool_use"] and last[0]["signature"] == "sig-6"


def test_a_blank_signature_reports_which_block_lost_it():
    history, request = final_request()
    with pytest.raises(OpaqueViolation) as caught:
        render(MUTATIONS["blank_signature_in_raw_resealed"](history), request)
    assert (caught.value.detail["message_id"], caught.value.detail["index"], caught.value.detail["block"]) == ("a1", A1, 0)


def test_a_lost_signature_names_the_block_position():
    history, request = final_request()

    def blank_second_block(content):
        swap_first_two(content)
        content[1]["signature"] = ""

    with pytest.raises(OpaqueViolation) as caught:
        render(replaced(history, A1, with_raw(history[A1], blank_second_block, reseal=True)), request)
    assert caught.value.detail["block"] == 1


def test_the_cache_ttl_follows_the_settings():
    history, request = final_request()
    long_lived = dataclasses.replace(request, settings=Settings(max_tokens=100, cache_ttl="1h"))
    default = dataclasses.replace(request, settings=Settings(max_tokens=100, cache_ttl="5m"))
    assert ADAPTER.render(long_lived, make_profile()).body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert ADAPTER.render(default, make_profile()).body["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in ADAPTER.render(long_lived, make_profile(cache="none")).body

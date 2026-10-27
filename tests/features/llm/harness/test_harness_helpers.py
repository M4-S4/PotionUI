import pytest

from src.features.llm.harness import finish, tool_ids
from src.features.llm.harness.events import FinishReason
from src.features.llm.harness.messages import Message, Origin, Role, TextPart, ToolCallPart
from src.features.llm.harness.profile import ToolIdRule
from src.features.llm.harness.tool_args import arguments_object, arguments_text, object_from_wire

ORIGIN = Origin(adapter="a", provider="a", model="m")


@pytest.mark.parametrize(
    "raw, reason",
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("tool_calls", FinishReason.TOOL_CALLS),
        ("function_call", FinishReason.TOOL_CALLS),
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("novel", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
    ],
)
def test_openai_compatible_finish_reasons(raw, reason):
    assert finish.from_openai_compat(raw) is reason


@pytest.mark.parametrize(
    "raw, reason",
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.LENGTH),
        ("load", FinishReason.UNKNOWN),
        ("tool_calls", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
    ],
)
def test_ollama_finish_reasons(raw, reason):
    assert finish.from_ollama(raw) is reason


@pytest.mark.parametrize(
    "reason, has_calls, expected",
    [
        (FinishReason.STOP, True, FinishReason.TOOL_CALLS),
        (FinishReason.UNKNOWN, True, FinishReason.TOOL_CALLS),
        (FinishReason.LENGTH, True, FinishReason.LENGTH),
        (FinishReason.CONTENT_FILTER, True, FinishReason.CONTENT_FILTER),
        (FinishReason.STOP, False, FinishReason.STOP),
        (FinishReason.UNKNOWN, False, FinishReason.UNKNOWN),
    ],
)
def test_tool_calls_upgrade_only_a_stop_or_an_unknown_reason(reason, has_calls, expected):
    assert finish.with_tool_calls(reason, has_calls) is expected


def call(args, raw=None):
    return ToolCallPart(call_id="c", name="t", args=args, args_raw=raw)


def test_arguments_text_uses_the_raw_string_only_when_it_means_the_same_as_the_parsed_ones():
    assert arguments_text(call({"a": 1}, '{"a":1}')) == '{"a":1}'
    assert arguments_text(call({"a": 1}, '{"a":2}')) == '{"a": 1}'
    assert arguments_text(call({"a": 1}, "not json")) == '{"a": 1}'
    assert arguments_text(call({"a": 1})) == '{"a": 1}'


def test_arguments_text_never_sends_something_a_strict_server_would_refuse():
    assert arguments_text(call(None, '{"a":')) == "{}"
    assert arguments_text(call({}, "  ")) == "{}"
    assert arguments_text(call({})) == "{}"


def test_arguments_object_is_always_an_object():
    assert arguments_object(call({"a": 1})) == {"a": 1}
    assert arguments_object(call(None)) == {}


@pytest.mark.parametrize(
    "value, expected",
    [
        ({"a": 1}, {"a": 1}),
        ('{"a": 1}', {"a": 1}),
        ("", {}),
        (None, {}),
        ("{oops", None),
        ("[1]", None),
        (5, None),
    ],
)
def test_object_from_wire(value, expected):
    assert object_from_wire(value) == expected


def test_minted_ids_are_deterministic_unique_per_position_and_seed():
    assert tool_ids.mint_call_id("m1", 0) == tool_ids.mint_call_id("m1", 0)
    assert tool_ids.mint_call_id("m1", 0) != tool_ids.mint_call_id("m1", 1)
    assert tool_ids.mint_call_id("m1", 0) != tool_ids.mint_call_id("m2", 0)


def test_nine_digit_ids_are_nine_digits_and_stable():
    ids = {tool_ids.nine_digit_id(f"call_{n}") for n in range(50)}

    assert all(len(i) == 9 and i.isdigit() for i in ids)
    assert len(ids) == 50
    assert tool_ids.nine_digit_id("x") == tool_ids.nine_digit_id("x")


def test_wire_ids_map_every_call_by_the_rule():
    messages = (
        Message("a", Role.ASSISTANT, (ToolCallPart(call_id="c1", name="t", args={}, wire_id="srv"), ToolCallPart(call_id="c2", name="t", args={})), origin=ORIGIN),
    )

    assert tool_ids.wire_ids(messages, ToolIdRule.PASSTHROUGH) == {"c1": "srv", "c2": "c2"}
    assert tool_ids.wire_ids(messages, ToolIdRule.MINT) == {"c1": "srv", "c2": "c2"}
    nine = tool_ids.wire_ids(messages, ToolIdRule.NINE_DIGITS)
    assert nine["c1"] == tool_ids.nine_digit_id("c1") != "srv"


def test_the_seed_is_the_last_message_id():
    messages = (Message("m1", Role.USER, (TextPart("a"),)), Message("m2", Role.USER, (TextPart("b"),)))

    assert tool_ids.seed_for(messages) == "m2"
    assert tool_ids.seed_for(()) == ""

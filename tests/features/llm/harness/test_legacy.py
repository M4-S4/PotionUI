from src.features.llm.harness.legacy import messages_from_dicts
from src.features.llm.harness.messages import LEGACY_ORIGIN, Origin, ReasoningPart, Role, TextPart, ToolCallPart


def test_dict_history_converts_into_messages_with_one_tool_message_per_batch():
    entries = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "find cats"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "need search",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": {"q": "cats"}}},
                {"id": "c2", "type": "function", "function": {"name": "count", "arguments": '{"n": 2}'}},
            ],
        },
        {"role": "tool", "content": "3 hits", "tool_call_id": "c1", "name": "search"},
        {"role": "tool", "content": "2", "tool_call_id": "c2", "name": "count"},
        {"role": "assistant", "content": "Found 3."},
    ]

    messages, sources = messages_from_dicts(entries)

    assert [message.role for message in messages] == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
    assert [len(group) for group in sources] == [1, 1, 1, 2, 1]
    calls = messages[2].tool_calls()
    assert calls[0] == ToolCallPart(call_id="c1", name="search", args={"q": "cats"}, wire_id="c1")
    assert calls[1] == ToolCallPart(call_id="c2", name="count", args={"n": 2}, args_raw='{"n": 2}', wire_id="c2")
    assert messages[2].reasoning() == (ReasoningPart(text="need search"),)
    assert messages[2].text() == ""
    assert [part.call_id for part in messages[3].tool_results()] == ["c1", "c2"]
    assert messages[3].tool_results()[1].text() == "2"
    assert messages[4].parts == (TextPart(text="Found 3."),)
    assert messages[4].origin == LEGACY_ORIGIN


def test_a_given_origin_is_applied_and_ids_are_minted_when_missing():
    origin = Origin(adapter="openai-chat", provider="openai", model="m")

    messages, _ = messages_from_dicts(
        [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x", "arguments": "not json"}}]}],
        origin,
    )

    call = messages[0].tool_calls()[0]
    assert messages[0].origin == origin
    assert call.call_id.startswith("call_") and call.wire_id is None
    assert call.args == {} and call.args_raw == "not json"

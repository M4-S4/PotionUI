from dataclasses import replace
from typing import List, Optional

from src.features.llm.harness.adapter import RenderRequest, Settings, ToolChoice, ToolSpec, WireRequest
from src.features.llm.harness.build import user_message
from src.features.llm.harness.cache import (
    Divergence,
    Segment,
    append_only,
    first_divergence,
    request_segments,
    wire_segments,
)
from src.features.llm.harness.messages import Message, Origin, Role, TextPart
from tests.fixtures.llm_engine import scripted_config

ORIGIN = Origin(adapter="scripted", provider="scripted", model="scripted-model")


def tool(name: str) -> ToolSpec:
    return ToolSpec(name=name, description=f"{name} tool", parameters={"type": "object"})


def assistant(text: str) -> Message:
    return Message(id=f"a-{text}", role=Role.ASSISTANT, parts=(TextPart(text=text),), origin=ORIGIN)


def request(
    messages: Optional[List[Message]] = None,
    tools: tuple = (tool("echo"), tool("find")),
    system: str = "SYS",
    settings: Optional[Settings] = None,
) -> RenderRequest:
    return RenderRequest(
        config=scripted_config(),
        system=system,
        messages=tuple(messages if messages is not None else [user_message("one", message_id="u1"), assistant("two")]),
        tools=tuple(tools),
        settings=settings or Settings(),
    )


def divergence(before: RenderRequest, after: RenderRequest) -> Optional[Divergence]:
    return first_divergence(request_segments(before), request_segments(after))


def test_the_same_request_twice_is_stable():
    assert divergence(request(), request()) is None


def test_appending_messages_keeps_the_prefix_stable():
    before = request()
    after = request([*before.messages, user_message("three", message_id="u2"), assistant("four")])

    assert divergence(before, after) is None


def test_an_edit_to_an_earlier_message_is_reported_where_it_happens():
    before = request()
    edited = request([user_message("one!", message_id="u1"), assistant("two")])

    found = divergence(before, edited)

    assert found.section == "messages[0]"
    head = b"".join(segment.data + b"\n" for segment in request_segments(before) if not segment.section.startswith("messages"))
    assert found.offset > len(head)
    assert found.offset == len(head) + common_length(request_segments(before)[-2].data, request_segments(edited)[-2].data)


def test_appending_text_to_the_last_message_is_a_divergence_at_the_end_of_the_old_text():
    before = request()
    grown = request([user_message("one", message_id="u1"), assistant("two and more")])

    found = divergence(before, grown)

    assert found.section == "messages[1]"


def test_a_new_tool_in_the_middle_of_the_list_diverges_at_that_tool():
    before = request(tools=(tool("echo"), tool("find")))
    after = request(tools=(tool("echo"), tool("late"), tool("find")))

    assert divergence(before, after).section == "tools[1]"


def test_a_tool_appended_to_the_end_breaks_the_cache_at_the_system_prompt_and_nowhere_earlier():
    before = request(tools=(tool("echo"), tool("find")))
    after = request(tools=(tool("echo"), tool("find"), tool("late")))

    found = divergence(before, after)

    assert found.section == "system"
    tools_end = sum(len(s.data) + 1 for s in request_segments(before) if s.section == "params" or s.section.startswith("tools"))
    assert found.offset == tools_end


def test_a_changed_system_prompt_diverges_at_the_system_segment():
    assert divergence(request(), request(system="SYS2")).section == "system"


def test_changing_reasoning_or_tool_choice_between_steps_diverges_at_the_params():
    base = request()

    assert divergence(base, request(settings=Settings(reasoning=True))).section == "params"
    assert divergence(base, request(settings=Settings(tool_choice=ToolChoice("none")))).section == "params"
    assert divergence(base, request(settings=Settings(temperature=0.2, max_tokens=64, keep_model_loaded=True))) is None


def test_a_request_that_drops_an_earlier_message_diverges_at_the_first_missing_segment():
    before = request()
    shorter = request([before.messages[0]])

    found = divergence(before, shorter)

    assert found.section == "messages[1]"
    assert found.offset == sum(len(s.data) + 1 for s in request_segments(shorter))


def test_opaque_data_kept_on_a_message_is_part_of_the_prefix():
    kept = replace(assistant("two"), raw={"signature": "abc"})
    dropped = assistant("two")
    before = request([user_message("one", message_id="u1"), kept])
    after = request([user_message("one", message_id="u1"), dropped])

    assert divergence(before, after).section == "messages[1]"


def test_first_divergence_counts_bytes_inside_a_changed_segment():
    before = [Segment("a", b"abcdef"), Segment("b", b"xyz")]
    after = [Segment("a", b"abcdef"), Segment("b", b"xyq")]

    assert first_divergence(before, after) == Divergence(offset=len(b"abcdef\n") + 2, section="b")


def test_append_only_keeps_the_sent_order_and_adds_new_tools_at_the_end():
    sent = (tool("echo"), tool("use_tools"))
    now = (tool("echo"), tool("late"), tool("use_tools"))

    merged = append_only(sent, now)

    assert [spec.name for spec in merged] == ["echo", "use_tools", "late"]


def test_append_only_keeps_the_schema_that_was_already_sent():
    sent = (ToolSpec(name="use_tools", description="groups: a, b", parameters={}),)
    now = (ToolSpec(name="use_tools", description="groups: b", parameters={}),)

    assert append_only(sent, now) == sent


def test_append_only_never_drops_a_tool_that_went_out_before():
    assert append_only((tool("echo"), tool("use_tools")), (tool("echo"),)) == (tool("echo"), tool("use_tools"))


def test_append_only_from_nothing_is_the_current_list():
    assert append_only((), (tool("b"), tool("a"))) == (tool("b"), tool("a"))


def wire(body: dict) -> WireRequest:
    return WireRequest(method="POST", url="https://x", headers={}, body=body)


def test_wire_segments_of_a_chat_body_put_tools_before_the_messages_and_ignore_max_tokens():
    first = wire({
        "model": "m", "max_tokens": 100, "stream": True,
        "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "echo"}}],
    })
    second = wire({
        "model": "m", "max_tokens": 90, "stream": True,
        "messages": [
            {"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ],
        "tools": [{"type": "function", "function": {"name": "echo"}}],
    })

    assert [s.section for s in wire_segments(first)] == ["params", "tools[0]", "messages[0]", "messages[1]"]
    assert first_divergence(wire_segments(first), wire_segments(second)) is None


def test_wire_segments_treat_a_top_level_system_and_an_input_list_like_the_other_wires():
    anthropic = wire({"model": "m", "system": "SYS", "messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "t"}]})
    responses = wire({"model": "m", "instructions": "SYS", "input": [{"role": "user", "content": "hi"}], "prompt_cache_key": "s"})

    assert [s.section for s in wire_segments(anthropic)] == ["params", "tools[0]", "system", "messages[0]"]
    assert [s.section for s in wire_segments(responses)] == ["params", "instructions", "input[0]"]


def test_wire_segments_report_a_changed_cache_relevant_parameter():
    before = wire({"model": "m", "thinking": {"type": "enabled", "budget_tokens": 1024}, "messages": []})
    after = wire({"model": "m", "thinking": {"type": "enabled", "budget_tokens": 2048}, "messages": []})

    assert first_divergence(wire_segments(before), wire_segments(after)).section == "params"


def test_wire_segments_do_not_depend_on_key_order():
    one = wire({"model": "m", "temperature": 0.1, "messages": [{"role": "user", "content": "hi"}]})
    two = wire({"messages": [{"content": "hi", "role": "user"}], "temperature": 0.1, "model": "m"})

    assert wire_segments(one) == wire_segments(two)


def common_length(left: bytes, right: bytes) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count

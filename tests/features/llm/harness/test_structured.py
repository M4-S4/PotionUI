from dataclasses import replace

import pytest

from src.features.llm.harness.adapter import Settings, ToolChoice
from src.features.llm.harness.messages import Role, TextPart, message
from src.features.llm.harness.profile import ReasoningKind, StructuredOutputMode
from src.features.llm.harness.structured import (
    STRUCTURED_TOOL_NAME,
    StructuredOutputFailed,
    extract_json,
    plan_structured,
    run_structured,
    validate,
)
from src.features.llm.harness.turn_state import NoticeKind
from tests.features.llm.harness.stepping import scripted_complete
from tests.fixtures.llm_harness import ScriptedAdapter, scripted_profile, text_step, tool_step

TITLE = {
    "type": "object",
    "properties": {"title": {"type": "string", "minLength": 1}},
    "required": ["title"],
    "additionalProperties": False,
}
NUMBERS = {"type": "array", "items": {"type": "integer"}}


def profile(mode: StructuredOutputMode, **changes):
    return replace(scripted_profile(), structured_output=mode, **changes)


def settings(schema=TITLE, **changes) -> Settings:
    return Settings(response_schema=schema, reasoning=False, **changes)


def user(text="name this chat"):
    return message("u1", Role.USER, TextPart(text))


def test_a_native_profile_hands_the_schema_to_the_adapter():
    plan = plan_structured("be brief", settings(), profile(StructuredOutputMode.NATIVE))
    assert plan.mode is StructuredOutputMode.NATIVE
    assert plan.settings.response_schema == TITLE
    assert (plan.system, plan.tools, plan.notices) == ("be brief", (), ())


def test_a_tool_profile_forces_a_single_result_tool():
    plan = plan_structured(
        "be brief", settings(), profile(StructuredOutputMode.TOOL, tool_choice=frozenset({"auto", "named"}), strict_tools=True),
    )
    assert plan.mode is StructuredOutputMode.TOOL
    assert plan.settings.response_schema is None
    assert plan.settings.tool_choice == ToolChoice("named", STRUCTURED_TOOL_NAME)
    (tool,) = plan.tools
    assert tool.name == STRUCTURED_TOOL_NAME
    assert tool.parameters == TITLE
    assert tool.strict is True
    assert plan.system == "be brief"


def test_a_prompted_profile_puts_the_schema_in_the_system_prompt():
    plan = plan_structured("be brief", settings(), profile(StructuredOutputMode.PROMPTED))
    assert plan.mode is StructuredOutputMode.PROMPTED
    assert plan.settings.response_schema is None
    assert plan.system.startswith("be brief\n\n")
    assert '"required":["title"]' in plan.system
    assert plan.tools == ()


def test_a_prompted_plan_without_a_system_prompt_is_just_the_instruction():
    assert not plan_structured("", settings(), profile(StructuredOutputMode.PROMPTED)).system.startswith("\n")


def test_a_tool_profile_that_cannot_force_a_tool_falls_back_to_prompting_with_a_notice():
    plan = plan_structured("", settings(), profile(StructuredOutputMode.TOOL, tool_choice=frozenset({"auto"})))
    assert plan.mode is StructuredOutputMode.PROMPTED
    assert plan.tools == ()
    (notice,) = plan.notices
    assert notice.kind is NoticeKind.DEGRADED_RETRY
    assert notice.detail["reason"] == "structured_output_fallback"
    assert "Schema:" in plan.system


def test_forcing_the_result_tool_is_not_attempted_while_the_model_reasons():
    reasoning = profile(
        StructuredOutputMode.TOOL,
        reasoning=ReasoningKind.ALWAYS_ON,
        tool_choice=frozenset({"auto", "named"}),
        tool_choice_with_reasoning=frozenset({"auto"}),
    )
    assert plan_structured("", settings(), reasoning).mode is StructuredOutputMode.PROMPTED


def test_a_profile_without_tools_cannot_use_the_tool_mode():
    plan = plan_structured("", settings(), profile(StructuredOutputMode.TOOL, tools=False, tool_choice=frozenset({"named"})))
    assert plan.mode is StructuredOutputMode.PROMPTED


def test_a_schema_that_is_not_an_object_is_wrapped_for_native_and_tool_modes():
    native = plan_structured("", settings(NUMBERS), profile(StructuredOutputMode.NATIVE))
    assert native.wrapped is True
    assert native.settings.response_schema["properties"]["value"] == NUMBERS
    assert native.settings.response_schema["required"] == ["value"]
    prompted = plan_structured("", settings(NUMBERS), profile(StructuredOutputMode.PROMPTED))
    assert prompted.wrapped is False


def test_definitions_move_to_the_wrapper_root_so_references_keep_resolving():
    schema = {"type": "array", "items": {"$ref": "#/$defs/n"}, "$defs": {"n": {"type": "integer"}}}
    wire = plan_structured("", settings(schema), profile(StructuredOutputMode.NATIVE)).settings.response_schema
    assert "$defs" in wire and "$defs" not in wire["properties"]["value"]


def test_a_missing_or_malformed_schema_is_a_programming_error():
    with pytest.raises(ValueError):
        plan_structured("", Settings(), profile(StructuredOutputMode.NATIVE))
    with pytest.raises(ValueError):
        plan_structured("", Settings(response_schema="{}"), profile(StructuredOutputMode.NATIVE))


async def test_a_valid_native_reply_is_parsed_without_a_repair():
    adapter = ScriptedAdapter(text_step('{"title": "Cats"}'))
    result = await run_structured(
        scripted_complete(adapter, profile(StructuredOutputMode.NATIVE)),
        system="s", messages=[user()], settings=settings(), profile=profile(StructuredOutputMode.NATIVE),
    )
    assert result.value == {"title": "Cats"}
    assert result.repaired is False
    assert adapter.requests[0].settings.response_schema == TITLE
    assert adapter.exhausted


async def test_a_reply_wrapped_in_a_code_fence_and_prose_is_still_accepted():
    adapter = ScriptedAdapter(text_step('Sure!\n```json\n{"title": "Cats"}\n```\nAnything else?'))
    wanted = profile(StructuredOutputMode.PROMPTED)
    result = await run_structured(
        scripted_complete(adapter, wanted), system="", messages=[user()], settings=settings(), profile=wanted,
    )
    assert result.value == {"title": "Cats"}
    assert adapter.requests[0].settings.response_schema is None
    assert "Schema:" in adapter.requests[0].system


async def test_a_wrapped_array_comes_back_unwrapped():
    adapter = ScriptedAdapter(text_step('{"value": [1, 2, 3]}'))
    wanted = profile(StructuredOutputMode.NATIVE)
    result = await run_structured(
        scripted_complete(adapter, wanted), system="", messages=[user()], settings=settings(NUMBERS), profile=wanted,
    )
    assert result.value == [1, 2, 3]


async def test_an_invalid_reply_gets_exactly_one_repair_attempt():
    adapter = ScriptedAdapter(text_step('{"title": 5}'), text_step('{"title": "Cats"}'))
    wanted = profile(StructuredOutputMode.NATIVE)
    result = await run_structured(
        scripted_complete(adapter, wanted), system="", messages=[user()], settings=settings(), profile=wanted,
    )
    assert result.value == {"title": "Cats"}
    assert result.repaired is True
    assert result.notices[-1].detail["reason"] == "structured_output_repair"
    sent = adapter.requests[1].messages
    assert [item.role for item in sent] == [Role.USER, Role.ASSISTANT, Role.USER]
    assert "$.title: expected string, got integer" in sent[-1].text()
    assert sent[1].text() == '{"title": 5}'


async def test_a_reply_that_stays_invalid_after_the_repair_fails():
    adapter = ScriptedAdapter(text_step("not json"), text_step("still not json"), text_step('{"title": "never asked"}'))
    wanted = profile(StructuredOutputMode.NATIVE)
    with pytest.raises(StructuredOutputFailed) as raised:
        await run_structured(
            scripted_complete(adapter, wanted), system="", messages=[user()], settings=settings(), profile=wanted,
        )
    assert "not valid JSON" in str(raised.value)
    assert raised.value.detail["mode"] == "native"
    assert len(adapter.requests) == 2


async def test_the_tool_mode_reads_the_result_from_the_forced_call():
    adapter = ScriptedAdapter(tool_step({"id": "t1", "name": STRUCTURED_TOOL_NAME, "args": {"title": "Cats"}}))
    wanted = profile(StructuredOutputMode.TOOL, tool_choice=frozenset({"auto", "named"}))
    result = await run_structured(
        scripted_complete(adapter, wanted), system="", messages=[user()], settings=settings(), profile=wanted,
    )
    assert result.value == {"title": "Cats"}
    request = adapter.requests[0]
    assert request.settings.tool_choice == ToolChoice("named", STRUCTURED_TOOL_NAME)
    assert [tool.name for tool in request.tools] == [STRUCTURED_TOOL_NAME]


async def test_the_tool_mode_repairs_with_a_tool_result_so_the_history_stays_valid():
    adapter = ScriptedAdapter(
        tool_step({"id": "t1", "name": STRUCTURED_TOOL_NAME, "args": {}}),
        tool_step({"id": "t2", "name": STRUCTURED_TOOL_NAME, "args": {"title": "Cats"}}),
    )
    wanted = profile(StructuredOutputMode.TOOL, tool_choice=frozenset({"auto", "named"}))
    result = await run_structured(
        scripted_complete(adapter, wanted), system="", messages=[user()], settings=settings(), profile=wanted,
    )
    assert result.repaired is True and result.value == {"title": "Cats"}
    sent = adapter.requests[1].messages
    assert [item.role for item in sent] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    (feedback,) = sent[-1].tool_results()
    assert feedback.call_id == "t1" and feedback.is_error
    assert "missing required property 'title'" in feedback.text()


async def test_the_tool_mode_repairs_a_plain_text_answer_with_a_user_message():
    adapter = ScriptedAdapter(text_step("Cats"), tool_step({"id": "t1", "name": STRUCTURED_TOOL_NAME, "args": {"title": "Cats"}}))
    wanted = profile(StructuredOutputMode.TOOL, tool_choice=frozenset({"auto", "named"}))
    result = await run_structured(
        scripted_complete(adapter, wanted), system="", messages=[user()], settings=settings(), profile=wanted,
    )
    assert result.repaired is True
    assert adapter.requests[1].messages[-1].role is Role.USER
    assert "tool was not called" in adapter.requests[1].messages[-1].text()


def test_extracting_json_handles_fences_prose_and_failures():
    assert extract_json('{"a": 1}') == ({"a": 1}, None)
    assert extract_json('```json\n[1, 2]\n```') == ([1, 2], None)
    assert extract_json('Here you go: {"a": {"b": 2}} hope it helps') == ({"a": {"b": 2}}, None)
    assert extract_json('Pick {one}:\n```json\n{"a": 1}\n```') == ({"a": 1}, None)
    assert extract_json("   ") == (None, "the reply was empty")
    assert extract_json("no json at all") == (None, "the reply was not valid JSON")


@pytest.mark.parametrize(
    "value, schema, expected",
    [
        ("x", {"type": "string"}, []),
        (1, {"type": "string"}, ["$: expected string, got integer"]),
        (True, {"type": "integer"}, ["$: expected integer, got boolean"]),
        (1.5, {"type": ["integer", "null"]}, ["$: expected integer or null, got number"]),
        (None, {"type": ["integer", "null"]}, []),
        ("c", {"enum": ["a", "b"]}, ['$: must be one of ["a", "b"]']),
        (3, {"const": 4}, ["$: must equal 4"]),
        ("", {"type": "string", "minLength": 1}, ["$: must have at least 1 characters"]),
        ("abc", {"type": "string", "maxLength": 2}, ["$: must have at most 2 characters"]),
        ("abc", {"type": "string", "pattern": "^x"}, ["$: must match the pattern ^x"]),
        (0, {"type": "number", "minimum": 1}, ["$: must be at least 1"]),
        (9, {"type": "number", "maximum": 5}, ["$: must be at most 5"]),
        ([], {"type": "array", "minItems": 1}, ["$: must have at least 1 items"]),
        ([1, 2], {"type": "array", "maxItems": 1}, ["$: must have at most 1 items"]),
        ([1, "a"], NUMBERS, ["$[1]: expected integer, got string"]),
        ({"a": 1}, {"type": "object", "required": ["b"]}, ["$: missing required property 'b'"]),
        ({"a": 1}, {"type": "object", "properties": {}, "additionalProperties": False}, ["$: unexpected property 'a'"]),
        ({"a": "x"}, {"type": "object", "additionalProperties": {"type": "integer"}}, ["$.a: expected integer, got string"]),
        ({"a": {"b": 1}}, {"properties": {"a": {"properties": {"b": {"type": "string"}}}}}, ["$.a.b: expected string, got integer"]),
        (5, {"anyOf": [{"type": "string"}, {"type": "integer"}]}, []),
        (5.5, {"anyOf": [{"type": "string"}, {"type": "integer"}]}, ["$: matches none of the allowed shapes"]),
        (5, {"oneOf": [{"type": "integer"}, {"type": "number"}]}, ["$: must match exactly one of the allowed shapes"]),
        (5, {"allOf": [{"type": "integer"}, {"minimum": 9}]}, ["$: must be at least 9"]),
    ],
)
def test_the_validator_reports_what_is_wrong(value, schema, expected):
    assert validate(value, schema) == expected


def test_references_resolve_against_the_document_root():
    schema = {"type": "array", "items": {"$ref": "#/$defs/n"}, "$defs": {"n": {"type": "integer"}}}
    assert validate([1, 2], schema) == []
    assert validate([1, "x"], schema) == ["$[1]: expected integer, got string"]


def test_recursive_references_follow_the_data_and_loops_without_data_are_refused():
    tree = {"type": "object", "properties": {"kids": {"type": "array", "items": {"$ref": "#"}}}, "required": ["kids"]}
    assert validate({"kids": [{"kids": []}]}, tree) == []
    assert validate({"kids": [{}]}, tree) == ["$.kids[0]: missing required property 'kids'"]
    with pytest.raises(ValueError):
        validate(1, {"$ref": "#/a", "a": {"$ref": "#/a"}})
    with pytest.raises(ValueError):
        validate(1, {"$ref": "https://example.com/schema"})

import json

import pytest

from src.features.llm.tools.builtin.prompt_variables_tool import CALL_EXAMPLE, ManagePromptVariablesTool
from tests.fixtures.fake_tab_drafts import drafts_context

MOOD = {"op": "set", "name": "mood", "type": "choice", "options": ["noir", "sunlit"], "mode": "shuffle"}
NOTE = {"op": "set", "name": "note", "type": "text", "value": "hi"}
FLOOD = 412


def _ctx():
    return drafts_context({"form_state": {"variables": []}})


def _stored(ctx):
    return set(ctx.tab_drafts.variables(ctx.user_id, "tab-1"))


class TestJsonStringOperations:
    @pytest.mark.asyncio
    async def test_a_json_string_array_previews_every_operation(self):
        result = await ManagePromptVariablesTool().execute(_ctx(), operations=json.dumps([MOOD, NOTE]))

        assert result.success is True
        payload = json.loads(result.data)
        assert payload["change_count"] == 2
        assert [row["field_name"] for row in payload["proposed_changes"]] == ["${mood}", "${note}"]
        assert "warnings" not in payload

    @pytest.mark.asyncio
    async def test_a_json_string_array_is_applied(self):
        ctx = _ctx()

        result = await ManagePromptVariablesTool().execute_confirmed(ctx, operations=json.dumps([MOOD, NOTE]))

        assert result.success is True
        assert json.loads(result.data)["applied_count"] == 2
        assert _stored(ctx) == {"mood", "note"}

    @pytest.mark.asyncio
    async def test_a_json_string_object_is_previewed_as_one_operation(self):
        result = await ManagePromptVariablesTool().execute(_ctx(), operations=json.dumps(NOTE))

        assert result.success is True
        assert json.loads(result.data)["change_count"] == 1

    @pytest.mark.asyncio
    async def test_a_json_string_object_is_applied_as_one_operation(self):
        ctx = _ctx()

        result = await ManagePromptVariablesTool().execute_confirmed(ctx, operations=json.dumps(NOTE))

        assert result.success is True
        assert _stored(ctx) == {"note"}

    @pytest.mark.asyncio
    async def test_a_bare_object_is_applied_as_one_operation(self):
        ctx = _ctx()

        result = await ManagePromptVariablesTool().execute_confirmed(ctx, operations=NOTE)

        assert result.success is True
        assert _stored(ctx) == {"note"}

    @pytest.mark.asyncio
    async def test_unquoted_keys_in_the_string_are_read(self):
        ctx = _ctx()
        text = '[{op: "set", name: "note", type: "text", value: "hi"}]'

        result = await ManagePromptVariablesTool().execute_confirmed(ctx, operations=text)

        assert result.success is True
        assert _stored(ctx) == {"note"}

    @pytest.mark.asyncio
    async def test_over_mcp_a_json_string_array_is_applied_to_the_named_tab(self):
        ctx = drafts_context({}, chat_session=False)

        result = await ManagePromptVariablesTool().execute_confirmed(
            ctx, tab_id="tab-1", operations=json.dumps([MOOD, NOTE])
        )

        assert result.success is True
        assert _stored(ctx) == {"mood", "note"}


class TestUnreadableOperations:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ['[{"op": "set", "name": "mood"', "make the variables moodier", '"[1]"', "42"])
    async def test_text_that_is_not_an_array_gets_one_teaching_error(self, text):
        ctx = _ctx()

        for call in (ManagePromptVariablesTool().execute, ManagePromptVariablesTool().execute_confirmed):
            result = await call(ctx, operations=text)

            assert result.success is False
            assert result.error.count(CALL_EXAMPLE) == 1
            assert "Operation must be an object" not in result.error
            assert len(result.error) < 1200
        assert _stored(ctx) == set()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value, type_name", [(7, "int"), (2.5, "float"), (True, "bool")])
    async def test_a_scalar_gets_one_teaching_error_naming_its_type(self, value, type_name):
        result = await ManagePromptVariablesTool().execute(_ctx(), operations=value)

        assert result.success is False
        assert f"got {type_name}" in result.error
        assert result.error.count(CALL_EXAMPLE) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", [None, [], {}, "", "[]"])
    async def test_nothing_to_apply_is_the_plain_no_operations_error(self, value):
        ctx = _ctx()

        for call in (ManagePromptVariablesTool().execute, ManagePromptVariablesTool().execute_confirmed):
            result = await call(ctx, operations=value)

            assert result.success is False
            assert result.error.startswith("No operations provided")

    @pytest.mark.asyncio
    async def test_the_call_example_is_a_working_call_for_this_tool(self):
        body = CALL_EXAMPLE.removeprefix("<tool_call>").removesuffix("</tool_call>")
        call = json.loads(body)
        ctx = drafts_context({"form_state": {"variables": [{"name": "old_style", "type": "text", "value": "x"}]}})

        result = await ManagePromptVariablesTool().execute(ctx, **call["arguments"])

        assert call["name"] == ManagePromptVariablesTool().name
        assert result.success is True
        payload = json.loads(result.data)
        assert payload["change_count"] == 3
        assert "warnings" not in payload
        assert [row["field_name"] for row in payload["proposed_changes"]] == ["${mood}", "${palette}", "${old_style}"]


class TestRepeatedErrorsAreCollapsed:
    @pytest.mark.asyncio
    async def test_a_list_of_non_objects_is_one_message_with_the_count(self):
        result = await ManagePromptVariablesTool().execute(_ctx(), operations=["x"] * FLOOD)

        assert result.success is False
        assert result.error == f"Operation must be an object. (×{FLOOD})"

    @pytest.mark.asyncio
    async def test_the_same_holds_when_the_list_arrives_as_a_json_string(self):
        result = await ManagePromptVariablesTool().execute(_ctx(), operations=json.dumps(["x"] * FLOOD))

        assert result.success is False
        assert result.error == f"Operation must be an object. (×{FLOOD})"

    @pytest.mark.asyncio
    async def test_applying_a_list_of_non_objects_is_one_message_with_the_count(self):
        ctx = _ctx()

        result = await ManagePromptVariablesTool().execute_confirmed(ctx, operations=["x"] * FLOOD)

        assert result.success is False
        assert result.error == f"Operation must be an object. (×{FLOOD})"
        assert _stored(ctx) == set()

    @pytest.mark.asyncio
    async def test_a_single_bad_item_has_no_count(self):
        result = await ManagePromptVariablesTool().execute(_ctx(), operations=["x"])

        assert result.error == "Operation must be an object."

    @pytest.mark.asyncio
    async def test_a_mixed_list_previews_the_objects_and_warns_once_about_the_rest(self):
        result = await ManagePromptVariablesTool().execute(_ctx(), operations=[MOOD, "x", NOTE, 3, None])

        assert result.success is True
        payload = json.loads(result.data)
        assert payload["change_count"] == 2
        assert payload["warnings"] == ["Operation must be an object. (×3)"]

    @pytest.mark.asyncio
    async def test_a_mixed_list_applies_the_objects_and_reports_the_rest_once(self):
        ctx = _ctx()

        result = await ManagePromptVariablesTool().execute_confirmed(ctx, operations=[MOOD, "x", NOTE, 3, None])

        assert result.success is True
        payload = json.loads(result.data)
        assert payload["applied_count"] == 2
        assert payload["rejected_operations"] == ["Operation must be an object. (×3)"]
        assert _stored(ctx) == {"mood", "note"}

    @pytest.mark.asyncio
    async def test_other_repeated_messages_collapse_and_distinct_ones_stay_apart(self):
        operations = [
            {"op": "set", "name": " ", "type": "text"},
            {"op": "set", "name": "", "type": "text"},
            {"op": "remove", "name": "ghost"},
            {"op": "remove", "name": "phantom"},
            "x",
        ]

        result = await ManagePromptVariablesTool().execute(_ctx(), operations=operations)

        assert result.success is False
        assert result.error == (
            "Empty variable name in operation. (×2); "
            "Unknown variable 'ghost' — cannot remove.; "
            "Unknown variable 'phantom' — cannot remove.; "
            "Operation must be an object."
        )

import json

import pytest

from src.features.llm.tools.builtin.prompt_variables_tool import (
    GetPromptVariablesTool,
    ListTabDraftsTool,
    ManagePromptVariablesTool,
)
from tests.fixtures.fake_tab_drafts import drafts_context, make_manager, seed

MOOD = {"op": "set", "name": "mood", "type": "choice", "options": ["noir", "sunlit"], "mode": "shuffle"}
NOTE = {"op": "set", "name": "note", "type": "text", "value": "hi"}
BAD_AGE = {"op": "set", "name": "age", "type": "number", "min": 0, "max": 100000}


def _ctx(manager, user_id="user-1", form_state=None):
    return drafts_context({"form_state": form_state} if form_state is not None else {}, user_id=user_id, manager=manager)


class TestPreviewNeverWrites:
    @pytest.mark.asyncio
    async def test_execute_leaves_the_draft_and_the_socket_alone(self):
        manager, notifier = make_manager()
        ctx = _ctx(manager, form_state={"tab_id": "tab-1"})

        result = await ManagePromptVariablesTool().execute(ctx, operations=[MOOD])

        assert result.success
        assert manager.find("user-1", "tab-1") is None
        assert notifier.sent == []

    @pytest.mark.asyncio
    async def test_preview_diffs_against_the_stored_draft(self):
        manager, _ = make_manager()
        seed(manager, "user-1", [{"name": "mood", "type": "text", "value": "old value"}])
        ctx = _ctx(manager, form_state={"tab_id": "tab-1"})

        result = await ManagePromptVariablesTool().execute(ctx, operations=[MOOD])

        assert "old value" in json.loads(result.data)["proposed_changes"][0]["old_value"]


class TestPreviewNamesTheTab:
    @pytest.mark.asyncio
    async def test_the_pending_approval_payload_carries_the_tab_it_was_computed_for(self):
        manager, _ = make_manager()
        ctx = _ctx(manager, form_state={"tab_id": "tab-1"})

        result = await ManagePromptVariablesTool().execute(ctx, operations=[MOOD])

        assert json.loads(result.data)["tab_id"] == "tab-1"

    def test_the_tab_id_parameter_is_documented_as_mcp_only(self):
        for tool in (ManagePromptVariablesTool(), GetPromptVariablesTool()):
            assert "MCP only" in tool.parameters["properties"]["tab_id"]["description"]

    def test_list_tab_drafts_is_withheld_from_a_chat_with_a_form(self):
        assert ListTabDraftsTool().is_available(None) is True
        assert ListTabDraftsTool().is_available({"tab_id": "tab-1"}) is False


class TestConfirmedPush:
    @pytest.mark.asyncio
    async def test_one_push_carries_the_new_doc_and_revision(self):
        manager, notifier = make_manager()
        ctx = _ctx(manager, form_state={"tab_id": "tab-1"})

        await ManagePromptVariablesTool().execute_confirmed(ctx, operations=[MOOD, NOTE])

        [(user_id, message)] = notifier.sent
        assert user_id == "user-1"
        assert message == {
            "type": "tab_variables_changed",
            "tab_id": "tab-1",
            "doc": {"variables": manager.variables("user-1", "tab-1")},
            "rev": 1,
        }
        assert set(message["doc"]["variables"]) == {"mood", "note"}

    @pytest.mark.asyncio
    async def test_revision_climbs_with_each_approved_change(self):
        manager, notifier = make_manager()
        ctx = _ctx(manager, form_state={"tab_id": "tab-1"})

        await ManagePromptVariablesTool().execute_confirmed(ctx, operations=[MOOD])
        await ManagePromptVariablesTool().execute_confirmed(ctx, operations=[NOTE])

        assert [m["rev"] for _, m in notifier.sent] == [1, 2]

    @pytest.mark.asyncio
    async def test_all_operations_rejected_writes_and_pushes_nothing(self):
        manager, notifier = make_manager()
        ctx = _ctx(manager, form_state={"tab_id": "tab-1"})

        result = await ManagePromptVariablesTool().execute_confirmed(ctx, operations=[BAD_AGE])

        assert not result.success
        assert manager.find("user-1", "tab-1") is None
        assert notifier.sent == []

    @pytest.mark.asyncio
    async def test_a_partly_valid_batch_applies_the_valid_part_and_reports_the_rest(self):
        manager, notifier = make_manager()
        ctx = _ctx(manager, form_state={"tab_id": "tab-1"})

        result = await ManagePromptVariablesTool().execute_confirmed(ctx, operations=[NOTE, BAD_AGE])

        payload = json.loads(result.data)
        assert payload["applied_count"] == 1 and payload["rejected_operations"]
        assert set(manager.variables("user-1", "tab-1")) == {"note"}
        assert len(notifier.sent) == 1

    @pytest.mark.asyncio
    async def test_a_draft_the_store_refuses_fails_the_call_and_pushes_nothing(self):
        manager, notifier = make_manager()
        huge = {"op": "set", "name": "t", "type": "text", "value": "x" * 2001}
        ctx = _ctx(manager, form_state={"tab_id": "tab-1"})

        result = await ManagePromptVariablesTool().execute_confirmed(ctx, operations=[huge])

        assert not result.success
        assert notifier.sent == []

    @pytest.mark.asyncio
    async def test_updating_a_linked_variable_keeps_it_linked(self):
        manager, _ = make_manager()
        seed(manager, "user-1", [{"name": "mood", "type": "choice", "options": ["a", "b"], "mode": "shuffle", "linked": True}])
        ctx = _ctx(manager, form_state={"tab_id": "tab-1"})

        await ManagePromptVariablesTool().execute_confirmed(ctx, operations=[MOOD])

        assert manager.variables("user-1", "tab-1")["mood"]["linked"] is True


class TestTabResolution:
    @pytest.mark.asyncio
    async def test_the_open_chat_tab_wins_over_a_model_supplied_tab_id(self):
        manager, _ = make_manager()
        ctx = _ctx(manager, form_state={"tab_id": "tab-1"})

        await ManagePromptVariablesTool().execute_confirmed(ctx, tab_id="other", operations=[NOTE])

        assert manager.find("user-1", "other") is None
        assert "note" in manager.variables("user-1", "tab-1")

    @pytest.mark.asyncio
    async def test_over_mcp_a_missing_tab_id_is_an_error_naming_the_fix(self):
        manager, notifier = make_manager()
        ctx = drafts_context({}, manager=manager, chat_session=False)

        result = await ManagePromptVariablesTool().execute_confirmed(ctx, operations=[NOTE])

        assert not result.success and "tab_id" in result.error
        assert notifier.sent == []

    @pytest.mark.asyncio
    async def test_over_mcp_the_tab_id_argument_selects_the_draft(self):
        manager, notifier = make_manager()
        ctx = drafts_context({}, manager=manager, chat_session=False)

        result = await ManagePromptVariablesTool().execute_confirmed(ctx, tab_id="tab-9", operations=[NOTE])

        assert result.success
        assert "note" in manager.variables("user-1", "tab-9")
        assert [m["tab_id"] for _, m in notifier.sent] == ["tab-9"]

    @pytest.mark.asyncio
    async def test_in_chat_a_model_supplied_tab_id_is_ignored_even_with_no_tab_open(self):
        manager, notifier = make_manager()
        ctx = _ctx(manager, form_state={"tab_id": None})

        read = await GetPromptVariablesTool().execute(ctx, tab_id="tab-9")
        write = await ManagePromptVariablesTool().execute_confirmed(ctx, tab_id="tab-9", operations=[NOTE])

        assert read.success and json.loads(read.data)["variables"] == []
        assert not write.success and write.error.startswith("No Generate tab is open")
        assert manager.find("user-1", "tab-9") is None
        assert notifier.sent == []

    @pytest.mark.asyncio
    async def test_a_malformed_tab_id_is_a_tool_error_not_a_crash(self):
        manager, _ = make_manager()
        ctx = drafts_context({}, manager=manager, chat_session=False)

        result = await GetPromptVariablesTool().execute(ctx, tab_id="../etc")

        assert not result.success and "tab_id" in result.error

    @pytest.mark.asyncio
    async def test_without_a_draft_store_the_tools_say_so(self):
        ctx = drafts_context({"form_state": {"tab_id": "t"}})
        ctx.tab_drafts = None

        result = await GetPromptVariablesTool().execute(ctx)

        assert not result.success


class TestUserScoping:
    @pytest.mark.asyncio
    async def test_a_tool_run_as_another_user_sees_and_changes_only_their_own_draft(self):
        manager, notifier = make_manager()
        seed(manager, "user-1", [{"name": "secret", "type": "text", "value": "owner"}])

        read = await GetPromptVariablesTool().execute(_ctx(manager, "user-2", {"tab_id": "tab-1"}))
        await ManagePromptVariablesTool().execute_confirmed(
            _ctx(manager, "user-2", {"tab_id": "tab-1"}), operations=[NOTE]
        )

        assert json.loads(read.data)["variables"] == []
        assert set(manager.variables("user-1", "tab-1")) == {"secret"}
        assert set(manager.variables("user-2", "tab-1")) == {"note"}
        assert [u for u, _ in notifier.sent] == ["user-2"]


class TestListTabDrafts:
    @pytest.mark.asyncio
    async def test_lists_only_the_callers_tabs_with_counts(self):
        manager, _ = make_manager()
        seed(manager, "user-1", [{"name": "a", "type": "text", "value": "x"}, {"name": "b", "type": "text", "value": "y"}], "tab-a")
        seed(manager, "user-2", [{"name": "c", "type": "text", "value": "z"}], "tab-z")

        result = await ListTabDraftsTool().execute(_ctx(manager))

        data = json.loads(result.data)
        assert [(t["tab_id"], t["variable_count"]) for t in data["tabs"]] == [("tab-a", 2)]
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_no_drafts_says_so(self):
        manager, _ = make_manager()

        data = json.loads((await ListTabDraftsTool().execute(_ctx(manager))).data)

        assert data["tabs"] == [] and "No Generate tab" in data["note"]

    def test_is_read_only_and_in_the_generation_mode(self):
        tool = ListTabDraftsTool()
        assert tool.requires_approval is False and tool.modes == ["generation"]
        assert tool.parameters["properties"] == {}


class TestNoGenerateTabOpen:
    @pytest.mark.asyncio
    async def test_reading_answers_with_the_short_message_and_succeeds(self):
        manager, _ = make_manager()
        ctx = _ctx(manager, form_state={"tab_id": None})

        result = await GetPromptVariablesTool().execute(ctx)

        assert result.success
        data = json.loads(result.data)
        assert data["variables"] == [] and data["note"].startswith("No Generate tab is open")

    @pytest.mark.asyncio
    async def test_changing_is_refused_with_the_same_message_and_nothing_is_written(self):
        manager, notifier = make_manager()
        ctx = _ctx(manager, form_state={"tab_id": None})

        preview = await ManagePromptVariablesTool().execute(ctx, operations=[NOTE])
        confirmed = await ManagePromptVariablesTool().execute_confirmed(ctx, operations=[NOTE])

        for result in (preview, confirmed):
            assert not result.success and result.error.startswith("No Generate tab is open")
        assert manager.list("user-1") == []
        assert notifier.sent == []

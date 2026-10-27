from src.features.llm.harness.baseline import turn_only_replay
import asyncio
from src.features.chat.reply_contract import approval_continuation_message
from src.features.llm.tools.base import ToolServices
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi import HTTPException

from src.features.chat import conversation as conversation_module
from src.features.chat.dto import ContinueFromRequest, SendMessageRequest
from src.features.chat.exceptions import ContinuationConflictException, ToolExecutionPendingException
from tests.fixtures.chat_modes import mode_registry
from src.features.chat.repository import ChatRepository
from src.features.chat.routes import ChatController
from src.features.chat.runtime import ChatRuntime
from src.features.chat.turns import ChatTurnRegistry
from src.features.llm.tools.base import BaseTool, ToolApprovalPreview, ToolContext, ToolResult
from src.features.llm.tools.builtin.prompt_variables_tool import ManagePromptVariablesTool
from tests.fixtures.fake_tab_drafts import make_manager
from src.features.llm.tools.builtin.run_generation_tool import RunGenerationTool
from src.features.llm.harness.tool_runner import ParallelToolRunner
from tests.features.llm.harness.stepping import turn_state
from src.features.llm.harness.engine import ToolRound
from src.features.llm.harness.turn_state import ToolFinished
from src.features.llm.harness.messages import ToolCallPart
from src.features.llm.tools.executor import ToolExecutor
from src.features.llm.tools.workflow import ToolCallGuard
from src.features.llm.harness.legacy import messages_from_dicts
from tests.fixtures.chat_engine import scripted_llm, turn_engine_for
from src.features.llm.tools.registry import ToolRegistry
from src.features.prompt.expander import expand_prompts
from src.platform.security.user import User
from tests.fixtures.seed import seed_user

USER_ID = "user-1"
CONTEXT = {"form_state": {"form_data": {"value": 1}}}


def _call(call_id, name, **arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class _SetValueTool(BaseTool):
    modes = ["generation"]

    def __init__(self, fail_times=0, delay=0.0):
        self.confirmed = 0
        self.fail_times = fail_times
        self.delay = delay

    @property
    def name(self) -> str:
        return "set_value"

    @property
    def description(self) -> str:
        return "Set the value."

    @property
    def parameters(self):
        return {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, context, **kwargs) -> ToolResult:
        current = ((context.session_metadata.get("form_state") or {}).get("form_data") or {}).get("value")
        return ToolResult(
            success=True,
            data=json.dumps({"status": "awaiting"}),
            preview=ToolApprovalPreview(
                action="Set value",
                fields=[{"label": "Value", "value": str(kwargs.get("value")), "old": str(current)}],
            ),
        )

    async def execute_confirmed(self, context, **kwargs) -> ToolResult:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.confirmed += 1
        if self.confirmed <= self.fail_times:
            return ToolResult(success=False, data="", error="disk full")
        return ToolResult(success=True, data=json.dumps({"applied": kwargs.get("value")}))


class _EchoTool(BaseTool):
    modes = ["generation"]

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo."

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, context, **kwargs) -> ToolResult:
        return ToolResult(success=True, data="echoed")


@pytest.fixture
def chat_db(mock_db):
    seed_user(mock_db, USER_ID, username="u", email="u@example.com")
    return mock_db


def _manager(tools, llm, generation_orchestrator=None, tab_drafts=None, store=None):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    executor = ToolExecutor(tool_registry=registry)
    processor = Mock()
    processor.process.side_effect = lambda content, mode=None: ((content or "").strip(), None)
    plugins = Mock()
    plugins.execute_hook.return_value = (Mock(data={}), [])
    repo = ChatRepository()
    manager = ChatRuntime(
        chat_repository=repo,
        llm_service=llm,
        response_processor=processor,
        plugin_registry=plugins,
        chat_mode_registry=mode_registry(),
        tool_executor=executor,
        tool_services=ToolServices(generation_orchestrator=generation_orchestrator, tab_drafts=tab_drafts),
        turn_engine=turn_engine_for(llm, executor, store=store),
        replay_selector=turn_only_replay,
    )
    session = repo.create_session(user_id=USER_ID, llm_config_id="llm-1", mode="generation")
    return manager, repo, session.id


async def _stream(manager, session_id, content="", **kwargs):
    kwargs.setdefault("context_metadata", copy.deepcopy(CONTEXT))
    return [
        event async for event in manager.send_message_stream(
            session_id=session_id, user_id=USER_ID, content=content, **kwargs,
        )
    ]


def _done(events):
    return next(e for e in events if e["event"] == "done")["data"]


async def _resolve(manager, session_id, message_id, approved=True, tool_index=0, **kwargs):
    return await manager.resolve_tool_execution(
        session_id=session_id, user_id=USER_ID, message_id=message_id, tool_index=tool_index, approved=approved,
        **kwargs,
    )


async def _paused(tool, extra_turns=None, extra_tools=None, siblings=False):
    calls = [_call("c1", "set_value", value=2)]
    if siblings:
        calls.append(_call("c2", "echo"))
    llm = scripted_llm([{"tool_calls": calls}] + list(extra_turns or []))
    manager, repo, session_id = _manager([tool, _EchoTool()] + list(extra_tools or []), llm)
    events = await _stream(manager, session_id, "set it to 2")
    message_id = _done(events)["assistant_message"]["id"]
    return manager, repo, session_id, llm, message_id, events


class TestStepEvents:
    @pytest.mark.asyncio
    async def test_each_model_call_streams_a_step_with_its_tool_call_count(self, chat_db):
        llm = scripted_llm([{"tool_calls": [_call("c1", "echo"), _call("c2", "echo")]}, {"content": "all done"}])
        manager, _repo, session_id = _manager([_EchoTool()], llm)

        events = await _stream(manager, session_id, "echo twice")

        steps = [e["data"] for e in events if e["event"] == "step"]
        assert [(step["finish"], step["tool_calls"]) for step in steps] == [("tool_calls", 2), ("stop", 0)]
        assert steps[0]["seq"] < steps[1]["seq"]


class TestPause:
    @pytest.mark.asyncio
    async def test_pending_call_is_recorded_and_siblings_are_skipped(self, chat_db):
        manager, repo, session_id, _llm, message_id, events = await _paused(_SetValueTool(), siblings=True)

        tool_end = next(e for e in events if e["event"] == "tool_end")["data"]
        assert tool_end["approval_status"] == "pending"
        assert tool_end["arguments"] == {"value": 2}
        assert tool_end["preview"]["action"] == "Set value"

        metadata = repo.get_message(message_id).metadata
        pending, skipped = metadata["tool_executions"]
        assert pending["approval_status"] == "pending"
        assert pending["approval_reason"] is None
        assert pending["skipped"] is False
        assert skipped["tool_name"] == "echo"
        assert skipped["skipped"] is True
        assert skipped["approval_status"] is None
        assert skipped["result"]["error"] == "Not run: waiting for approval of set_value."

        state = manager.turn_engine.store.open_turn(session_id)
        assert [call.call_id for call in state.pending] == ["c1"]
        assert [(part.call_id, part.text()) for part in state.gathered] == [(
            "c2", "Not run: waiting for approval of set_value. Call it again if still needed.",
        )]
        assert metadata["turn_rounds"] == 1


class TestResolve:
    @pytest.mark.asyncio
    async def test_resolve_is_idempotent(self, chat_db):
        tool = _SetValueTool()
        manager, repo, session_id, _llm, message_id, _events = await _paused(tool)

        first = await _resolve(manager, session_id, message_id)
        second = await _resolve(manager, session_id, message_id)
        rejected = await _resolve(manager, session_id, message_id, approved=False)

        assert first["status"] == "approved"
        assert first["execution"]["result"] == {"success": True, "data": json.dumps({"applied": 2}), "error": None}
        assert second["status"] == "already_resolved"
        assert rejected["status"] == "already_resolved"
        assert rejected["execution"]["approval_status"] == "approved"
        assert tool.confirmed == 1
        assert repo.get_message(message_id).metadata["tool_executions"][0]["approval_status"] == "approved"

    @pytest.mark.asyncio
    async def test_concurrent_resolves_execute_once(self, chat_db):
        tool = _SetValueTool(delay=0.05)
        manager, _repo, session_id, _llm, message_id, _events = await _paused(tool)

        results = await asyncio.gather(
            _resolve(manager, session_id, message_id),
            _resolve(manager, session_id, message_id),
        )

        assert sorted(r["status"] for r in results) == ["already_resolved", "approved"]
        assert tool.confirmed == 1

    @pytest.mark.asyncio
    async def test_failed_then_retry(self, chat_db):
        tool = _SetValueTool(fail_times=1)
        manager, repo, session_id, _llm, message_id, _events = await _paused(tool)

        failed = await _resolve(manager, session_id, message_id)
        assert failed["status"] == "failed"
        assert failed["execution"]["result"]["error"] == "disk full"

        not_rejectable = await _resolve(manager, session_id, message_id, approved=False)
        assert not_rejectable["status"] == "already_resolved"

        retried = await _resolve(manager, session_id, message_id)
        assert retried["status"] == "approved"
        assert tool.confirmed == 2
        state = manager.turn_engine.store.open_turn(session_id)
        assert [(part.call_id, part.text(), part.is_error) for part in state.gathered] == [
            ("c1", json.dumps({"applied": 2}), False),
        ]
        assert state.pending == ()

    @pytest.mark.asyncio
    async def test_continuation_asks_for_a_one_sentence_confirmation(self, chat_db):
        tool = _SetValueTool()
        manager, _repo, session_id, llm, message_id, _events = await _paused(tool)
        first_system = llm.adapter.calls[0]["system"]

        await _resolve(manager, session_id, message_id)
        await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        assert llm.adapter.calls[-1]["system"] == first_system
        assert "just resolved the pending" not in json.dumps(llm.adapter.calls[0]["messages"])
        last = llm.adapter.calls[-1]["messages"][-1]
        assert last == approval_continuation_message("set_value", "c1")

    @pytest.mark.asyncio
    async def test_failure_reaches_continuation(self, chat_db):
        tool = _SetValueTool(fail_times=5)
        manager, _repo, session_id, llm, message_id, _events = await _paused(tool)

        await _resolve(manager, session_id, message_id)
        await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        tool_message = next(m for m in llm.adapter.calls[-1]["messages"] if m.get("role") == "tool")
        assert tool_message == {"role": "tool", "tool_call_id": "c1", "name": "set_value", "content": "Error: disk full"}

    @pytest.mark.asyncio
    async def test_stale_form_returns_stale_and_keeps_pending(self, chat_db):
        tool = _SetValueTool()
        manager, repo, session_id, _llm, message_id, _events = await _paused(tool)

        changed = {"form_state": {"form_data": {"value": 7}}}
        stale = await _resolve(manager, session_id, message_id, context_metadata=changed)

        assert stale["status"] == "stale"
        assert tool.confirmed == 0
        assert repo.get_message(message_id).metadata["tool_executions"][0]["approval_status"] == "pending"

        unchanged = await _resolve(manager, session_id, message_id, context_metadata=copy.deepcopy(CONTEXT))
        assert unchanged["status"] == "approved"
        assert tool.confirmed == 1

    @pytest.mark.asyncio
    async def test_force_skips_the_stale_check(self, chat_db):
        tool = _SetValueTool()
        manager, _repo, session_id, _llm, message_id, _events = await _paused(tool)

        forced = await _resolve(
            manager, session_id, message_id,
            context_metadata={"form_state": {"form_data": {"value": 7}}}, force=True,
        )

        assert forced["status"] == "approved"
        assert tool.confirmed == 1

    @pytest.mark.asyncio
    async def test_reject_reason_reaches_continuation(self, chat_db):
        tool = _SetValueTool()
        manager, repo, session_id, llm, message_id, _events = await _paused(tool)

        rejected = await _resolve(manager, session_id, message_id, approved=False, reason="too dark")
        assert rejected["status"] == "rejected"
        assert rejected["execution"]["approval_reason"] == "too dark"

        events = await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        assert tool.confirmed == 0
        tool_message = next(m for m in llm.adapter.calls[-1]["messages"] if m.get("role") == "tool")
        assert tool_message["content"] == "User rejected this. Reason: too dark"
        assert _done(events)["user_message"] is None
        assert [m.role for m in repo.get_messages(session_id)] == ["user", "assistant", "assistant"]


class TestContinuation:
    @pytest.mark.asyncio
    async def test_continuation_context_equals_normal_send_plus_transcript(self, chat_db):
        tool = _SetValueTool()
        manager, repo, session_id, llm, message_id, _events = await _paused(
            tool, extra_turns=[{"content": "Set to 2."}], siblings=True,
        )
        await _resolve(manager, session_id, message_id)

        events = await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        send_call, continuation_call = llm.adapter.calls
        assert continuation_call["system"] == send_call["system"]
        assert continuation_call["tools"] == send_call["tools"]
        assert continuation_call["messages"][:len(send_call["messages"])] == send_call["messages"]
        assert continuation_call["messages"][len(send_call["messages"]):] == [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "set_value", "arguments": {"value": 2}}},
                    {"id": "c2", "type": "function", "function": {"name": "echo", "arguments": {}}},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "set_value", "content": json.dumps({"applied": 2})},
            {
                "role": "tool", "tool_call_id": "c2", "name": "echo",
                "content": "Not run: waiting for approval of set_value. Call it again if still needed.",
            },
            approval_continuation_message("set_value", "c1"),
        ]

        reply = _done(events)["assistant_message"]
        assert reply["content"] == "Set to 2."
        assert reply["metadata"]["continued_from"] == {"message_id": message_id, "tool_index": 0}
        assert len(repo.get_messages(session_id)) == 3

    @pytest.mark.asyncio
    async def test_continuation_keeps_a_group_loaded_earlier_in_the_turn(self, chat_db):
        tool = _SetValueTool()
        tool.context_group = "extra"
        llm = scripted_llm(provider_options={"context_window": 8192}, turns=[
            {"tool_calls": [_call("c0", "use_tools", group="extra")]},
            {"tool_calls": [_call("c1", "set_value", value=2)]},
            {"content": "Set to 2."},
        ])
        manager, _repo, session_id = _manager([tool, _EchoTool()], llm)
        events = await _stream(manager, session_id, "set it to 2")
        message_id = _done(events)["assistant_message"]["id"]
        await _resolve(manager, session_id, message_id, tool_index=1)

        await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 1})

        names = [[schema["function"]["name"] for schema in call["tools"]] for call in llm.adapter.calls]
        assert names[0] == ["echo", "use_tools"]
        assert names[1] == ["echo", "use_tools", "set_value"]
        assert names[2] == ["echo", "set_value"]

    @pytest.mark.asyncio
    async def test_continuation_while_pending_is_refused(self, chat_db):
        manager, _repo, session_id, _llm, message_id, _events = await _paused(_SetValueTool())

        with pytest.raises(ToolExecutionPendingException):
            await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        controller = ChatController(chat_runtime=manager, turn_registry=ChatTurnRegistry())
        user = User(id=USER_ID, username="u", email="u@example.com", password_hash="h", account_type="USER")
        request = SendMessageRequest(content="", continue_from=ContinueFromRequest(message_id=message_id, tool_index=0))
        with pytest.raises(HTTPException) as excinfo:
            await controller.send_message_stream(session_id, request, user)
        assert excinfo.value.status_code == 409

    @pytest.mark.asyncio
    async def test_second_continuation_conflicts(self, chat_db):
        manager, _repo, session_id, _llm, message_id, _events = await _paused(_SetValueTool())
        await _resolve(manager, session_id, message_id)
        await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        with pytest.raises(ContinuationConflictException):
            await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

    @pytest.mark.asyncio
    async def test_buffered_send_cannot_continue(self, chat_db):
        manager, _repo, session_id, _llm, message_id, _events = await _paused(_SetValueTool())
        controller = ChatController(chat_runtime=manager, turn_registry=ChatTurnRegistry())
        user = User(id=USER_ID, username="u", email="u@example.com", password_hash="h", account_type="USER")
        request = SendMessageRequest(content="", continue_from=ContinueFromRequest(message_id=message_id, tool_index=0))

        with pytest.raises(HTTPException) as excinfo:
            await controller.send_message(session_id, request, user)
        assert excinfo.value.status_code == 400

    @pytest.mark.asyncio
    async def test_round_budget_carries_across_the_chain(self, chat_db, monkeypatch):
        monkeypatch.setattr(conversation_module, "_MAX_TOOL_ROUNDS", 2)
        tool = _SetValueTool()
        llm = scripted_llm([
            {"tool_calls": [_call("e1", "echo")]},
            {"tool_calls": [_call("c1", "set_value", value=2)]},
            {"content": "Wrapped up."},
        ])
        manager, repo, session_id = _manager([tool, _EchoTool()], llm)
        message_id = _done(await _stream(manager, session_id, "go"))["assistant_message"]["id"]
        assert repo.get_message(message_id).metadata["turn_rounds"] == 2

        resolved = await _resolve(manager, session_id, message_id, tool_index=1)
        assert resolved["status"] == "approved"
        events = await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 1})

        assert any(e["event"] == "status" and e["data"]["step"] == "tool_budget_exhausted" for e in events)
        assert len(llm.adapter.calls) == 3
        assert llm.adapter.calls[-1]["tools"] is None
        reply = _done(events)["assistant_message"]
        assert reply["content"] == "Wrapped up."
        assert reply["metadata"]["turn_rounds"] == 2

    @pytest.mark.asyncio
    async def test_reproposing_an_approved_call_does_not_pause_again(self, chat_db):
        tool = _SetValueTool()
        manager, repo, session_id, llm, message_id, _events = await _paused(
            tool, extra_turns=[{"tool_calls": [_call("c9", "set_value", value=2)]}, {"content": "Done."}],
        )
        await _resolve(manager, session_id, message_id)

        events = await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        reply = _done(events)["assistant_message"]
        assert reply["content"] == "Done."
        [execution] = reply["metadata"]["tool_executions"]
        assert execution["approval_status"] is None
        assert execution["result"]["data"] == "Already applied earlier in this turn."
        assert tool.confirmed == 1
        tool_message = [m for m in llm.adapter.calls[-1]["messages"] if m.get("role") == "tool"][-1]
        assert tool_message["content"] == "Already applied earlier in this turn."

    @pytest.mark.asyncio
    async def test_new_user_turn_resets_the_reproposal_guard(self, chat_db):
        tool = _SetValueTool()
        manager, _repo, session_id, _llm, message_id, _events = await _paused(
            tool, extra_turns=[{"content": "Set."}, {"tool_calls": [_call("c5", "set_value", value=2)]}],
        )
        await _resolve(manager, session_id, message_id)
        await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        events = await _stream(manager, session_id, "do it again")

        [execution] = _done(events)["assistant_message"]["metadata"]["tool_executions"]
        assert execution["approval_status"] == "pending"


class TestSendWhilePending:
    @pytest.mark.asyncio
    async def test_reply_rejects_the_pending_call(self, chat_db):
        tool = _SetValueTool()
        manager, repo, session_id, llm, message_id, _events = await _paused(
            tool, extra_turns=[{"content": "Okay, 3 it is."}],
        )

        events = await _stream(manager, session_id, "no, make it 3")

        resolved = _done(events)["resolved_message"]
        assert resolved["id"] == message_id
        assert resolved["tool_executions"][0]["approval_status"] == "rejected"
        assert resolved["tool_executions"][0]["approval_reason"] == "no, make it 3"
        assert repo.get_message(message_id).metadata["tool_executions"][0]["approval_status"] == "rejected"
        assert tool.confirmed == 0

        messages = llm.adapter.calls[-1]["messages"]
        assert messages[-2] == {
            "role": "tool", "tool_call_id": "c1", "name": "set_value",
            "content": "User rejected this. Reason: no, make it 3",
        }
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"].startswith("no, make it 3")

    @pytest.mark.asyncio
    async def test_reply_reason_is_truncated(self, chat_db):
        manager, repo, session_id, _llm, message_id, _events = await _paused(_SetValueTool())

        await _stream(manager, session_id, "x" * 2500)

        reason = repo.get_message(message_id).metadata["tool_executions"][0]["approval_reason"]
        assert reason == "x" * 2000

    @pytest.mark.asyncio
    async def test_buffered_send_rejects_the_pending_call(self, chat_db):
        llm = scripted_llm([{"tool_calls": [_call("c1", "set_value", value=2)]}])
        manager, repo, session_id = _manager([_SetValueTool()], llm)
        message_id = _done(await _stream(manager, session_id, "set"))["assistant_message"]["id"]

        llm.adapter.turns.append({"content": "Fine."})
        response = await manager.send_message(
            session_id=session_id, user_id=USER_ID, content="never mind", context_metadata=copy.deepcopy(CONTEXT),
        )

        assert response.resolved_message["id"] == message_id
        assert repo.get_message(message_id).metadata["tool_executions"][0]["approval_status"] == "rejected"
        assert response.assistant_message.content == "Fine."


class TestHistoryBudget:
    def _runner(self, budget_tokens):
        manager = Mock()
        manager.tool_services.settings.get_setting = Mock(return_value=budget_tokens)
        return conversation_module.ConversationRunner(manager)

    def test_trims_whole_rounds_only(self):
        history, _ = messages_from_dicts([
            {"role": "user", "content": "u" * 40},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "function": {"name": "t", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "a", "content": "r" * 40},
            {"role": "assistant", "content": "a" * 20},
            {"role": "user", "content": "q" * 20},
        ])
        runner = self._runner(budget_tokens=20)

        info = runner._apply_history_budget(history)

        assert info["truncated"] is True
        assert [m.role.value for m in history] == ["assistant", "user"]

    def test_keeps_current_turn_rounds(self):
        history, _ = messages_from_dicts([
            {"role": "user", "content": "old" * 50},
            {"role": "user", "content": "q" * 10},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "function": {"name": "t", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "a", "content": "r" * 400},
        ])

        info = self._runner(budget_tokens=10)._apply_history_budget(history)

        assert info["messages_sent"] == 3
        assert [m.role.value for m in history] == ["user", "assistant", "tool"]

    def test_counts_a_rounds_reasoning_toward_its_cost(self):
        call = {"id": "a", "function": {"name": "t", "arguments": {}}}
        entries = [
            {"role": "user", "content": "u" * 4},
            {"role": "assistant", "content": "", "tool_calls": [call], "reasoning": "r" * 400},
            {"role": "tool", "tool_call_id": "a", "content": "ok"},
            {"role": "assistant", "content": "a" * 4},
            {"role": "user", "content": "q" * 4},
        ]
        history, _ = messages_from_dicts(entries)
        without_reasoning, _ = messages_from_dicts([{key: value for key, value in entry.items() if key != "reasoning"} for entry in entries])

        reasoned = self._runner(budget_tokens=25)._apply_history_budget(history)
        plain = self._runner(budget_tokens=25)._apply_history_budget(without_reasoning)

        assert plain["truncated"] is False
        assert reasoned["truncated"] is True
        assert [m.role.value for m in history] == ["assistant", "user"]


class TestFormStateSyncAcrossChainedApprovals:
    _VARIABLE_OPS = [{
        "op": "set", "name": "mood", "type": "choice",
        "options": ["noir", "sunlit"], "mode": "pin", "pinned_index": 1,
    }]
    _CONTEXT = {
        "form_state": {
            "preset": "preset-sdxl",
            "mode": "txt2img",
            "form_data": {"prompt": "${mood} lighting on the subject"},
            "tab_id": "tab-1",
        },
    }

    async def _chain(self):
        orchestrator = AsyncMock()
        orchestrator.start_generation.return_value = {"generation_id": "gen-1", "status": {"status": "pending"}}
        llm = scripted_llm([
            {"tool_calls": [_call("v1", "manage_prompt_variables", operations=self._VARIABLE_OPS)]},
            {"tool_calls": [_call("g1", "run_generation")]},
        ])
        self.drafts, self.notifier = make_manager()
        manager, repo, session_id = _manager(
            [ManagePromptVariablesTool(), RunGenerationTool()], llm,
            generation_orchestrator=orchestrator, tab_drafts=self.drafts,
        )
        events = await _stream(manager, session_id, "pin mood to sunlit, then generate",
                               context_metadata=copy.deepcopy(self._CONTEXT))
        first_id = _done(events)["assistant_message"]["id"]
        await _resolve(manager, session_id, first_id)
        events = await _stream(manager, session_id, context_metadata=None,
                               continue_from={"message_id": first_id, "tool_index": 0})
        return manager, repo, session_id, orchestrator, _done(events)["assistant_message"]

    @pytest.mark.asyncio
    async def test_the_approved_change_lands_in_the_draft_and_is_pushed(self, chat_db):
        _manager_, _repo, _sid, _orchestrator, second = await self._chain()

        [run_generation] = second["metadata"]["tool_executions"]
        assert run_generation["tool_name"] == "run_generation"
        assert run_generation["approval_status"] == "pending"
        assert "variables" not in second["metadata"]["context_metadata"]["form_state"]
        assert self.drafts.variables(USER_ID, "tab-1") == {
            "mood": {"type": "choice", "options": ["noir", "sunlit"], "mode": "pin", "pinnedIndex": 1},
        }
        [(pushed_to, message)] = self.notifier.sent
        assert pushed_to == USER_ID
        assert message["type"] == "tab_variables_changed"
        assert message["tab_id"] == "tab-1" and message["rev"] == 1
        assert message["doc"] == {"variables": self.drafts.variables(USER_ID, "tab-1")}

    @pytest.mark.asyncio
    async def test_approving_after_switching_tabs_writes_and_pushes_the_tab_it_was_proposed_on(self, chat_db):
        llm = scripted_llm([{"tool_calls": [_call("v1", "manage_prompt_variables", operations=self._VARIABLE_OPS)]}])
        drafts, notifier = make_manager()
        manager, _repo, session_id = _manager([ManagePromptVariablesTool()], llm, tab_drafts=drafts)
        events = await _stream(manager, session_id, "pin mood", context_metadata=copy.deepcopy(self._CONTEXT))
        message_id = _done(events)["assistant_message"]["id"]
        switched = copy.deepcopy(self._CONTEXT)
        switched["form_state"]["tab_id"] = "tab-2"

        result = await _resolve(manager, session_id, message_id, context_metadata=switched)

        assert result["status"] == "approved"
        assert "mood" in drafts.variables(USER_ID, "tab-1")
        assert drafts.find(USER_ID, "tab-2") is None
        assert [m["tab_id"] for _, m in notifier.sent] == ["tab-1"]

    @pytest.mark.asyncio
    async def test_follow_up_turn_sees_the_applied_rows_and_a_note_naming_the_approved_call(self, chat_db):
        llm = scripted_llm([
            {"tool_calls": [_call("v1", "manage_prompt_variables", operations=self._VARIABLE_OPS)]},
            {"tool_calls": [_call("e1", "echo")]},
            {"content": "Pinned mood to sunlit."},
        ])
        drafts, _notifier = make_manager()
        manager, _repo, session_id = _manager([ManagePromptVariablesTool(), _EchoTool()], llm, tab_drafts=drafts)
        events = await _stream(manager, session_id, "pin mood", context_metadata=copy.deepcopy(self._CONTEXT))
        message_id = _done(events)["assistant_message"]["id"]
        await _resolve(manager, session_id, message_id)

        await _stream(manager, session_id, context_metadata=None,
                      continue_from={"message_id": message_id, "tool_index": 0})

        final = llm.adapter.calls[-1]["messages"]
        applied = next(m for m in final if m.get("tool_call_id") == "v1")
        payload = json.loads(applied["content"])
        assert payload["status"] == "applied"
        assert [row["field_name"] for row in payload["applied_changes"]] == ["${mood}"]
        assert payload["applied_changes"][0]["new_value"]
        assert "${mood}" in payload["summary"]
        note = next(m for m in final if m["role"] == "user" and "just resolved the pending" in m["content"])
        assert "manage_prompt_variables" in note["content"] and "v1" in note["content"]
        assert "last tool result" not in note["content"]
        assert final.index(applied) < final.index(note)
        assert any(m.get("tool_call_id") == "e1" for m in final)

    @pytest.mark.asyncio
    async def test_declining_the_change_writes_and_pushes_nothing(self, chat_db):
        llm = scripted_llm([{"tool_calls": [_call("v1", "manage_prompt_variables", operations=self._VARIABLE_OPS)]}])
        drafts, notifier = make_manager()
        manager, _repo, session_id = _manager([ManagePromptVariablesTool()], llm, tab_drafts=drafts)
        events = await _stream(manager, session_id, "pin mood", context_metadata=copy.deepcopy(self._CONTEXT))
        message_id = _done(events)["assistant_message"]["id"]

        result = await _resolve(manager, session_id, message_id, approved=False)

        assert result["status"] != "approved"
        assert drafts.find(USER_ID, "tab-1") is None
        assert notifier.sent == []

    @pytest.mark.asyncio
    async def test_approving_the_chained_run_generation_expands_the_new_variable(self, chat_db):
        manager, _repo, session_id, orchestrator, second = await self._chain()

        result = await _resolve(manager, session_id, second["id"])

        assert result["status"] == "approved"
        request = orchestrator.start_generation.call_args[1]["request"]
        assert request.variables == {"mood": "sunlit"}
        expanded = expand_prompts(
            request.prompts[0].positive, request.prompts[0].negative,
            count=1, base_seed=0, variables=request.variables,
        )
        assert expanded[0].positive == "sunlit lighting on the subject"


class TestApprovalPreviewPayload:
    @pytest.mark.asyncio
    async def test_remove_tool_fills_structured_preview(self):
        from src.features.llm.tools.builtin.phrasebook_tool import RemovePhrasebookValuesTool

        category_repo = MagicMock()
        category_repo.get_by_path.return_value = SimpleNamespace(id="cat-1", path="camera")
        value_repo = MagicMock()
        value_repo.get_by_id.side_effect = [
            SimpleNamespace(id="1", label="low angle", value="low angle", category_id="cat-1"),
            SimpleNamespace(id="2", label="high angle", value="high angle", category_id="cat-1"),
        ]
        ctx = ToolContext(
            user_id="user-1",
            phrasebook_category_repository=category_repo,
            phrasebook_value_repository=value_repo,
        )

        result = await RemovePhrasebookValuesTool().execute(ctx, value_ids=["1", "2"], category_path="camera")

        assert result.success
        assert result.preview.action == "Remove"
        assert result.preview.target == "from category camera"
        assert result.preview.items == ["low angle", "high angle"]

    @pytest.mark.asyncio
    async def test_preview_serializes_into_tool_end_event(self):
        registry = ToolRegistry()
        registry.register(_SetValueTool())
        executor = ToolExecutor(tool_registry=registry)
        context = ToolRound(
            tool_context=ToolContext(user_id="user-1"), allowed_tools=["set_value"], deferred_tools={}, guard=ToolCallGuard(),
        )

        events = [
            event async for event in ParallelToolRunner(executor).run(
                [ToolCallPart(call_id="c1", name="set_value", args={"value": 2})], turn_state(), context,
            )
        ]

        finished = next(event for event in events if isinstance(event, ToolFinished))
        payload = ToolExecutor.tool_end_payload(finished.record)
        assert payload["approval_status"] == "pending"
        assert payload["preview"]["action"] == "Set value"

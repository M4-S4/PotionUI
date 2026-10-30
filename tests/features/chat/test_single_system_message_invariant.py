import asyncio
from typing import Any
from unittest.mock import Mock

from src.features.chat.context_builder import ChatContextBuilder
from src.features.chat.modes import ChatMode
from src.features.chat.reply_contract import REPLY_CONTRACT_REMINDER
from src.features.llm.harness.build import user_message
from src.features.llm.harness.context import user_text
from src.features.llm.harness.engine import TOOL_BUDGET_EXHAUSTED_MESSAGE, TurnInput
from src.features.llm.harness.events import Finish, FinishReason, TextDelta
from src.features.llm.harness.legacy import messages_from_dicts
from src.features.llm.harness.messages import ContextPart, Role
from src.features.llm.tools.base import BaseTool, ToolContext, ToolResult
from tests.fixtures.llm_engine import drain, scripted_engine, tool_context
from tests.fixtures.llm_harness import ScriptedAdapter, text_step, tool_step


def _no_system_messages(messages):
    return not any(message.role is Role.SYSTEM for message in messages)


class TestChatContextBuilderNeverInsertsSystemMessages:
    def test_every_static_injector_folds_into_the_last_user_message(self):
        history, _ = messages_from_dicts([
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "current question"},
        ])

        ChatContextBuilder.inject_resource_block(history, [])
        ChatContextBuilder.inject_prompt_state_block(
            history,
            {"segments": [{"index": 0, "id": "a", "type": "content", "enabled": True, "content": "x"}]},
        )
        ChatContextBuilder.inject_reply_contract_reminder_block(
            history, ChatMode(id="test", name="Test", structured_reply=True)
        )
        ChatContextBuilder.inject_tool_availability_block(history, [], {"get_active_models": "off_by_toggle"})

        assert _no_system_messages(history)
        assert len(history) == 3
        assert user_text(history[0]) == "earlier"
        assert history[1].text() == "reply"

        folded = user_text(history[2])
        assert history[2].role is Role.USER
        assert folded.startswith("current question\n\n<context>\n")
        assert "PROMPT STATE" in folded
        assert REPLY_CONTRACT_REMINDER in folded
        assert "Tools are unavailable in this conversation" in folded

    def test_contributor_and_memory_blocks_fold_in_too(self):
        manager = Mock()
        manager.chat_mode_registry.get.return_value = ChatMode(
            id="custom", name="Custom", context_contributor=lambda cm, s, u: "CONTRIBUTOR BLOCK",
        )
        manager.tool_services.llm_memory_repository = None
        builder = ChatContextBuilder(manager)

        history = [user_message("hello")]
        asyncio.run(builder.inject_contributor_block(history, Mock(mode="custom"), {}, "user-1"))

        assert _no_system_messages(history)
        assert len(history) == 1
        assert user_text(history[0]) == "hello\n\n<context>\nCONTRIBUTOR BLOCK\n</context>"


class _Echo(BaseTool):
    name = "echo"
    description = "Echo."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="ok")


def _turn(**overrides: Any) -> TurnInput:
    values = dict(
        session_id="s1", llm_id="cfg-scripted", system="SYS", history=(),
        new_messages=(user_message("go"),), tool_context=tool_context(),
    )
    values.update(overrides)
    return TurnInput(**values)


def _notes(request, kind):
    return [
        message for message in request.messages
        if message.role is Role.USER and any(isinstance(part, ContextPart) and part.kind == kind for part in message.parts)
    ]


class TestTurnEngineNeverInsertsSystemMessages:
    async def test_the_iteration_nudge_is_a_trailing_user_message(self):
        adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "echo"}), text_step("done"))
        rig = scripted_engine(adapter, [_Echo()])

        await drain(rig.engine.run(_turn(iteration_nudge="call a tool now")))

        request = adapter.requests[1]
        assert all(_no_system_messages(item.messages) for item in adapter.requests)
        assert user_text(request.messages[0]) == "go"
        assert [user_text(note) for note in _notes(request, "nudge")] == ["<context>\ncall a tool now\n</context>"]

    async def test_the_final_wrap_up_note_is_a_trailing_user_message(self):
        adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "echo"}), text_step("done"))
        rig = scripted_engine(adapter, [_Echo()])

        await drain(rig.engine.run(_turn(max_rounds=1)))

        request = adapter.requests[-1]
        assert _no_system_messages(request.messages)
        assert user_text(request.messages[0]) == "go"
        assert [user_text(note) for note in _notes(request, "budget")] == [f"<context>\n{TOOL_BUDGET_EXHAUSTED_MESSAGE}\n</context>"]

    async def test_the_steer_nudge_is_a_trailing_user_message(self):
        adapter = ScriptedAdapter(
            [TextDelta(block=1, text='<tool_call>{"name": "echo", "arguments": {'), Finish(reason=FinishReason.LENGTH)],
            text_step("done"),
        )
        rig = scripted_engine(adapter, [_Echo()])

        await drain(rig.engine.run(_turn()))

        request = adapter.requests[1]
        assert _no_system_messages(request.messages)
        assert user_text(request.messages[0]) == "go"
        assert len(_notes(request, "rescue")) == 1

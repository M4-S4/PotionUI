import copy
import json
from unittest.mock import Mock

import pytest

from src.features.chat.repository import ChatRepository
from src.features.chat.runtime import ChatRuntime
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    ReasoningDelta,
    TextDelta,
    ToolArgsDelta,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.tools.base import BaseTool, ToolResult, ToolServices
from src.features.llm.tools.executor import ToolExecutor
from src.features.llm.tools.registry import ToolRegistry
from tests.fixtures.chat_engine import scripted_llm, turn_engine_for
from tests.fixtures.chat_modes import mode_registry
from tests.fixtures.seed import seed_user

USER_ID = "user-1"


class _WriteNote(BaseTool):
    modes = ["generation"]
    name = "write_note"
    description = "Saves a note."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def __init__(self) -> None:
        self.saved = []

    async def execute(self, context, **kwargs) -> ToolResult:
        self.saved.append(kwargs)
        return ToolResult(success=True, data="saved")


@pytest.fixture
def chat_db(mock_db):
    seed_user(mock_db, USER_ID, username="u", email="u@example.com")
    return mock_db


def _manager(tool, llm):
    registry = ToolRegistry()
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
        tool_services=ToolServices(),
        turn_engine=turn_engine_for(llm, executor),
    )
    session = repo.create_session(user_id=USER_ID, llm_config_id="llm-1", mode="generation")
    return manager, repo, session.id


def _cut_off_step():
    return [
        ReasoningDelta(block=0, text="doomed plan"),
        TextDelta(block=1, text="Saving it. "),
        ToolCallStart(block=2, call_id="t1", name="write_note"),
        ToolArgsDelta(block=2, text='{"text": "a lo'),
        UsageReported(Usage(input_tokens=40, output_tokens=8, total_tokens=48)),
        Finish(reason=FinishReason.LENGTH, raw_reason="length"),
    ]


def _complete_step():
    return {
        "thinking": "kept plan",
        "content": "Saving it. ",
        "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "write_note", "arguments": json.dumps({"text": "a long note"})}}],
        "usage": {"input_tokens": 50, "output_tokens": 9, "total_tokens": 59},
    }


def _answer_step():
    return {"content": "Saved.", "usage": {"input_tokens": 70, "output_tokens": 3, "total_tokens": 73}}


async def _stream(manager, session_id):
    return [
        event async for event in manager.send_message_stream(
            session_id=session_id, user_id=USER_ID, content="save a note",
            context_metadata=copy.deepcopy({"form_state": {"form_data": {}}}),
        )
    ]


@pytest.mark.asyncio
async def test_a_retried_step_is_announced_and_its_text_and_thinking_are_not_kept(chat_db):
    note = _WriteNote()
    llm = scripted_llm(turns=[_cut_off_step(), _complete_step(), _answer_step()], show_thinking=True, max_tokens=2000)
    manager, repo, session_id = _manager(note, llm)

    events = await _stream(manager, session_id)

    discarded = [event["data"] for event in events if event["event"] == "step_discarded"]
    steps = [event["data"] for event in events if event["event"] == "step"]
    assert len(discarded) == 1
    assert discarded[0]["seq"] == steps[0]["seq"]
    assert note.saved == [{"text": "a long note"}]
    done = next(event["data"] for event in events if event["event"] == "done")
    saved = repo.get_message(done["assistant_message"]["id"])
    assert saved.content == "Saving it. Saved."
    assert [segment["text"] for segment in saved.metadata["thinking"]["segments"]] == ["kept plan"]


@pytest.mark.asyncio
async def test_the_reply_reports_the_prompt_size_of_its_last_request(chat_db):
    llm = scripted_llm(turns=[_complete_step(), _answer_step()])
    manager, repo, session_id = _manager(_WriteNote(), llm)

    events = await _stream(manager, session_id)

    done = next(event["data"] for event in events if event["event"] == "done")
    assert done["assistant_message"]["prompt_tokens"] == 70
    assert repo.get_message(done["assistant_message"]["id"]).prompt_tokens == 70

import copy
from unittest.mock import Mock

import pytest

from src.features.chat.repository import ChatRepository
from src.features.chat.runtime import ChatRuntime
from src.features.llm.harness.replay import policy_replay
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

    async def execute(self, context, **kwargs) -> ToolResult:
        return ToolResult(success=True, data="saved")


@pytest.fixture
def chat_db(mock_db):
    seed_user(mock_db, USER_ID, username="u", email="u@example.com")
    return mock_db


def _manager(llm):
    registry = ToolRegistry()
    registry.register(_WriteNote())
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
        replay_selector=policy_replay,
    )
    session = repo.create_session(user_id=USER_ID, llm_config_id="llm-1", mode="generation")
    return manager, repo, session.id


@pytest.mark.asyncio
async def test_broken_tool_markup_in_a_streamed_reply_never_reaches_the_user(chat_db):
    llm = scripted_llm(turns=[{
        "chunks": [
            "Saving it. ",
            '<tool_action type="write_note">',
            "half a note",
            " <tool_call>{\"name\": \"write_note\", \"arguments\": {\"text\": \"a lo",
        ],
    }])
    manager, repo, session_id = _manager(llm)

    events = [
        event async for event in manager.send_message_stream(
            session_id=session_id, user_id=USER_ID, content="save a note",
            context_metadata=copy.deepcopy({"form_state": {"form_data": {}}}),
        )
    ]

    streamed = "".join(event["data"]["content"] for event in events if event["event"] == "token")
    done = next(event["data"] for event in events if event["event"] == "done")
    stored = repo.get_message(done["assistant_message"]["id"]).content
    for shown in (streamed, stored, done["assistant_message"]["content"]):
        assert "<tool_call" not in shown and "<tool_action" not in shown and "half a note" not in shown
    assert "Saving it." in streamed

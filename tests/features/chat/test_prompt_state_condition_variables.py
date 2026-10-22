from unittest.mock import AsyncMock, Mock

import pytest

from src.features.chat.dto import MessageResponse
from src.features.chat.runtime import ChatRuntime
from src.features.llm.tools.base import ToolServices
from tests.fixtures.chat_modes import mode_registry
from tests.fixtures.fake_tab_drafts import TAB_ID, make_manager, seed

CONTEXT = {
    "form_state": {"tab_id": TAB_ID, "preset": "sdxl/base"},
    "segments": [
        {"index": 0, "id": "s1", "type": "content", "enabled": True, "content": 'woman, <<if $mood = "Noir">>rain<<end>>'},
    ],
}


def _runtime(owner="user-1"):
    drafts, _ = make_manager()
    seed(drafts, owner, [{"name": "mood", "type": "choice", "options": ["noir", '12" vinyl'], "mode": "shuffle"}])
    plugins = Mock()
    plugins.execute_hook.return_value = (Mock(data={}), [])
    processor = Mock()
    processor.process.side_effect = lambda content, mode=None: (content, {"raw": content})
    repo = Mock()
    session = Mock(id="session-1", user_id="user-1", status="active", llm_config_id="llm-1", mode="generation", metadata=None)
    repo.get_session.return_value = session
    repo.get_conversation_history.return_value = []
    repo.add_message.side_effect = [
        MessageResponse(id="m-user", session_id="session-1", role="user", content="hi"),
        MessageResponse(id="m-asst", session_id="session-1", role="assistant", content="ok"),
    ]
    llm = Mock()
    runtime = ChatRuntime(
        chat_repository=repo,
        llm_service=llm,
        response_processor=processor,
        plugin_registry=plugins,
        chat_mode_registry=mode_registry(),
        tool_services=ToolServices(tab_drafts=drafts),
    )
    return runtime, llm


def _assert_prompt_state_carries_the_tab_variables(messages):
    turn = messages[-1]["content"]
    assert '  $mood (choice): "noir", "12\\" vinyl"' in turn
    assert '"Noir" is not an option of $mood; did you mean "noir"?' in turn


@pytest.mark.asyncio
async def test_the_streaming_send_shows_the_tab_variables_in_prompt_state():
    runtime, llm = _runtime()
    seen = {}

    async def _stream(**kwargs):
        seen["messages"] = kwargs["messages"]
        yield {"type": "token", "content": "ok"}

    llm.stream_with_history = Mock(side_effect=_stream)
    async for _ in runtime.send_message_stream(
        session_id="session-1", user_id="user-1", content="explain my condition", context_metadata=CONTEXT,
    ):
        pass
    _assert_prompt_state_carries_the_tab_variables(seen["messages"])


@pytest.mark.asyncio
async def test_the_buffered_send_shows_the_tab_variables_in_prompt_state():
    runtime, llm = _runtime()
    llm.generate_with_history = AsyncMock(return_value=Mock(
        content="ok", model="m", tokens_used=1, prompt_tokens=1, completion_tokens=1,
    ))
    await runtime.send_message(
        session_id="session-1", user_id="user-1", content="explain my condition", context_metadata=CONTEXT,
    )
    _assert_prompt_state_carries_the_tab_variables(llm.generate_with_history.call_args.kwargs["messages"])


async def _prompt_state_turn(runtime, llm, form_state):
    seen = {}

    async def _stream(**kwargs):
        seen["messages"] = kwargs["messages"]
        yield {"type": "token", "content": "ok"}

    llm.stream_with_history = Mock(side_effect=_stream)
    async for _ in runtime.send_message_stream(
        session_id="session-1", user_id="user-1", content="hi", context_metadata={**CONTEXT, "form_state": form_state},
    ):
        pass
    return seen["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_another_users_draft_on_the_same_tab_lends_no_variables():
    runtime, llm = _runtime(owner="user-2")
    turn = await _prompt_state_turn(runtime, llm, {"tab_id": TAB_ID})
    assert "PROMPT STATE" in turn
    assert "Variables (exact values for conditions)" not in turn
    assert "condition problem" not in turn


@pytest.mark.asyncio
async def test_an_unknown_tab_lends_no_variables():
    runtime, llm = _runtime()
    turn = await _prompt_state_turn(runtime, llm, {"tab_id": "tab-of-nobody"})
    assert "Variables (exact values for conditions)" not in turn
    assert "condition problem" not in turn

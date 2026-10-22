import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from src.features.chat.turns import ChatTurnRegistry
from tests.features.chat.test_tool_approval_flow import (
    CONTEXT,
    USER_ID,
    _call,
    _done,
    _manager,
    _resolve,
    _ScriptedLLM,
    _SetValueTool,
    _stream,
    chat_db,
)

SECRET = "SECRET-PLAN-7731"



class _ThinkingLLM(_ScriptedLLM):
    def __init__(self, turns=None, show_thinking=True):
        super().__init__(turns)
        self.repository.get_configuration.return_value = SimpleNamespace(
            provider_options={}, type="openai", supports_vision=True, show_thinking=show_thinking,
        )

    def stream_with_tools(self, **kwargs):
        turn = self.turns[0] if self.turns else {}
        thinking = turn.get("thinking")
        inner = super().stream_with_tools(**kwargs)

        async def gen():
            for start in range(0, len(thinking or ""), 2000):
                yield {"type": "thinking", "content": thinking[start:start + 2000]}
            async for event in inner:
                yield event

        return gen()

    def set_knob(self, shown):
        self.repository.get_configuration.return_value.show_thinking = shown


async def _paused_with_thinking(show_thinking=True):
    llm = _ThinkingLLM(
        [
            {"thinking": SECRET, "tool_calls": [_call("c1", "set_value", value=2)]},
            {"content": "Set to 2."},
        ],
        show_thinking=show_thinking,
    )
    manager, repo, session_id = _manager([_SetValueTool()], llm)
    events = await _stream(manager, session_id, "set it to 2")
    return manager, repo, session_id, llm, _done(events)["assistant_message"]["id"]


def _read(manager, session_id):
    return manager.get_session(session_id, USER_ID).messages[-1].metadata


class TestStoredWithTheMessage:
    @pytest.mark.asyncio
    async def test_a_paused_turn_stores_the_reasoning_before_its_tool_call(self, chat_db):
        _m, repo, _sid, _llm, message_id = await _paused_with_thinking()

        segments = repo.get_message(message_id).metadata["thinking"]["segments"]

        assert [s["text"] for s in segments] == [SECRET]

    @pytest.mark.asyncio
    async def test_nothing_is_stored_while_the_knob_is_off(self, chat_db):
        _m, repo, _sid, _llm, message_id = await _paused_with_thinking(show_thinking=False)

        assert "thinking" not in repo.get_message(message_id).metadata

    @pytest.mark.asyncio
    async def test_reopening_shows_it_while_the_knob_is_on(self, chat_db):
        manager, _repo, session_id, _llm, _mid = await _paused_with_thinking()

        assert _read(manager, session_id)["thinking"]["segments"][0]["text"] == SECRET

    @pytest.mark.asyncio
    async def test_reopening_hides_it_when_the_knob_is_turned_off_and_restores_it_on_again(self, chat_db):
        manager, repo, session_id, llm, message_id = await _paused_with_thinking()

        llm.set_knob(False)
        hidden = _read(manager, session_id)
        llm.set_knob(True)
        shown = _read(manager, session_id)

        assert "thinking" not in hidden
        assert shown["thinking"]["segments"][0]["text"] == SECRET
        assert repo.get_message(message_id).metadata["thinking"]["segments"][0]["text"] == SECRET

    @pytest.mark.asyncio
    async def test_resolving_the_tool_keeps_the_stored_reasoning(self, chat_db):
        manager, repo, session_id, llm, message_id = await _paused_with_thinking()
        llm.set_knob(False)

        await _resolve(manager, session_id, message_id, approved=False, reason="no")

        assert repo.get_message(message_id).metadata["thinking"]["segments"][0]["text"] == SECRET

    @pytest.mark.asyncio
    async def test_a_message_page_is_gated_like_the_session(self, chat_db):
        manager, _repo, session_id, llm, _mid = await _paused_with_thinking()

        llm.set_knob(False)
        hidden, _ = manager.get_session_messages(session_id, USER_ID)
        llm.set_knob(True)
        shown, _ = manager.get_session_messages(session_id, USER_ID)

        assert "thinking" not in hidden[-1].metadata
        assert "thinking" in shown[-1].metadata


class TestNeverBackInTheContext:
    @pytest.mark.asyncio
    async def test_the_approval_continuation_does_not_replay_it(self, chat_db):
        manager, _repo, session_id, llm, message_id = await _paused_with_thinking()
        await _resolve(manager, session_id, message_id, approved=False, reason="no")

        await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        assert SECRET not in json.dumps(llm.calls[-1], default=str)

    @pytest.mark.asyncio
    async def test_the_next_user_turn_does_not_replay_it(self, chat_db):
        llm = _ThinkingLLM([{"thinking": SECRET, "content": "first answer"}, {"content": "second answer"}])
        manager, _repo, session_id = _manager([_SetValueTool()], llm)
        await _stream(manager, session_id, "hello")

        await _stream(manager, session_id, "and then?")

        assert SECRET not in json.dumps(llm.calls[-1], default=str)
        assert "first answer" in json.dumps(llm.calls[-1], default=str)


class TestDoneLeavesTheThinkingToTheStream:
    LONG = "a long chain of reasoning. " * 3000

    async def _long_turn(self):
        llm = _ThinkingLLM([{"thinking": self.LONG, "content": "the answer"}])
        manager, repo, session_id = _manager([_SetValueTool()], llm)
        registry = ChatTurnRegistry()
        turn = registry.start(
            session_id,
            USER_ID,
            lambda: manager.send_message_stream(
                session_id=session_id, user_id=USER_ID, content="hi", context_metadata=copy.deepcopy(CONTEXT),
            ),
        )
        await asyncio.wait_for(turn.done.wait(), timeout=10)
        return repo, session_id, turn

    @pytest.mark.asyncio
    async def test_the_done_payload_has_no_thinking_and_stays_under_the_retention_cap(self, chat_db):
        _repo, _sid, turn = await self._long_turn()

        done = next(e for e in turn.events if e["event"] == "done")

        assert not done["data"].get("truncated")
        assert "thinking" not in done["data"]["assistant_message"]["metadata"]
        assert len(json.dumps(done)) < 16 * 1024

    @pytest.mark.asyncio
    async def test_the_stored_row_keeps_the_whole_reasoning(self, chat_db):
        repo, session_id, turn = await self._long_turn()

        done = next(e for e in turn.events if e["event"] == "done")
        stored = repo.get_message(done["data"]["assistant_message"]["id"]).metadata["thinking"]["segments"]

        assert [s["text"] for s in stored] == [self.LONG]

    @pytest.mark.asyncio
    async def test_the_thinking_still_streams_in_pieces_before_done(self, chat_db):
        _repo, _sid, turn = await self._long_turn()

        pieces = [
            e["data"]["content"] for e in turn.events if e["event"] == "thinking" and not e["data"].get("truncated")
        ]

        assert len(pieces) > 1
        assert "".join(pieces) == self.LONG

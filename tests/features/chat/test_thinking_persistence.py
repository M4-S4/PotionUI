import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from src.features.chat.turns import ChatTurnRegistry
from tests.fixtures.chat_engine import scripted_llm, set_config
from tests.features.chat.test_tool_approval_flow import (
    CONTEXT,
    USER_ID,
    _call,
    _done,
    _EchoTool,
    _manager,
    _resolve,
    _SetValueTool,
    _stream,
    chat_db,
)

SECRET = "SECRET-PLAN-7731"



def _ThinkingLLM(turns=None, show_thinking=True):
    return scripted_llm(turns or [], show_thinking=show_thinking)


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
    return manager.get_session(session_id, USER_ID).messages[-1]


def _stored_reasoning(manager, session_id):
    return [
        part.text
        for step in manager.turn_engine.store.steps(session_id)
        for part in step.message.reasoning()
    ]


class TestProjectedFromTheSteps:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("show_thinking", [True, False])
    async def test_a_paused_turn_stores_the_reasoning_before_its_tool_call(self, chat_db, show_thinking):
        manager, _repo, session_id, _llm, _mid = await _paused_with_thinking(show_thinking)

        assert _stored_reasoning(manager, session_id) == [SECRET]

    @pytest.mark.asyncio
    async def test_reopening_shows_it_while_the_knob_is_on(self, chat_db):
        manager, _repo, session_id, _llm, _mid = await _paused_with_thinking()

        [block] = _read(manager, session_id).reasoning

        assert (block.kind, block.text) == ("raw", SECRET)

    @pytest.mark.asyncio
    async def test_reopening_hides_it_when_the_knob_is_turned_off_and_restores_it_on_again(self, chat_db):
        manager, _repo, session_id, llm, _message_id = await _paused_with_thinking()

        set_config(llm, show_thinking=False)
        hidden = _read(manager, session_id)
        set_config(llm, show_thinking=True)
        shown = _read(manager, session_id)

        assert hidden.reasoning is None
        assert SECRET not in hidden.model_dump_json()
        assert [block.text for block in shown.reasoning] == [SECRET]

    @pytest.mark.asyncio
    async def test_resolving_the_tool_keeps_the_stored_reasoning(self, chat_db):
        manager, _repo, session_id, llm, message_id = await _paused_with_thinking()
        set_config(llm, show_thinking=False)

        await _resolve(manager, session_id, message_id, approved=False, reason="no")

        assert _stored_reasoning(manager, session_id) == [SECRET]

    @pytest.mark.asyncio
    async def test_a_message_page_is_gated_like_the_session(self, chat_db):
        manager, _repo, session_id, llm, _mid = await _paused_with_thinking()

        set_config(llm, show_thinking=False)
        hidden, _ = manager.get_session_messages(session_id, USER_ID)
        set_config(llm, show_thinking=True)
        shown, _ = manager.get_session_messages(session_id, USER_ID)

        assert hidden[-1].reasoning is None
        assert [block.text for block in shown[-1].reasoning] == [SECRET]

    @pytest.mark.asyncio
    async def test_the_stored_row_carries_no_reasoning_copy(self, chat_db):
        _m, repo, _sid, _llm, message_id = await _paused_with_thinking()

        metadata = repo.get_message(message_id).metadata

        assert "thinking" not in metadata
        assert "tool_transcript" not in metadata
        assert SECRET not in json.dumps(metadata)


def _tool_call_messages(call):
    return [message for message in call["messages"] if message.get("tool_calls")]


class TestBackInTheContextWithinTheTurn:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("show_thinking", [True, False])
    @pytest.mark.parametrize("approved", [True, False])
    async def test_the_approval_continuation_sends_it_back_on_the_paused_tool_call(
        self, chat_db, show_thinking, approved,
    ):
        manager, _repo, session_id, llm, message_id = await _paused_with_thinking(show_thinking)
        await _resolve(manager, session_id, message_id, approved=approved, reason=None if approved else "no")

        await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        assert _tool_call_messages(llm.adapter.calls[0]) == []
        (sent,) = _tool_call_messages(llm.adapter.calls[-1])
        assert sent["reasoning"] == SECRET
        assert sent["tool_calls"][0]["id"] == "c1"

    @pytest.mark.asyncio
    async def test_a_turn_paused_twice_sends_back_the_reasoning_of_both_pauses(self, chat_db):
        llm = _ThinkingLLM([
            {"thinking": "R1", "tool_calls": [_call("c1", "set_value", value=2)]},
            {"thinking": "R2", "tool_calls": [_call("c2", "set_value", value=3)]},
            {"content": "Both set."},
        ])
        manager, _repo, session_id = _manager([_SetValueTool()], llm)
        first = _done(await _stream(manager, session_id, "set it twice"))["assistant_message"]["id"]
        await _resolve(manager, session_id, first)
        second = _done(
            await _stream(manager, session_id, continue_from={"message_id": first, "tool_index": 0})
        )["assistant_message"]["id"]
        await _resolve(manager, session_id, second)

        await _stream(manager, session_id, continue_from={"message_id": second, "tool_index": 0})

        assert [m["reasoning"] for m in _tool_call_messages(llm.adapter.calls[-1])] == ["R1", "R2"]

    @pytest.mark.asyncio
    async def test_an_earlier_turn_stays_without_its_reasoning_while_the_continued_turn_has_it(self, chat_db):
        llm = _ThinkingLLM([
            {"thinking": "EARLIER-SECRET", "tool_calls": [_call("e1", "echo")]},
            {"content": "first answer"},
            {"thinking": SECRET, "tool_calls": [_call("c1", "set_value", value=2)]},
            {"content": "Set to 2."},
        ])
        manager, _repo, session_id = _manager([_SetValueTool(), _EchoTool()], llm)
        await _stream(manager, session_id, "look around")
        message_id = _done(await _stream(manager, session_id, "set it to 2"))["assistant_message"]["id"]
        await _resolve(manager, session_id, message_id)

        await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        earlier, current = _tool_call_messages(llm.adapter.calls[-1])
        assert earlier["tool_calls"][0]["id"] == "e1"
        assert "reasoning" not in earlier
        assert current["reasoning"] == SECRET
        assert "EARLIER-SECRET" not in json.dumps(llm.adapter.calls[-1], default=str)

    @pytest.mark.asyncio
    async def test_the_next_user_turn_does_not_replay_it(self, chat_db):
        llm = _ThinkingLLM([{"thinking": SECRET, "content": "first answer"}, {"content": "second answer"}])
        manager, _repo, session_id = _manager([_SetValueTool()], llm)
        await _stream(manager, session_id, "hello")

        await _stream(manager, session_id, "and then?")

        assert SECRET not in json.dumps(llm.adapter.calls[-1], default=str)
        assert "first answer" in json.dumps(llm.adapter.calls[-1], default=str)

    @pytest.mark.asyncio
    async def test_a_reply_to_a_paused_turn_continues_it_with_its_reasoning(self, chat_db):
        manager, _repo, session_id, llm, _message_id = await _paused_with_thinking()

        await _stream(manager, session_id, "never mind, do something else")

        (sent,) = _tool_call_messages(llm.adapter.calls[-1])
        assert sent["reasoning"] == SECRET


class TestNeverShownThroughTheMetadata:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("show_thinking", [True, False])
    async def test_reopening_the_session_leaves_it_out_of_the_metadata(self, chat_db, show_thinking):
        manager, _repo, session_id, _llm, _mid = await _paused_with_thinking(show_thinking)

        metadata = _read(manager, session_id).metadata

        assert "tool_transcript" not in metadata and "thinking" not in metadata
        assert SECRET not in json.dumps(metadata)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("show_thinking", [True, False])
    async def test_the_done_event_leaves_the_reasoning_to_the_stream(self, chat_db, show_thinking):
        llm = _ThinkingLLM(
            [{"thinking": SECRET, "tool_calls": [_call("c1", "set_value", value=2)]}], show_thinking=show_thinking,
        )
        manager, _repo, session_id = _manager([_SetValueTool()], llm)

        message = _done(await _stream(manager, session_id, "set it to 2"))["assistant_message"]

        assert message["metadata"]["tool_executions"][0]["tool_call_id"] == "c1"
        assert "reasoning" not in message
        assert SECRET not in json.dumps(message)

    @pytest.mark.asyncio
    async def test_a_long_reasoning_chain_does_not_inflate_the_done_payload(self, chat_db):
        long_chain = "a long chain of reasoning. " * 3000
        llm = _ThinkingLLM([{"thinking": long_chain, "tool_calls": [_call("c1", "set_value", value=2)]}])
        manager, _repo, session_id = _manager([_SetValueTool()], llm)

        events = await _stream(manager, session_id, "set it to 2")

        assert len(json.dumps(_done(events))) < 16 * 1024

    @pytest.mark.asyncio
    async def test_the_buffered_send_leaves_it_out_but_stores_it(self, chat_db):
        llm = _ThinkingLLM([
            {"thinking": SECRET, "tool_calls": [_call("e1", "echo")]},
            {"content": "done"},
        ])
        manager, _repo, session_id = _manager([_EchoTool()], llm)

        response = await manager.send_message(
            session_id=session_id, user_id=USER_ID, content="look around", context_metadata=copy.deepcopy(CONTEXT),
        )

        assert response.assistant_message.reasoning is None
        assert SECRET not in response.assistant_message.model_dump_json()
        assert _stored_reasoning(manager, session_id) == [SECRET]


class TestDoneLeavesTheThinkingToTheStream:
    LONG = "a long chain of reasoning. " * 3000

    async def _long_turn(self):
        llm = _ThinkingLLM([{"thinking": self.LONG, "content": "the answer"}])
        manager, repo, session_id = _manager([_SetValueTool()], llm)
        self.store = manager.turn_engine.store
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
    async def test_the_done_payload_has_no_reasoning_and_stays_under_the_retention_cap(self, chat_db):
        _repo, _sid, turn = await self._long_turn()

        done = next(e for e in turn.events if e["event"] == "done")

        assert not done["data"].get("truncated")
        assert "reasoning" not in done["data"]["assistant_message"]
        assert len(json.dumps(done)) < 16 * 1024

    @pytest.mark.asyncio
    async def test_the_stored_steps_keep_the_whole_reasoning(self, chat_db):
        repo, session_id, turn = await self._long_turn()

        done = next(e for e in turn.events if e["event"] == "done")
        row = repo.get_message(done["data"]["assistant_message"]["id"])
        steps = self.store.turn_steps(row.turn_id)

        assert [part.text for step in steps for part in step.message.reasoning()] == [self.LONG]

    @pytest.mark.asyncio
    async def test_the_thinking_still_streams_in_pieces_before_done(self, chat_db):
        _repo, _sid, turn = await self._long_turn()

        pieces = [
            e["data"]["text"] for e in turn.events if e["event"] == "reasoning" and not e["data"].get("truncated")
        ]

        assert len(pieces) > 1
        assert "".join(pieces) == self.LONG

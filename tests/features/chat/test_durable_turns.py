import asyncio
import json
from typing import Any, List
from unittest.mock import patch

import pytest

from src.features.chat.history import replayed_history
from src.features.chat.exceptions import ToolExecutionPendingException
from src.features.llm.harness import codec
from src.features.llm.harness.adapter import RenderRequest, Settings
from src.features.llm.harness.adapters.openai_chat import OpenAIChatAdapter
from src.features.llm.harness.baseline import turn_only_replay
from src.features.llm.harness.errors import ServerFailed
from src.features.llm.harness.events import TextDelta
from src.features.llm.harness.messages import ImagePart, Origin, Role
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.harness.store import SqliteTurnStore
from src.features.llm.harness.turn_state import StepStatus, TurnState, TurnStatus
from src.features.llm.tools.base import BaseTool, ToolResult
from tests.features.chat.test_tool_approval_flow import (
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
from tests.fixtures.chat_engine import ScriptedTurns, _Replay, scripted_llm
from tests.fixtures.legacy_chats import chats, env, execution, previous_template
from tests.fixtures.llm_engine import scripted_config

SECRET = "SECRET-PLAN-7731"
ORIGIN = Origin(adapter="scripted", provider="scripted", model="scripted-model")


class _SetOtherTool(_SetValueTool):
    @property
    def name(self) -> str:
        return "set_other"


class _BrokenTool(BaseTool):
    modes = ["generation"]

    def __init__(self):
        self.calls = 0

    @property
    def name(self) -> str:
        return "broken"

    @property
    def description(self) -> str:
        return "Always fails."

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, context, **kwargs) -> ToolResult:
        self.calls += 1
        return ToolResult(success=False, data="", error="disk on fire")


class _PartialDecoder:
    def __init__(self, partial: str):
        self.partial = partial

    def feed(self, chunk: bytes):
        return [TextDelta(block=1, text=self.partial)]

    def close(self):
        return []


class _FailingMidStream(ScriptedTurns):
    def __init__(self, failure: BaseException, partial: str, *turns: Any):
        super().__init__([{"content": "unused"}, *turns])
        self.failure = failure
        self.partial = partial
        self.failing = True

    def decoder(self, request, profile):
        if self.failing:
            return _PartialDecoder(self.partial)
        return super().decoder(request, profile)

    async def send(self, wire, request):
        if not self.failing:
            async for chunk in super().send(wire, request):
                yield chunk
            return
        self.failing = False
        yield b"partial"
        raise self.failure


def _failing_llm(failure: BaseException, *turns: Any):
    from tests.fixtures.llm_engine import scripted_gateway

    adapter = _FailingMidStream(failure, "half an ans", *turns)
    gateway = scripted_gateway(adapter, scripted_config(id="llm-1", supports_vision=True))
    gateway.adapter = adapter
    return gateway


async def _flow(chat_db, monkeypatch):
    store = SqliteTurnStore()
    broken = _BrokenTool()
    setter = _SetValueTool()
    llm = scripted_llm(
        [
            {
                "thinking": SECRET,
                "tool_calls": [_call("c0", "broken"), _call("c1", "echo"), _call("c2", "set_value", value=2)],
            },
            {"tool_calls": [_call("c3", "broken")]},
            {"content": "All done."},
        ],
        show_thinking=True,
    )
    manager, repo, session_id = _manager([broken, _EchoTool(), setter], llm, store=store)
    monkeypatch.setattr("src.features.chat.conversation.convert_image_to_base64", lambda data: "IMG" if data else None)
    first = await _stream(manager, session_id, "do three things", image_data="shot.png")
    message_id = _done(first)["assistant_message"]["id"]
    await _resolve(manager, session_id, message_id, tool_index=2)
    final = await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 2})
    return manager, repo, session_id, llm, store, broken, message_id, final


class TestPauseResolveContinue:
    @pytest.mark.asyncio
    async def test_nothing_is_lost_across_the_pause(self, chat_db, monkeypatch):
        _manager_, repo, session_id, llm, store, broken, message_id, final = await _flow(chat_db, monkeypatch)

        paused_request, resumed_request, last_request = llm.adapter.requests
        user = resumed_request.messages[0]
        assert [part.data for part in user.parts if isinstance(part, ImagePart)] == ["IMG"]
        asks = resumed_request.messages[1]
        assert [part.text for part in asks.reasoning()] == [SECRET]
        assert [call.call_id for call in asks.tool_calls()] == ["c0", "c1", "c2"]
        tool = resumed_request.messages[2]
        assert tool.role is Role.TOOL
        assert [(part.call_id, part.is_error) for part in tool.tool_results()] == [("c0", True), ("c1", False), ("c2", False)]
        assert tool.tool_results()[2].text() == json.dumps({"applied": 2})
        assert "disk on fire" in tool.tool_results()[0].text()
        assert len(paused_request.messages) == 1

        assert broken.calls == 1
        refused = last_request.messages[-2].tool_results()[0]
        assert refused.is_error and "already called broken" in refused.text()
        assert _done(final)["assistant_message"]["content"] == "All done."

    @pytest.mark.asyncio
    async def test_the_resumed_turn_keeps_one_turn_id_and_numbers_steps_without_gaps(self, chat_db, monkeypatch):
        _manager_, repo, session_id, _llm, store, _broken, message_id, final = await _flow(chat_db, monkeypatch)

        steps = store.steps(session_id)
        assert [step.seq for step in steps] == list(range(len(steps)))
        assert len({step.turn_id for step in steps}) == 1
        assert [step.message.role for step in steps] == [
            Role.USER, Role.ASSISTANT, Role.TOOL, Role.USER, Role.ASSISTANT, Role.TOOL, Role.USER, Role.ASSISTANT,
        ]
        state = store.load_state(steps[0].turn_id)
        assert state.status is TurnStatus.COMPLETED and state.rounds_used == 3
        assert set(state.failed_calls) == {"broken:{}"}

    @pytest.mark.asyncio
    async def test_the_second_row_covers_the_steps_after_the_pause(self, chat_db, monkeypatch):
        _manager_, repo, session_id, _llm, store, _broken, message_id, final = await _flow(chat_db, monkeypatch)

        rows = [row for row in repo.get_messages(session_id) if row.role == "assistant"]
        windows = [row.metadata["step_range"] for row in rows]
        assert windows == [{"first": 0, "last": 1}, {"first": 2, "last": 7}]
        assert rows[1].metadata["continued_from"] == {"message_id": message_id, "tool_index": 2}
        assert {row.turn_id for row in rows} == {store.steps(session_id)[0].turn_id}

    @pytest.mark.asyncio
    async def test_rerendering_the_stored_history_reproduces_every_request(self, chat_db, monkeypatch):
        _manager_, _repo, session_id, llm, _store, _broken, _message_id, _final = await _flow(chat_db, monkeypatch)
        fresh = SqliteTurnStore()
        stored = [step.message for step in fresh.steps(session_id)]
        adapter = OpenAIChatAdapter()
        profile = CapabilityProfile(adapter="openai_chat", model="scripted-model")
        config = scripted_config(id="llm-1", type="openai-compatible")

        for request in llm.adapter.requests:
            expected = list(stored[:len(request.messages)])
            assert expected == list(request.messages)
            original = adapter.render(
                RenderRequest(config=config, system=request.system, messages=request.messages, tools=request.tools, settings=Settings()),
                profile,
            )
            replayed = adapter.render(
                RenderRequest(config=config, system=request.system, messages=tuple(expected), tools=request.tools, settings=Settings()),
                profile,
            )
            assert json.dumps(original.body, sort_keys=True) == json.dumps(replayed.body, sort_keys=True)

    @pytest.mark.asyncio
    async def test_the_stored_state_round_trips_the_contract(self, chat_db, monkeypatch):
        _manager_, _repo, session_id, _llm, store, _broken, _message_id, _final = await _flow(chat_db, monkeypatch)

        turn_id = store.steps(session_id)[0].turn_id
        state = SqliteTurnStore().load_state(turn_id)

        assert codec.dumps(codec.loads(TurnState, codec.dumps(state))) == codec.dumps(state)


class TestPausedTwice:
    @pytest.mark.asyncio
    async def test_a_second_pause_in_the_same_turn_resumes_again(self, chat_db):
        store = SqliteTurnStore()
        llm = scripted_llm(
            [
                {"thinking": "R1", "tool_calls": [_call("c1", "set_value", value=2)]},
                {"thinking": "R2", "tool_calls": [_call("c2", "set_value", value=3)]},
                {"content": "Both set."},
            ],
            show_thinking=True,
        )
        tool = _SetValueTool()
        manager, repo, session_id = _manager([tool], llm, store=store)
        first = _done(await _stream(manager, session_id, "set twice"))["assistant_message"]["id"]
        await _resolve(manager, session_id, first)
        second = _done(
            await _stream(manager, session_id, continue_from={"message_id": first, "tool_index": 0})
        )["assistant_message"]["id"]

        paused = store.open_turn(session_id)
        assert paused.status is TurnStatus.AWAITING_DECISION
        assert [call.call_id for call in paused.pending] == ["c2"]
        assert paused.rounds_used == 2

        await _resolve(manager, session_id, second)
        final = await _stream(manager, session_id, continue_from={"message_id": second, "tool_index": 0})

        assert tool.confirmed == 2
        assert _done(final)["assistant_message"]["content"] == "Both set."
        assert store.open_turn(session_id) is None
        steps = store.steps(session_id)
        assert [part.text for step in steps for part in step.message.reasoning()] == ["R1", "R2"]
        last = llm.adapter.requests[-1].messages
        assert [message.role for message in last] == [
            Role.USER, Role.ASSISTANT, Role.TOOL, Role.USER, Role.ASSISTANT, Role.TOOL, Role.USER,
        ]
        assert [[part.text for part in message.reasoning()] for message in last if message.role is Role.ASSISTANT] == [["R1"], ["R2"]]

    @pytest.mark.asyncio
    async def test_a_reply_instead_of_deciding_closes_the_call_and_continues_the_turn(self, chat_db):
        store = SqliteTurnStore()
        llm = scripted_llm([{"tool_calls": [_call("c1", "set_value", value=2)]}, {"content": "Okay, 3."}])
        tool = _SetValueTool()
        manager, _repo, session_id = _manager([tool], llm, store=store)
        await _stream(manager, session_id, "set it to 2")

        await _stream(manager, session_id, "no, 3")

        steps = store.steps(session_id)
        assert len({step.turn_id for step in steps}) == 1
        assert [step.message.role for step in steps] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.USER, Role.ASSISTANT]
        assert steps[2].message.tool_results()[0].text() == "User rejected this. Reason: no, 3"
        assert tool.confirmed == 0
        assert store.open_turn(session_id) is None


class TestSeveralApprovals:
    @pytest.mark.asyncio
    async def test_the_turn_continues_only_after_every_call_is_decided_and_answers_them_in_call_order(self, chat_db):
        store = SqliteTurnStore()
        llm = scripted_llm([
            {"tool_calls": [_call("c1", "set_value", value=2), _call("c2", "set_other", value=5)]},
            {"content": "Both applied."},
        ])
        manager, _repo, session_id = _manager([_SetValueTool(), _SetOtherTool()], llm, store=store)
        message_id = _done(await _stream(manager, session_id, "set both"))["assistant_message"]["id"]
        assert [call.call_id for call in store.open_turn(session_id).pending] == ["c1", "c2"]

        await _resolve(manager, session_id, message_id, tool_index=1)
        with pytest.raises(ToolExecutionPendingException):
            await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 1})

        await _resolve(manager, session_id, message_id, tool_index=0, approved=False, reason="no")
        final = await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        (tool_message,) = [step.message for step in store.steps(session_id) if step.message.role is Role.TOOL]
        assert [(part.call_id, part.text()) for part in tool_message.tool_results()] == [
            ("c1", "User rejected this. Reason: no"), ("c2", json.dumps({"applied": 5})),
        ]
        assert _done(final)["assistant_message"]["content"] == "Both applied."


class TestInterruptedTurns:
    @pytest.mark.asyncio
    async def test_a_provider_error_mid_stream_keeps_the_partial_step(self, chat_db):
        store = SqliteTurnStore()
        llm = _failing_llm(ServerFailed("upstream down"), {"content": "Recovered."})
        manager, _repo, session_id = _manager([_EchoTool()], llm, store=store)

        events = await _stream(manager, session_id, "hello")

        assert any(event["event"] == "error" for event in events)
        steps = store.steps(session_id)
        assert [(step.message.role, step.status) for step in steps] == [
            (Role.USER, StepStatus.COMPLETED), (Role.ASSISTANT, StepStatus.FAILED),
        ]
        assert steps[1].message.text() == "half an ans"
        assert store.load_state(steps[0].turn_id).status is TurnStatus.FAILED

        recovered = await _stream(manager, session_id, "try again")

        assert _done(recovered)["assistant_message"]["content"] == "Recovered."
        sent = llm.adapter.requests[-1].messages
        assert [message.role for message in sent] == [Role.USER, Role.ASSISTANT, Role.USER]
        assert sent[1].text() == "half an ans"

    @pytest.mark.asyncio
    async def test_cancelling_mid_stream_keeps_the_partial_step_and_closes_the_turn(self, chat_db):
        store = SqliteTurnStore()
        llm = _failing_llm(asyncio.CancelledError(), {"content": "unused"})
        manager, _repo, session_id = _manager([_EchoTool()], llm, store=store)

        with pytest.raises(asyncio.CancelledError):
            await _stream(manager, session_id, "hello")

        steps = store.steps(session_id)
        assert [(step.message.role, step.status) for step in steps] == [
            (Role.USER, StepStatus.COMPLETED), (Role.ASSISTANT, StepStatus.CANCELLED),
        ]
        assert steps[1].message.text() == "half an ans"
        assert store.load_state(steps[0].turn_id).status is TurnStatus.CANCELLED
        assert store.open_turn(session_id) is None

    @pytest.mark.asyncio
    async def test_cancelling_a_resumed_turn_closes_its_open_call_so_the_history_stays_valid(self, chat_db):
        store = SqliteTurnStore()
        llm = scripted_llm([{"tool_calls": [_call("c1", "set_value", value=2)]}])
        llm.adapter.turns.append(ServerFailed("down"))
        tool = _SetValueTool()
        manager, _repo, session_id = _manager([tool], llm, store=store)
        message_id = _done(await _stream(manager, session_id, "set it"))["assistant_message"]["id"]
        await _resolve(manager, session_id, message_id)

        events = await _stream(manager, session_id, continue_from={"message_id": message_id, "tool_index": 0})

        assert any(event["event"] == "error" for event in events)
        steps = store.steps(session_id)
        assert [step.message.role for step in steps] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.USER]
        assert steps[2].message.tool_results()[0].text() == json.dumps({"applied": 2})
        assert store.load_state(steps[0].turn_id).status is TurnStatus.FAILED


class TestMigratedChatsReplay:
    def test_a_legacy_tool_turn_replays_as_a_valid_history(self, env, chats):
        _, migration = env
        session = chats.session("s1")
        chats.row(session, "user", "find cats")
        chats.row(session, "assistant", "Found 2 cats.", {
            "thinking": {"segments": [{"text": "plan", "duration_ms": 9}, {"text": "sum up", "duration_ms": 4}]},
            "tool_transcript": [{
                "calls": [{"id": "call_1", "name": "search", "arguments": {"q": "cats"}}],
                "results": [{"tool_call_id": "call_1", "content": "2 hits"}],
                "reasoning": "plan",
            }],
            "tool_executions": [execution("search", "call_1", {"q": "cats"}, data="2 hits")],
        })
        migration.up()
        steps = chats.steps(session)
        target = Origin(adapter="a", provider="p", model="m")

        messages, _ = turn_only_replay(
            steps, turn_id="another-turn", target=target, profile=CapabilityProfile(adapter="a", model="m"),
        )

        assert [message.role for message in messages] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
        call = messages[1].tool_calls()[0]
        assert (call.call_id, call.args) == ("call_1", {"q": "cats"})
        assert messages[2].tool_results()[0].call_id == "call_1"
        assert messages[3].text() == "Found 2 cats."
        assert not any(message.reasoning() for message in messages)

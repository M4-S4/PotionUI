import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.features.chat.dto import MessageResponse, SessionResponse
from src.features.llm.harness.baseline import InMemoryTurnStore
from src.features.llm.harness.messages import Message, Origin, ReasoningPart, Role, TextPart
from src.features.llm.harness.turn_state import StepRecord, StepStatus
from src.features.chat.routes import ChatController
from src.features.chat.runtime import ChatRuntime
from src.features.chat.turns import ChatTurnRegistry
from src.platform.security.user import User
from tests.fixtures.chat_modes import mode_registry

STORED = {"segments": [{"text": "weighing options", "duration_ms": 1200}]}
TURN = "t1"


class Knob:
    def __init__(self, shown):
        self.shown = shown
        self.lookups = []

    def get_configuration(self, config_id):
        self.lookups.append(config_id)
        return SimpleNamespace(show_thinking=self.shown)


def assistant(msg_id="a1", **metadata):
    return MessageResponse(
        id=msg_id, session_id="s1", role="assistant", content="the answer",
        metadata={"tokens_used": 7, **metadata},
    )


def stored_message(msg_id="a1"):
    return assistant(
        msg_id,
        thinking=STORED,
        behavior_trace={"version": 1, "thinking_visible": True},
    ).model_copy(update={"turn_id": TURN})


def projected_message(msg_id="a1"):
    return assistant(
        msg_id,
        step_range={"first": 0, "last": 0},
        behavior_trace={"version": 1, "thinking_visible": True},
    ).model_copy(update={"turn_id": TURN})


@pytest.fixture
def store():
    store = InMemoryTurnStore()
    store.append(StepRecord(
        id="s0", session_id="s1", turn_id=TURN, seq=0, status=StepStatus.COMPLETED,
        message=Message(
            id="m0", role=Role.ASSISTANT,
            parts=(ReasoningPart(text="weighing options", duration_ms=1200), TextPart(text="the answer")),
            origin=Origin(adapter="a", provider="p", model="m"),
        ),
    ))
    return store


def session(messages, llm_config_id="cfg-1"):
    return SessionResponse(
        id="s1", user_id="u1", mode="generation", status="active",
        llm_config_id=llm_config_id, messages=messages,
    )


@pytest.fixture
def knob():
    return Knob(shown=True)


@pytest.fixture
def repo():
    return Mock()


@pytest.fixture
def runtime(repo, knob, store):
    return ChatRuntime(
        chat_repository=repo,
        llm_service=SimpleNamespace(repository=knob),
        response_processor=Mock(),
        plugin_registry=Mock(),
        chat_mode_registry=mode_registry(),
        turn_engine=SimpleNamespace(store=store),
    )


def has_thinking(message):
    return bool(message.reasoning)


class TestSessionRead:
    def test_stored_thinking_is_returned_while_the_knob_is_on(self, runtime, repo):
        repo.get_session_with_messages.return_value = session([projected_message()])

        result = runtime.get_session("s1", "u1")

        [block] = result.messages[0].reasoning
        assert (block.kind, block.text, block.duration_ms) == ("raw", "weighing options", 1200)
        assert result.messages[0].metadata["behavior_trace"]["thinking_visible"] is True

    def test_a_legacy_thinking_copy_in_the_metadata_is_never_returned(self, runtime, repo):
        repo.get_session_with_messages.return_value = session([stored_message()])

        result = runtime.get_session("s1", "u1")

        assert "thinking" not in result.messages[0].metadata

    def test_it_is_hidden_while_the_knob_is_off(self, runtime, repo, knob):
        knob.shown = False
        repo.get_session_with_messages.return_value = session([projected_message()])

        result = runtime.get_session("s1", "u1")

        assert not has_thinking(result.messages[0])
        assert "thinking_visible" not in result.messages[0].metadata["behavior_trace"]
        assert result.messages[0].metadata["tokens_used"] == 7

    def test_it_reappears_when_the_knob_is_turned_back_on(self, runtime, repo, knob):
        repo.get_session_with_messages.return_value = session([projected_message()])

        knob.shown = False
        assert not has_thinking(runtime.get_session("s1", "u1").messages[0])

        knob.shown = True
        assert has_thinking(runtime.get_session("s1", "u1").messages[0])

    def test_the_stored_data_is_never_modified(self, runtime, repo, knob, store):
        knob.shown = False
        stored = session([stored_message()])
        repo.get_session_with_messages.return_value = stored

        runtime.get_session("s1", "u1")

        assert stored.messages[0].metadata["thinking"] == STORED
        assert [part.text for part in store.steps("s1")[0].message.reasoning()] == ["weighing options"]

    def test_the_tail_read_is_gated_too(self, runtime, repo, knob):
        knob.shown = False
        repo.get_session_with_messages_tail.return_value = session([projected_message()])

        result = runtime.get_session("s1", "u1", tail=20)

        assert not has_thinking(result.messages[0])

    def test_the_session_current_config_decides(self, runtime, repo, knob):
        repo.get_session_with_messages.return_value = session([stored_message()], llm_config_id="other")

        runtime.get_session("s1", "u1")

        assert knob.lookups == ["other"]

    def test_a_session_without_projected_turns_still_gates_by_the_knob(self, runtime, repo, knob):
        repo.get_session_with_messages.return_value = session([assistant()])

        runtime.get_session("s1", "u1")

        assert knob.lookups == ["cfg-1"]


class TestMessagePageRead:
    def test_the_tail_page_is_gated(self, runtime, repo, knob):
        knob.shown = False
        repo.get_session.return_value = session(None)
        repo.get_message_tail.return_value = ([projected_message()], True)

        messages, has_earlier = runtime.get_session_messages("s1", "u1")

        assert not has_thinking(messages[0])
        assert has_earlier is True

    def test_an_earlier_page_is_gated(self, runtime, repo, knob):
        knob.shown = False
        repo.get_session.return_value = session(None)
        repo.get_messages_before.return_value = ([projected_message()], False)

        messages, has_earlier = runtime.get_session_messages("s1", "u1", before="a9")

        assert not has_thinking(messages[0])
        assert has_earlier is False

    def test_a_page_shows_thinking_while_the_knob_is_on(self, runtime, repo):
        repo.get_session.return_value = session(None)
        repo.get_message_tail.return_value = ([projected_message()], False)

        messages, _ = runtime.get_session_messages("s1", "u1")

        assert has_thinking(messages[0])


class TestAdminDetailRead:
    def test_the_admin_viewer_follows_the_same_gate(self, runtime, repo, knob, monkeypatch):
        import src.features.llm.trace_repository as traces

        monkeypatch.setattr(traces.chat_call_trace_repository, "list_for_session", lambda session_id: [])
        knob.shown = False
        repo.get_session_with_messages.return_value = session([stored_message()])

        detail = runtime.get_admin_session_detail("s1")

        assert "thinking" not in detail["session"]["messages"][0]["metadata"]


class TestReattachRead:
    USER = User(
        id="u1", username="testuser", email="test@example.com",
        password_hash="hash", account_type="USER",
    )

    @staticmethod
    def _events():
        stored = {
            "tokens_used": 7,
            "thinking": STORED,
            "behavior_trace": {"version": 1, "thinking_visible": True},
        }
        return [
            {"event": "status", "data": {"step": "thinking", "state": "started", "detail": {"thinking_visible": True}}},
            {"event": "status", "data": {"step": "tools", "state": "started", "detail": {"thinking_visible": True, "count": 2}}},
            {"event": "reasoning", "data": {"block": 0, "kind": "raw", "text": "weighing"}},
            {"event": "token", "data": {"content": "the answer"}},
            {"event": "done", "data": {"assistant_message": {"id": "a1", "content": "the answer", "metadata": stored}}},
        ]

    @staticmethod
    def _parse(chunks):
        events = []
        for block in "".join(chunks).split("\n\n"):
            if not block.strip():
                continue
            lines = block.split("\n")
            if lines[0].startswith("id: "):
                lines = lines[1:]
            events.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
        return events

    async def _reattach(self, runtime, repo):
        repo.get_session_with_messages.return_value = session(None)
        controller = ChatController(chat_runtime=runtime, turn_registry=ChatTurnRegistry())
        events = self._events()

        async def factory():
            for event in events:
                yield event

        turn = controller.turn_registry.start("s1", "u1", factory)
        await turn.done.wait()

        async def read():
            response = await controller.reattach_stream("s1", self.USER)
            return self._parse([chunk async for chunk in response.body_iterator])

        return read, turn

    @pytest.mark.asyncio
    async def test_a_knob_turned_off_hides_replayed_thinking_everywhere(self, runtime, repo, knob):
        knob.shown = False
        read, _turn = await self._reattach(runtime, repo)

        replayed = await read()

        assert [name for name, _ in replayed] == ["status", "status", "token", "done"]
        first, second = replayed[0][1], replayed[1][1]
        assert "detail" not in first
        assert second["detail"] == {"count": 2}
        metadata = replayed[-1][1]["assistant_message"]["metadata"]
        assert metadata == {"tokens_used": 7, "behavior_trace": {"version": 1}}
        assert replayed[2][1]["content"] == "the answer"

    @pytest.mark.asyncio
    async def test_a_knob_that_is_on_replays_everything(self, runtime, repo):
        read, _turn = await self._reattach(runtime, repo)

        replayed = await read()

        assert [name for name, _ in replayed] == ["status", "status", "reasoning", "token", "done"]
        assert replayed[0][1]["detail"] == {"thinking_visible": True}
        metadata = replayed[-1][1]["assistant_message"]["metadata"]
        assert metadata["thinking"] == STORED
        assert metadata["behavior_trace"]["thinking_visible"] is True

    @pytest.mark.asyncio
    async def test_hiding_never_alters_the_retained_events_so_turning_it_on_restores_them(self, runtime, repo, knob):
        read, turn = await self._reattach(runtime, repo)

        knob.shown = False
        hidden = await read()
        knob.shown = True
        shown = await read()

        assert "reasoning" not in [name for name, _ in hidden]
        assert "reasoning" in [name for name, _ in shown]
        assert turn.events[-1]["data"]["assistant_message"]["metadata"]["thinking"] == STORED

    @pytest.mark.asyncio
    async def test_a_session_with_no_config_replays_without_thinking(self, runtime, repo):
        repo.get_session_with_messages.return_value = session(None, llm_config_id=None)
        controller = ChatController(chat_runtime=runtime, turn_registry=ChatTurnRegistry())

        async def factory():
            yield {"event": "reasoning", "data": {"block": 0, "kind": "raw", "text": "x"}}
            yield {"event": "done", "data": {}}

        turn = controller.turn_registry.start("s1", "u1", factory)
        await turn.done.wait()
        response = await controller.reattach_stream("s1", self.USER)
        replayed = self._parse([chunk async for chunk in response.body_iterator])

        assert [name for name, _ in replayed] == ["done"]

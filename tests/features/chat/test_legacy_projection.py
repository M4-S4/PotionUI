import pytest

from src.features.llm.harness.baseline import turn_only_replay
from src.features.llm.harness.messages import Origin, Role
from src.features.llm.harness.profile import CapabilityProfile

from src.features.chat import projection
from src.features.chat.dto import MessageResponse
from src.features.llm.harness import codec
from src.features.llm.harness.turn_state import TurnState
from tests.fixtures.legacy_chats import chats, env, execution, previous_template


def test_migrated_reasoning_reaches_the_api_only_through_the_projection(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "find cats")
    assistant = chats.row(session, "assistant", "Found 2 cats.", {
        "prompt_tokens": 100, "completion_tokens": 20,
        "thinking": {"segments": [{"text": "plan the search", "duration_ms": 900}, {"text": "summarise results", "duration_ms": 400}]},
        "tool_transcript": [{
            "calls": [{"id": "call_1", "name": "search", "arguments": {}}],
            "results": [{"tool_call_id": "call_1", "content": "2 hits"}],
            "reasoning": "plan the search",
        }],
        "tool_executions": [execution("search", "call_1", data="2 hits")],
    })
    migration.up()
    turn_id, metadata = chats.message(assistant)
    row = MessageResponse(id=assistant, session_id=session, role="assistant", content="Found 2 cats.", metadata=metadata, turn_id=turn_id)
    steps = {turn_id: chats.steps(session)}
    states = {turn_id: codec.loads(TurnState, chats.turns(session)[0]["state_json"])}

    shown = projection.present([row], steps, states, show_thinking=True)[0]
    hidden = projection.present([row], steps, states, show_thinking=False)[0]

    assert [(block.text, block.duration_ms) for block in shown.reasoning] == [("plan the search", 900), ("summarise results", 400)]
    assert hidden.reasoning is None
    assert "reasoning" not in hidden.model_dump()
    assert "thinking" not in shown.metadata and "thinking" not in hidden.metadata
    assert shown.usage.input_tokens == 100 and shown.usage.output_tokens == 20


def test_a_legacy_tool_turn_replays_as_a_valid_history(env, chats):
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

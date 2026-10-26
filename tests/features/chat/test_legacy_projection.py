import pytest

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

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from src.features.llm.clients.base import LLMResponse
from src.features.llm.tools.base import BaseTool, ToolContext, ToolResult
from src.features.llm.tools.executor import ToolExecutor
from src.features.llm.tools.registry import ToolRegistry


@dataclass
class Turn:
    content: str = ""
    thinking: List[str] = field(default_factory=list)
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def stream_events(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = [{"type": "thinking", "content": part} for part in self.thinking]
        if self.content:
            events.append({"type": "token", "content": self.content})
        if self.tool_calls:
            events.append({"type": "tool_calls", "tool_calls": copy.deepcopy(self.tool_calls)})
        events.append({"type": "usage", "tokens_used": 1, "prompt_tokens": 1, "completion_tokens": 1})
        return events


class ScriptedLLM:
    def __init__(self, turns: List[Turn], config_type: str):
        self._turns = list(turns)
        self._index = 0
        config = SimpleNamespace(type=config_type, provider_options={})
        self.repository = SimpleNamespace(get_configuration=lambda llm_id: config)

    def _next(self) -> Turn:
        turn = self._turns[self._index]
        self._index += 1
        return turn

    async def generate_with_tools(self, messages, llm_id, tools=None, image_data=None,
                                  custom_system_message=None, mode=None, options_override=None):
        turn = self._next()
        return LLMResponse(
            content=turn.content,
            model="model-1",
            provider_id="p",
            tool_calls=copy.deepcopy(turn.tool_calls) or [],
            thinking="".join(turn.thinking) or None,
        )

    def stream_with_tools(self, messages, llm_id, tools=None, image_data=None,
                          custom_system_message=None, mode=None, options_override=None):
        turn = self._next()

        async def _gen():
            for event in turn.stream_events():
                yield event

        return _gen()


class EchoTool(BaseTool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes a message back."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}

    async def execute(self, context: ToolContext, **kwargs) -> ToolResult:
        return ToolResult(success=True, data=f"echo: {kwargs.get('message', '')}")


def call(call_id: str, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


async def events_for(turns: List[Turn], config_type: str, **options: Any) -> List[Dict[str, Any]]:
    registry = ToolRegistry()
    registry.register(EchoTool())
    executor = ToolExecutor(tool_registry=registry, llm_service=ScriptedLLM(turns, config_type))
    return [
        event
        async for event in executor.execute_with_tools_stream(
            messages=[{"role": "user", "content": "hi"}],
            llm_id="model-1",
            system_message="sys",
            tool_context=ToolContext(user_id="u1"),
            **options,
        )
    ]


def kinds(events: List[Dict[str, Any]]) -> List[str]:
    return [event["type"] for event in events]


def thinking_text(events: List[Dict[str, Any]]) -> str:
    return "".join(e["data"]["content"] for e in events if e["type"] == "thinking")


ANSWER_AFTER_THOUGHT = [Turn(thinking=["let me ", "think"], content="the answer")]

TOOL_THEN_ANSWER = [
    Turn(thinking=["need the tool"], tool_calls=[call("call_1", "echo", {"message": "hi"})]),
    Turn(thinking=["now answer"], content="final"),
]


class TestLiveStream:
    @pytest.mark.asyncio
    async def test_thinking_deltas_surface_before_the_answer(self):
        events = await events_for(ANSWER_AFTER_THOUGHT, "ollama")

        assert kinds(events)[:3] == ["thinking", "thinking", "token"]
        assert thinking_text(events) == "let me think"

    @pytest.mark.asyncio
    async def test_thinking_never_reaches_the_answer_or_the_done_content(self):
        events = await events_for(ANSWER_AFTER_THOUGHT, "ollama")

        tokens = "".join(e["data"]["content"] for e in events if e["type"] == "token")
        done = next(e for e in events if e["type"] == "done")
        assert tokens == "the answer"
        assert done["data"]["full_content"] == "the answer"

    @pytest.mark.asyncio
    async def test_each_model_turn_of_a_tool_round_thinks_in_order(self):
        events = await events_for(TOOL_THEN_ANSWER, "ollama")

        order = kinds(events)
        thinking_at = [i for i, kind in enumerate(order) if kind == "thinking"]
        assert len(thinking_at) == 2
        assert thinking_at[0] < order.index("tool_start")
        assert thinking_at[1] > order.index("tool_end")
        assert thinking_text(events) == "need the toolnow answer"

    @pytest.mark.asyncio
    async def test_a_turn_with_no_reasoning_emits_none(self):
        events = await events_for([Turn(content="plain")], "ollama")

        assert "thinking" not in kinds(events)

    @pytest.mark.asyncio
    async def test_a_turn_with_reasoning_is_closed_by_an_end_marker(self):
        events = await events_for(ANSWER_AFTER_THOUGHT, "ollama")

        order = kinds(events)
        assert order.index("thinking_end") > max(i for i, kind in enumerate(order) if kind == "thinking")
        assert order.index("thinking_end") < order.index("done")

    @pytest.mark.asyncio
    async def test_a_turn_without_reasoning_has_no_end_marker(self):
        events = await events_for([Turn(content="plain")], "ollama")

        assert "thinking_end" not in kinds(events)

    @pytest.mark.asyncio
    async def test_the_end_marker_follows_a_held_tail_the_filter_releases_last(self):
        events = await events_for([Turn(thinking=["hm"], content="<think>plan <tool")], "ollama")

        order = kinds(events)
        tokens = "".join(e["data"]["content"] for e in events if e["type"] == "token")
        assert tokens == "<think>plan <tool"
        last_token = max(i for i, kind in enumerate(order) if kind == "token")
        assert order.index("thinking_end") > last_token

    @pytest.mark.asyncio
    async def test_each_tool_round_turn_is_closed_separately(self):
        events = await events_for(TOOL_THEN_ANSWER, "ollama")

        assert kinds(events).count("thinking_end") == 2

    @pytest.mark.asyncio
    async def test_the_reasoning_of_each_abandoned_empty_attempt_is_kept_as_its_own_block(self):
        events = await events_for(
            [Turn(thinking=["first "]), Turn(thinking=["second "]), Turn(thinking=["third"])],
            "ollama",
        )

        order = [kind for kind in kinds(events) if kind in ("thinking", "thinking_end", "token")]
        assert order == ["thinking", "thinking_end"] * 3
        assert thinking_text(events) == "first second third"
        done = next(e for e in events if e["type"] == "done")
        assert done["data"]["full_content"] == ""


class TestBudgetWrapUp:
    @pytest.mark.asyncio
    async def test_the_tool_free_wrap_up_call_reports_its_reasoning_too(self):
        events = await events_for(
            [
                Turn(thinking=["need the tool"], tool_calls=[call("call_1", "echo", {"message": "hi"})]),
                Turn(thinking=["wrapping up"], content="final"),
            ],
            "ollama",
            max_iterations=1,
        )

        assert thinking_text(events) == "need the toolwrapping up"
        assert kinds(events).index("thinking", kinds(events).index("tool_end")) < kinds(events).index("token")


class TestBufferedStream:
    @pytest.mark.asyncio
    async def test_a_buffered_turn_reports_its_reasoning_before_the_answer(self):
        events = await events_for(ANSWER_AFTER_THOUGHT, "native")

        assert kinds(events)[:2] == ["thinking", "token"]
        assert thinking_text(events) == "let me think"
        assert "thinking_end" not in kinds(events)

    @pytest.mark.asyncio
    async def test_a_buffered_tool_round_reports_each_turn(self):
        events = await events_for(TOOL_THEN_ANSWER, "native")

        assert thinking_text(events) == "need the toolnow answer"
        assert kinds(events).index("thinking") < kinds(events).index("tool_start")

    @pytest.mark.asyncio
    async def test_a_buffered_turn_without_reasoning_emits_none(self):
        events = await events_for([Turn(content="plain")], "native")

        assert "thinking" not in kinds(events)

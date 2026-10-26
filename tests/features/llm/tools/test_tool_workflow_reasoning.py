import copy
from typing import Any, Dict, List

import pytest

from src.features.llm.tools.base import BaseTool, ToolApprovalPreview, ToolContext, ToolResult
from src.features.llm.tools.executor import ToolExecutor
from src.features.llm.tools.registry import ToolRegistry
from tests.features.llm.tools.test_tool_workflow_thinking import EchoTool, ScriptedLLM, Turn, call

LIVE = "ollama"
BUFFERED = "native"
ECHO_XML = '<tool_call>{"name": "echo", "arguments": {"message": "hi"}}</tool_call>'
GATE_XML = '<tool_call>{"name": "gate", "arguments": {}}</tool_call>'


class RecordingLLM(ScriptedLLM):
    def __init__(self, turns: List[Turn], config_type: str):
        super().__init__(turns, config_type)
        self.requests: List[List[Dict[str, Any]]] = []

    async def generate_with_tools(self, messages, llm_id, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        return await super().generate_with_tools(messages, llm_id, **kwargs)

    def stream_with_tools(self, messages, llm_id, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        return super().stream_with_tools(messages, llm_id, **kwargs)


class GateTool(BaseTool):
    @property
    def name(self) -> str:
        return "gate"

    @property
    def description(self) -> str:
        return "Needs approval."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, context: ToolContext, **kwargs) -> ToolResult:
        return ToolResult(success=True, data="awaiting", preview=ToolApprovalPreview(action="Gate", fields=[]))


async def run(turns: List[Turn], config_type: str = LIVE):
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(GateTool())
    llm = RecordingLLM(turns, config_type)
    executor = ToolExecutor(tool_registry=registry, llm_service=llm)
    events = [
        event
        async for event in executor.execute_with_tools_stream(
            messages=[{"role": "user", "content": "hi"}],
            llm_id="model-1",
            system_message="sys",
            tool_context=ToolContext(user_id="u1"),
        )
    ]
    return events, llm


def done(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return next(event for event in events if event["type"] == "done")["data"]


def tool_call_messages(request: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [message for message in request if message.get("tool_calls")]


def echo_round(reasoning: List[str]) -> Turn:
    return Turn(thinking=reasoning, tool_calls=[call("call_1", "echo", {"message": "hi"})])


@pytest.mark.parametrize("config_type", [LIVE, BUFFERED])
class TestReasoningBetweenToolRounds:
    @pytest.mark.asyncio
    async def test_a_rounds_reasoning_rides_its_tool_call_into_the_next_request(self, config_type):
        _events, llm = await run([echo_round(["need ", "the tool"]), Turn(content="final")], config_type)

        assert tool_call_messages(llm.requests[0]) == []
        (sent,) = tool_call_messages(llm.requests[1])
        assert sent["reasoning"] == "need the tool"
        assert sent["tool_calls"][0]["function"]["name"] == "echo"

    @pytest.mark.asyncio
    async def test_a_round_without_reasoning_carries_no_reasoning_key(self, config_type):
        _events, llm = await run([echo_round([]), Turn(content="final")], config_type)

        (sent,) = tool_call_messages(llm.requests[1])
        assert "reasoning" not in sent

    @pytest.mark.asyncio
    async def test_every_round_keeps_its_own_reasoning(self, config_type):
        turns = [
            Turn(thinking=["first"], tool_calls=[call("call_1", "echo", {"message": "a"})]),
            Turn(thinking=["second"], tool_calls=[call("call_2", "echo", {"message": "b"})]),
            Turn(content="final"),
        ]

        _events, llm = await run(turns, config_type)

        assert [m["reasoning"] for m in tool_call_messages(llm.requests[2])] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_the_transcript_stores_each_rounds_reasoning(self, config_type):
        events, _llm = await run([echo_round(["need the tool"]), Turn(thinking=["wrap up"], content="final")], config_type)

        transcript = done(events)["tool_transcript"]
        assert [round_["reasoning"] for round_ in transcript] == ["need the tool"]

    @pytest.mark.asyncio
    async def test_a_round_without_reasoning_stores_none(self, config_type):
        events, _llm = await run([echo_round([]), Turn(content="final")], config_type)

        assert "reasoning" not in done(events)["tool_transcript"][0]


class TestLiveSource:
    @pytest.mark.asyncio
    async def test_reasoning_from_an_abandoned_empty_attempt_is_not_carried(self):
        turns = [Turn(thinking=["lost "]), echo_round(["kept"]), Turn(content="final")]

        _events, llm = await run(turns, LIVE)

        (sent,) = tool_call_messages(llm.requests[-1])
        assert sent["reasoning"] == "kept"

    @pytest.mark.asyncio
    async def test_an_inline_tool_call_keeps_the_reasoning_that_preceded_it(self):
        events, llm = await run([Turn(thinking=["plan"], content=ECHO_XML), Turn(content="final")], LIVE)

        (sent,) = tool_call_messages(llm.requests[1])
        assert sent["reasoning"] == "plan"
        assert done(events)["tool_transcript"][0]["reasoning"] == "plan"

    @pytest.mark.asyncio
    async def test_a_pausing_structured_call_keeps_its_reasoning_in_the_transcript(self):
        events, _llm = await run([Turn(thinking=["weigh it"], tool_calls=[call("call_1", "gate", {})])], LIVE)

        assert done(events)["pending_tool_approval"] is True
        assert done(events)["tool_transcript"][0]["reasoning"] == "weigh it"

    @pytest.mark.asyncio
    async def test_a_pausing_inline_call_keeps_its_reasoning_in_the_transcript(self):
        events, _llm = await run([Turn(thinking=["weigh it"], content=GATE_XML)], LIVE)

        assert done(events)["pending_tool_approval"] is True
        assert done(events)["tool_transcript"][0]["reasoning"] == "weigh it"


class TestBufferedSource:
    @pytest.mark.asyncio
    async def test_reasoning_a_provider_copied_into_the_content_is_not_sent_twice(self):
        same = Turn(thinking=["only thoughts"], content="only thoughts", tool_calls=[call("call_1", "echo", {})])

        _events, llm = await run([same, Turn(content="final")], BUFFERED)

        (sent,) = tool_call_messages(llm.requests[1])
        assert sent["content"] == "only thoughts"
        assert "reasoning" not in sent

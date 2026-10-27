from unittest.mock import Mock

import pytest

from src.features.llm.tools.base import ToolExecution, ToolResult
from tests.features.chat.test_runtime_streaming import BaseStreamingTest, collect_stream, make_message_response


def thinking_event(text):
    return {"type": "thinking", "content": text}


def token_event(text):
    return {"type": "token", "content": text}


def stream_of(*events):
    async def _gen():
        for event in events:
            yield event
        yield {"type": "usage", "tokens_used": None, "prompt_tokens": None, "completion_tokens": None}

    return _gen()


def kinds(events):
    return [e["event"] for e in events]


def sse(events, name):
    return [e["data"] for e in events if e["event"] == name]


def thinking_status(events):
    return next(d for d in sse(events, "status") if d["step"] == "thinking")


class ThinkingStreamBase(BaseStreamingTest):
    def _prepare(self, show_thinking, tools=False):
        config = Mock()
        config.show_thinking = show_thinking
        self.mock_llm.repository.get_configuration.return_value = config
        self.mock_repo.get_session.return_value = self._make_active_session(
            metadata={"enable_tools": True} if tools else None,
        )
        self.mock_repo.get_conversation_history.return_value = []
        self.mock_repo.add_message.side_effect = [
            make_message_response("msg-user", role="user", content="hello"),
            make_message_response("msg-asst", content="x"),
        ]
        if tools:
            executor = Mock()
            tool = Mock()
            tool.name = "get_data"
            tool.hint = ""
            executor.tool_registry.get_for_mode.return_value = [tool]
            executor.tool_registry.get_tool_hints_text.return_value = ""
            self.manager.tool_executor = executor

    async def _run(self):
        return await collect_stream(
            self.manager.send_message_stream(session_id="session-123", user_id="user-123", content="hello")
        )

    def _saved_metadata(self):
        return self.mock_repo.add_message.call_args_list[1].kwargs["metadata"]

    def _processed(self):
        return self.mock_processor.process.call_args.args[0]


class TestPlainPath(ThinkingStreamBase):
    @pytest.mark.asyncio
    async def test_the_knob_off_streams_no_reasoning_and_stores_none(self):
        self._prepare(show_thinking=False)
        self.engine.script(stream_of(thinking_event("secret"), token_event("answer")))

        events = await self._run()

        assert "reasoning" not in kinds(events)
        assert "".join(d["content"] for d in sse(events, "token")) == "answer"
        assert self._processed() == "answer"
        assert "thinking" not in self._saved_metadata()
        assert "thinking_visible" not in self._saved_metadata()["behavior_trace"]
        assert thinking_status(events) == {"step": "thinking", "state": "started"}

    @pytest.mark.asyncio
    async def test_provider_reasoning_streams_before_the_answer_and_is_stored(self):
        self._prepare(show_thinking=True)
        self.engine.script(stream_of(thinking_event("let me "), thinking_event("think"), token_event("the answer")))

        events = await self._run()

        assert sse(events, "reasoning") == [
            {"block": 0, "kind": "raw", "text": "let me "},
            {"block": 0, "kind": "raw", "text": "think"},
        ]
        names = kinds(events)
        assert names.index("reasoning") < names.index("reasoning_end") < names.index("token")
        assert self._processed() == "the answer"
        segments = self._saved_metadata()["thinking"]["segments"]
        assert [s["text"] for s in segments] == ["let me think"]
        assert segments[0]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_a_reply_without_reasoning_stores_no_thinking_but_marks_the_trace(self):
        self._prepare(show_thinking=True)
        self.engine.script(stream_of(token_event("plain")))

        events = await self._run()

        assert "reasoning" not in kinds(events)
        assert "thinking" not in self._saved_metadata()
        assert self._saved_metadata()["behavior_trace"]["thinking_visible"] is True
        assert thinking_status(events) == {
            "step": "thinking", "state": "started", "detail": {"thinking_visible": True},
        }


class TestToolsPath(ThinkingStreamBase):
    TOOL_ROUND = [
        {"type": "thinking", "data": {"content": "need the tool"}},
        {"type": "tool_start", "data": {"tool_name": "get_data", "arguments": {}}},
        {"type": "tool_end", "data": {"tool_name": "get_data", "success": True, "duration_ms": 1}},
        {"type": "thinking", "data": {"content": "now answer"}},
        {"type": "token", "data": {"content": "the answer"}},
    ]

    def _script(self, script, full_content="answer"):
        execution = ToolExecution(
            tool_name="get_data", arguments={}, result=ToolResult(success=True, data="r", error=None), duration_ms=1,
        )

        async def _stream(**kwargs):
            for event in script:
                yield event
            yield {"type": "done", "data": {"tool_executions": [execution], "full_content": full_content}}

        self.engine.script(_stream)

    @pytest.mark.asyncio
    async def test_each_model_turn_around_a_tool_call_is_its_own_segment(self):
        self._prepare(True, tools=True)
        self._script(self.TOOL_ROUND, full_content="the answer")

        events = await self._run()

        order = [e["event"] for e in events if e["event"] in ("reasoning", "reasoning_end", "tool_start", "tool_end", "token")]
        assert order == ["reasoning", "reasoning_end", "tool_start", "tool_end", "reasoning", "reasoning_end", "token"]
        segments = self._saved_metadata()["thinking"]["segments"]
        assert [s["text"] for s in segments] == ["need the tool", "now answer"]

    @pytest.mark.asyncio
    async def test_the_knob_off_drops_every_reasoning_event_on_the_tools_path(self):
        self._prepare(False, tools=True)
        self._script(self.TOOL_ROUND, full_content="the answer")

        events = await self._run()

        assert "reasoning" not in kinds(events)
        assert kinds(events).count("tool_start") == 1
        assert "thinking" not in self._saved_metadata()

    @pytest.mark.asyncio
    async def test_a_pending_approval_turn_keeps_the_reasoning_before_the_tool(self):
        self._prepare(True, tools=True)
        self._script([
            {"type": "thinking", "data": {"content": "should I?"}},
            {"type": "tool_start", "data": {"tool_name": "get_data", "arguments": {}}},
            {"type": "tool_end", "data": {"tool_name": "get_data", "success": True, "duration_ms": 1}},
        ], full_content="")

        await self._run()

        assert [s["text"] for s in self._saved_metadata()["thinking"]["segments"]] == ["should I?"]

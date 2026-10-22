from typing import List
from unittest.mock import Mock

import pytest

from src.features.chat.runtime import ChatRuntime
from src.features.llm.tools.base import ToolExecution, ToolResult
from tests.features.chat.test_runtime_streaming import (
    BaseStreamingTest,
    collect_stream,
    make_message_response,
)
from tests.fixtures.chat_modes import mode_registry


def thinking_event(text):
    return {"type": "thinking", "content": text}


def token_event(text):
    return {"type": "token", "content": text}


def stream_of(*events):
    usage = {"type": "usage", "tokens_used": None, "prompt_tokens": None, "completion_tokens": None}

    async def _gen():
        for event in events:
            yield event
        yield usage

    return _gen()


def kinds(events):
    return [e["event"] for e in events]


def sse(events, name):
    return [e["data"] for e in events if e["event"] == name]


def thinking_status(events):
    return next(d for d in sse(events, "status") if d["step"] == "thinking")


class ThinkingStreamBase(BaseStreamingTest):
    def _configure(self, show_thinking):
        config = Mock()
        config.show_thinking = show_thinking
        self.mock_llm.repository.get_configuration.return_value = config

    def _prepare(self, show_thinking):
        self._configure(show_thinking)
        self.mock_repo.get_session.return_value = self._make_active_session()
        self.mock_repo.get_conversation_history.return_value = []
        self.mock_repo.add_message.side_effect = [
            make_message_response("msg-user", role="user", content="hello"),
            make_message_response("msg-asst", content="x"),
        ]

    async def _run(self):
        return await collect_stream(
            self.manager.send_message_stream(session_id="session-123", user_id="user-123", content="hello")
        )

    def _saved_metadata(self):
        return self.mock_repo.add_message.call_args_list[1].kwargs["metadata"]

    def _processed(self):
        return self.mock_processor.process.call_args.args[0]


class TestPlainPath(ThinkingStreamBase):
    def _stream(self, *events):
        self.mock_llm.stream_with_history = Mock(return_value=stream_of(*events))

    @pytest.mark.asyncio
    async def test_the_knob_off_changes_nothing(self):
        self._prepare(show_thinking=False)
        self._stream(thinking_event("secret"), token_event("<think>raw</think>answer"))

        events = await self._run()

        assert "thinking" not in kinds(events)
        assert "".join(d["content"] for d in sse(events, "token")) == "<think>raw</think>answer"
        assert self._processed() == "<think>raw</think>answer"
        assert "thinking" not in self._saved_metadata()
        assert "thinking_visible" not in self._saved_metadata()["behavior_trace"]
        assert thinking_status(events) == {"step": "thinking", "state": "started"}

    @pytest.mark.asyncio
    async def test_provider_reasoning_streams_and_is_stored(self):
        self._prepare(show_thinking=True)
        self._stream(thinking_event("let me "), thinking_event("think"), token_event("the answer"))

        events = await self._run()

        assert sse(events, "thinking") == [
            {"content": "let me ", "segment": 0},
            {"content": "think", "segment": 0},
        ]
        assert "".join(d["content"] for d in sse(events, "token")) == "the answer"
        assert self._processed() == "the answer"
        segments = self._saved_metadata()["thinking"]["segments"]
        assert [s["text"] for s in segments] == ["let me think"]
        assert segments[0]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_thinking_comes_before_the_answer_on_the_wire(self):
        self._prepare(show_thinking=True)
        self._stream(thinking_event("hm"), token_event("answer"))

        events = await self._run()

        names = kinds(events)
        assert names.index("thinking") < names.index("token")

    @pytest.mark.asyncio
    async def test_inline_tags_split_across_chunks_are_captured_and_removed_from_the_reply(self):
        self._prepare(show_thinking=True)
        self._stream(token_event("<thi"), token_event("nk>reas"), token_event("on</th"), token_event("ink>\n\nthe answer"))

        events = await self._run()

        assert "".join(d["content"] for d in sse(events, "thinking")) == "reason"
        assert "".join(d["content"] for d in sse(events, "token")) == "the answer"
        assert self._processed() == "the answer"
        assert self._saved_metadata()["thinking"]["segments"][0]["text"] == "reason"

    @pytest.mark.asyncio
    async def test_an_unclosed_block_at_the_end_of_the_stream_is_all_thinking(self):
        self._prepare(show_thinking=True)
        self._stream(token_event("<think>never "), token_event("closed"))

        events = await self._run()

        assert "".join(d["content"] for d in sse(events, "thinking")) == "never closed"
        assert sse(events, "token") == []
        assert self._processed() == ""
        assert self._saved_metadata()["thinking"]["segments"][0]["text"] == "never closed"

    @pytest.mark.asyncio
    async def test_a_dangling_partial_tag_at_the_end_is_still_shown_as_text(self):
        self._prepare(show_thinking=True)
        self._stream(token_event("see <thi"))

        events = await self._run()

        assert "".join(d["content"] for d in sse(events, "token")) == "see <thi"
        assert self._processed() == "see <thi"

    @pytest.mark.asyncio
    async def test_a_reply_without_reasoning_stores_no_thinking_but_marks_the_trace(self):
        self._prepare(show_thinking=True)
        self._stream(token_event("plain"))

        events = await self._run()

        assert "thinking" not in kinds(events)
        assert "thinking" not in self._saved_metadata()
        assert self._saved_metadata()["behavior_trace"]["thinking_visible"] is True

    @pytest.mark.asyncio
    async def test_the_status_step_tells_the_client_thinking_is_shown(self):
        self._prepare(show_thinking=True)
        self._stream(token_event("plain"))

        events = await self._run()

        assert thinking_status(events) == {
            "step": "thinking", "state": "started", "detail": {"thinking_visible": True},
        }


class TestToolsPath(ThinkingStreamBase):
    def _prepare_tools(self, show_thinking, script, full_content="answer"):
        self._prepare(show_thinking)
        self.mock_repo.get_session.return_value = self._make_active_session(metadata={"enable_tools": True})
        executor = Mock()
        tool = Mock()
        tool.name = "get_data"
        tool.hint = ""
        executor.tool_registry.get_for_mode.return_value = [tool]
        executor.tool_registry.get_tool_hints_text.return_value = ""
        execution = ToolExecution(
            tool_name="get_data", arguments={}, result=ToolResult(success=True, data="r", error=None), duration_ms=1,
        )

        async def _stream(*args, **kwargs):
            for event in script:
                yield event
            yield {"type": "done", "data": {"tool_executions": [execution], "full_content": full_content}}

        executor.execute_with_tools_stream = Mock(side_effect=_stream)
        self.manager = ChatRuntime(
            chat_repository=self.mock_repo,
            llm_service=self.mock_llm,
            response_processor=self.mock_processor,
            plugin_registry=self.mock_plugins,
            chat_mode_registry=mode_registry(),
            tool_executor=executor,
        )

    TOOL_ROUND = [
        {"type": "thinking", "data": {"content": "need the tool"}},
        {"type": "tool_start", "data": {"tool_name": "get_data", "arguments": {}}},
        {"type": "tool_end", "data": {"tool_name": "get_data", "success": True, "duration_ms": 1}},
        {"type": "thinking", "data": {"content": "now answer"}},
        {"type": "token", "data": {"content": "the answer"}},
    ]

    @pytest.mark.asyncio
    async def test_each_model_turn_around_a_tool_call_is_its_own_segment(self):
        self._prepare_tools(True, self.TOOL_ROUND, full_content="the answer")

        events = await self._run()

        order = [(e["event"], e["data"].get("segment")) for e in events if e["event"] in ("thinking", "tool_start", "tool_end", "token")]
        assert order == [
            ("thinking", 0), ("tool_start", None), ("tool_end", None), ("thinking", 1), ("token", None),
        ]
        segments = self._saved_metadata()["thinking"]["segments"]
        assert [s["text"] for s in segments] == ["need the tool", "now answer"]

    @pytest.mark.asyncio
    async def test_each_abandoned_empty_attempt_keeps_its_reasoning_as_its_own_segment(self):
        attempt = [
            {"type": "thinking", "data": {"content": "hm"}},
            {"type": "thinking_end", "data": {}},
        ]
        self._prepare_tools(True, attempt * 3, full_content="")

        events = await self._run()

        assert [d["segment"] for d in sse(events, "thinking")] == [0, 1, 2]
        assert [s["text"] for s in self._saved_metadata()["thinking"]["segments"]] == ["hm", "hm", "hm"]
        assert sse(events, "token") == []
        assert self._processed() == ""

    @pytest.mark.asyncio
    async def test_without_an_end_marker_consecutive_attempts_share_a_segment(self):
        script = [{"type": "thinking", "data": {"content": "hm"}}] * 3
        self._prepare_tools(True, script, full_content="")

        await self._run()

        assert [s["text"] for s in self._saved_metadata()["thinking"]["segments"]] == ["hmhmhm"]

    @pytest.mark.asyncio
    async def test_a_held_tail_released_before_the_end_marker_stays_in_the_open_block(self):
        script = [
            {"type": "token", "data": {"content": "<think>plan "}},
            {"type": "token", "data": {"content": "<tool"}},
            {"type": "thinking_end", "data": {}},
        ]
        self._prepare_tools(True, script, full_content="<think>plan <tool")

        events = await self._run()

        assert "".join(d["content"] for d in sse(events, "thinking")) == "plan <tool"
        assert sse(events, "token") == []
        assert self._saved_metadata()["thinking"]["segments"][0]["text"] == "plan <tool"

    @pytest.mark.asyncio
    async def test_a_partial_tag_never_joins_text_from_the_next_attempt(self):
        script = [
            {"type": "token", "data": {"content": "see <thi"}},
            {"type": "thinking_end", "data": {}},
            {"type": "token", "data": {"content": "nk>x"}},
        ]
        self._prepare_tools(True, script, full_content="see <think>x")

        events = await self._run()

        assert "".join(d["content"] for d in sse(events, "token")) == "see <think>x"
        assert sse(events, "thinking") == []

    @pytest.mark.asyncio
    async def test_the_end_marker_changes_nothing_while_the_knob_is_off(self):
        script = [
            {"type": "thinking", "data": {"content": "hm"}},
            {"type": "thinking_end", "data": {}},
            {"type": "token", "data": {"content": "the answer"}},
        ]
        self._prepare_tools(False, script, full_content="the answer")

        events = await self._run()

        assert kinds(events).count("token") == 1
        assert "thinking" not in kinds(events)

    @pytest.mark.asyncio
    async def test_the_knob_off_drops_every_thinking_event_on_the_tools_path(self):
        self._prepare_tools(False, self.TOOL_ROUND, full_content="the answer")

        events = await self._run()

        assert "thinking" not in kinds(events)
        assert kinds(events).count("tool_start") == 1
        assert "thinking" not in self._saved_metadata()

    @pytest.mark.asyncio
    async def test_inline_thinking_in_the_tool_path_is_split_out(self):
        script = [{"type": "token", "data": {"content": "<think>plan</think>the answer"}}]
        self._prepare_tools(True, script, full_content="<think>plan</think>the answer")

        events = await self._run()

        assert "".join(d["content"] for d in sse(events, "thinking")) == "plan"
        assert "".join(d["content"] for d in sse(events, "token")) == "the answer"
        assert self._processed() == "the answer"

    @pytest.mark.asyncio
    async def test_a_done_only_reply_never_leaks_an_unclosed_think_tag(self):
        self._prepare_tools(True, [], full_content="<think>never closed")

        await self._run()

        assert self._processed() == ""

    @pytest.mark.asyncio
    async def test_a_done_only_reply_is_untouched_when_the_knob_is_off(self):
        self._prepare_tools(False, [], full_content="<think>never closed")

        await self._run()

        assert self._processed() == "<think>never closed"

    @pytest.mark.asyncio
    async def test_a_pending_approval_turn_keeps_the_thinking_before_the_tool(self):
        script = [
            {"type": "thinking", "data": {"content": "should I?"}},
            {"type": "tool_start", "data": {"tool_name": "get_data", "arguments": {}}},
            {"type": "tool_end", "data": {"tool_name": "get_data", "success": True, "duration_ms": 1}},
        ]
        self._prepare_tools(True, script, full_content="")

        await self._run()

        assert [s["text"] for s in self._saved_metadata()["thinking"]["segments"]] == ["should I?"]

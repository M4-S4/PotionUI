import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.features.llm import context_budget
from src.features.llm.clients.base import LLMResponse
from src.features.llm.tools.base import BaseTool, ToolContext, ToolResult
from src.features.llm.tools.executor import ToolExecutor
from src.features.llm.tools.registry import ToolRegistry
from src.features.llm.tools.use_tools import (
    USE_TOOLS_NAME,
    deferred_tool_groups,
    load_tool_group,
    use_tools_schema,
)


class _NamedTool(BaseTool):
    def __init__(self, name: str, context_group: str = "core"):
        self._name = name
        self.context_group = context_group

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"The {self._name} tool."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, context: ToolContext, **kwargs) -> ToolResult:
        return ToolResult(success=True, data=f"{self._name} ran")


def _call(name: str, arguments: Dict[str, Any], call_id: str) -> Dict[str, Any]:
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}


def _response(content: str = "", tool_calls=None) -> LLMResponse:
    return LLMResponse(content=content, model="m", provider_id="p", tool_calls=tool_calls or [])


def _executor(config_type: str = "openai"):
    registry = ToolRegistry()
    registry.register(_NamedTool("echo"))
    registry.register(_NamedTool("shout", "extra"))
    llm_service = AsyncMock()
    config = MagicMock()
    config.type = config_type
    config.provider_options = {}
    llm_service.repository.get_configuration = MagicMock(return_value=config)
    return ToolExecutor(tool_registry=registry, llm_service=llm_service), llm_service


def _context() -> ToolContext:
    return ToolContext(user_id="user-1")


def _tool_names(schemas):
    return [schema["function"]["name"] for schema in schemas]


async def _drain(gen):
    return [event async for event in gen]


class TestDeferredToolGroups:
    def _tools(self):
        return [_NamedTool("a"), _NamedTool("b", "library"), _NamedTool("c", "memory_maintenance")]

    def test_large_defers_nothing(self):
        assert deferred_tool_groups(self._tools(), context_budget.TOOL_TIER_LARGE, ("core",)) == {}

    def test_standard_defers_only_the_library(self):
        assert deferred_tool_groups(self._tools(), context_budget.TOOL_TIER_STANDARD, ("core",)) == {"library": ["b"]}

    def test_compact_defers_everything_outside_the_core_groups(self):
        deferred = deferred_tool_groups(self._tools(), context_budget.TOOL_TIER_COMPACT, ("core", "library"))
        assert deferred == {"memory_maintenance": ["c"]}

    def test_pinned_names_and_loaded_groups_stay(self):
        deferred = deferred_tool_groups(
            self._tools(), context_budget.TOOL_TIER_COMPACT, ("core",),
            always_core=("b",), loaded_groups=("memory_maintenance",),
        )
        assert deferred == {}


class TestLoadToolGroup:
    def test_moves_a_group_into_the_allowed_list(self):
        allowed, deferred = ["echo"], {"extra": ["shout"], "other": ["x"]}
        result = load_tool_group("extra", allowed, deferred)
        assert result.success
        assert allowed == ["echo", "shout"]
        assert deferred == {"other": ["x"]}

    def test_refuses_a_group_that_is_not_deferred(self):
        allowed, deferred = ["echo"], {"extra": ["shout"]}
        result = load_tool_group("nope", allowed, deferred)
        assert not result.success
        assert "extra" in result.error
        assert allowed == ["echo"]

    def test_schema_offers_exactly_the_deferred_groups(self):
        schema = use_tools_schema({"library": ["a"], "custom": ["b", "c"]})
        assert schema["function"]["name"] == USE_TOOLS_NAME
        assert schema["function"]["parameters"]["properties"]["group"]["enum"] == ["custom", "library"]
        assert "'custom': b, c" in schema["function"]["description"]


class TestUseToolsInTheLoop:
    @pytest.mark.asyncio
    async def test_buffered_loop_adds_the_group_to_the_next_request(self):
        executor, llm_service = _executor()
        llm_service.generate_with_tools.side_effect = [
            _response(tool_calls=[_call(USE_TOOLS_NAME, {"group": "extra"}, "c1")]),
            _response(tool_calls=[_call("shout", {}, "c2")]),
            _response("done"),
        ]
        allowed, deferred = ["echo"], {"extra": ["shout"]}

        response, executions = await executor.execute_with_tools(
            messages=[{"role": "user", "content": "shout"}], llm_id="m", system_message="sys",
            tool_context=_context(), allowed_tools=allowed, deferred_tools=deferred,
        )

        sent = [call.kwargs["tools"] for call in llm_service.generate_with_tools.call_args_list]
        assert _tool_names(sent[0]) == ["echo", USE_TOOLS_NAME]
        assert _tool_names(sent[1]) == ["echo", "shout"]
        assert [e.tool_name for e in executions] == [USE_TOOLS_NAME, "shout"]
        assert all(e.result.success for e in executions)
        assert response.content == "done"
        assert allowed == ["echo"]
        assert deferred == {"extra": ["shout"]}

    @pytest.mark.asyncio
    async def test_live_stream_sends_the_grown_tool_set(self):
        executor, llm_service = _executor()
        rounds = [
            [{"type": "tool_calls", "tool_calls": [_call(USE_TOOLS_NAME, {"group": "extra"}, "c1")]}],
            [{"type": "token", "content": "done"}],
        ]
        sent = []

        def _stream(**kwargs):
            sent.append(kwargs["tools"])
            events = rounds[len(sent) - 1]

            async def _gen():
                for event in events:
                    yield event

            return _gen()

        llm_service.stream_with_tools = _stream

        events = await _drain(executor.execute_with_tools_stream(
            messages=[{"role": "user", "content": "shout"}], llm_id="m", system_message="sys",
            tool_context=_context(), allowed_tools=["echo"], deferred_tools={"extra": ["shout"]},
        ))

        assert _tool_names(sent[0]) == ["echo", USE_TOOLS_NAME]
        assert _tool_names(sent[1]) == ["echo", "shout"]
        done = next(e for e in events if e["type"] == "done")
        assert done["data"]["full_content"] == "done"

    @pytest.mark.asyncio
    async def test_prompt_tools_path_sends_the_grown_tool_set(self):
        executor, llm_service = _executor(config_type="native")
        llm_service.generate_with_tools.side_effect = [
            _response(content='<tool_call>{"name": "use_tools", "arguments": {"group": "extra"}}</tool_call>'),
            _response("done"),
        ]

        await _drain(executor.execute_with_tools_stream(
            messages=[{"role": "user", "content": "shout"}], llm_id="m", system_message="sys",
            tool_context=_context(), allowed_tools=["echo"], deferred_tools={"extra": ["shout"]},
        ))

        sent = [call.kwargs["tools"] for call in llm_service.generate_with_tools.call_args_list]
        assert _tool_names(sent[0]) == ["echo", USE_TOOLS_NAME]
        assert _tool_names(sent[1]) == ["echo", "shout"]

    @pytest.mark.asyncio
    async def test_calling_a_deferred_tool_directly_loads_its_group_and_runs_it(self):
        executor, llm_service = _executor()
        llm_service.generate_with_tools.side_effect = [
            _response(tool_calls=[_call("shout", {}, "c1")]),
            _response("done"),
        ]

        _response_obj, executions = await executor.execute_with_tools(
            messages=[{"role": "user", "content": "shout"}], llm_id="m", system_message="sys",
            tool_context=_context(), allowed_tools=["echo"], deferred_tools={"extra": ["shout"]},
        )

        assert executions[0].result.success
        assert executions[0].result.data == "shout ran"
        second = llm_service.generate_with_tools.call_args_list[1].kwargs["tools"]
        assert _tool_names(second) == ["echo", "shout"]

    @pytest.mark.asyncio
    async def test_a_tool_outside_every_group_is_still_refused(self):
        executor, llm_service = _executor()
        llm_service.generate_with_tools.side_effect = [
            _response(tool_calls=[_call("ghost", {}, "c1")]),
            _response("done"),
        ]

        _response_obj, executions = await executor.execute_with_tools(
            messages=[{"role": "user", "content": "go"}], llm_id="m", system_message="sys",
            tool_context=_context(), allowed_tools=["echo"], deferred_tools={"extra": ["shout"]},
        )

        assert not executions[0].result.success
        assert "do not call it again" in executions[0].result.error

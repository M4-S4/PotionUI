import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.features.llm import context_budget
from src.features.llm.harness.build import user_message
from src.features.llm.harness.engine import TurnInput, TurnOutcome
from src.features.llm.harness.events import Finish, FinishReason, TextDelta
from tests.fixtures.llm_engine import drain, scripted_engine
from tests.fixtures.llm_harness import ScriptedAdapter, text_step, tool_step
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
    async def _run(self, *steps, allowed=("echo",), deferred=None):
        adapter = ScriptedAdapter(*steps)
        rig = scripted_engine(adapter, [_NamedTool("echo"), _NamedTool("shout", "extra")])
        events = await drain(rig.engine.run(TurnInput(
            session_id="s1", llm_id="cfg-scripted", system="sys", history=(),
            new_messages=(user_message("shout"),), tool_context=_context(),
            allowed_tools=allowed, deferred_tools=deferred if deferred is not None else {"extra": ("shout",)},
        )))
        outcome = next(event for event in events if isinstance(event, TurnOutcome))
        return adapter, outcome

    async def test_loading_a_group_grows_the_next_requests_tool_set(self):
        adapter, outcome = await self._run(
            tool_step({"id": "c1", "name": USE_TOOLS_NAME, "args": {"group": "extra"}}),
            tool_step({"id": "c2", "name": "shout", "args": {}}),
            text_step("done"),
        )

        sent = [[tool.name for tool in request.tools] for request in adapter.requests]
        assert sent[0] == ["echo", USE_TOOLS_NAME]
        assert sent[1] == ["echo", USE_TOOLS_NAME, "shout"]
        assert [e.tool_name for e in outcome.executions] == [USE_TOOLS_NAME, "shout"]
        assert all(e.result.success for e in outcome.executions)
        assert outcome.content == "done"
        assert outcome.state.loaded_tool_groups == ("extra",)

    async def test_an_xml_use_tools_call_grows_the_tool_set_too(self):
        adapter, _ = await self._run(
            [TextDelta(block=1, text='<tool_call>{"name": "use_tools", "arguments": {"group": "extra"}}</tool_call>'), Finish(reason=FinishReason.STOP)],
            text_step("done"),
        )

        assert [tool.name for tool in adapter.requests[1].tools] == ["echo", USE_TOOLS_NAME, "shout"]

    async def test_calling_a_deferred_tool_directly_loads_its_group_and_runs_it(self):
        adapter, outcome = await self._run(tool_step({"id": "c1", "name": "shout", "args": {}}), text_step("done"))

        assert outcome.executions[0].result.success
        assert outcome.executions[0].result.data == "shout ran"
        assert [tool.name for tool in adapter.requests[1].tools] == ["echo", USE_TOOLS_NAME, "shout"]

    async def test_a_tool_outside_every_group_is_still_refused(self):
        _, outcome = await self._run(tool_step({"id": "c1", "name": "ghost", "args": {}}), text_step("done"))

        assert not outcome.executions[0].result.success
        assert "do not call it again" in outcome.executions[0].result.error


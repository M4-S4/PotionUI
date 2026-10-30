import asyncio

from src.features.llm.tools.base import BaseTool, ToolContext, ToolResult
from src.features.llm.tools.executor import ToolExecutor
from src.features.llm.tools.registry import ToolRegistry


class Shifting(BaseTool):
    name = "shifting"
    requires_approval = True

    def __init__(self):
        self.revision = 1
        self.builds = 0

    @property
    def description(self):
        self.builds += 1
        return f"revision {self.revision}"

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "required": []}

    def schema_revision(self):
        return self.revision

    def approval_required(self, arguments):
        return bool(arguments.get("risky"))

    async def execute(self, context, **kwargs):
        return ToolResult(success=True, data="preview" if kwargs.get("risky") else "done")

    async def execute_confirmed(self, context, **kwargs):
        return ToolResult(success=True, data="applied")


class Plain(BaseTool):
    name = "plain"

    @property
    def description(self):
        return "plain"

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, context, **kwargs):
        return ToolResult(success=True, data="ok")


def run(coro):
    return asyncio.run(coro)


class TestSchemaRevision:
    def test_a_tool_without_a_revision_keeps_the_cached_schema_list(self):
        registry = ToolRegistry()
        registry.register(Plain())

        assert registry.get_schemas() is registry.get_schemas()

    def test_the_schema_list_is_cached_while_a_tools_revision_is_unchanged(self):
        registry = ToolRegistry()
        tool = Shifting()
        registry.register(tool)

        first = registry.get_schemas()
        second = registry.get_schemas()

        assert first is second
        assert tool.builds == 1

    def test_a_new_revision_rebuilds_the_schema_list_for_every_name_filter(self):
        registry = ToolRegistry()
        tool = Shifting()
        registry.register(tool)
        registry.register(Plain())
        before = registry.get_schemas(["shifting"])
        everything = registry.get_schemas()

        tool.revision = 2

        after = registry.get_schemas(["shifting"])
        assert before[0]["function"]["description"] == "revision 1"
        assert after[0]["function"]["description"] == "revision 2"
        assert {s["function"]["description"] for s in registry.get_schemas()} == {"revision 2", "plain"}
        assert everything is not registry.get_schemas()


class TestPerCallApproval:
    def setup_method(self):
        self.tool = Shifting()
        registry = ToolRegistry()
        registry.register(self.tool)
        registry.register(Plain())
        self.executor = ToolExecutor(registry)
        self.context = ToolContext(user_id="u")

    def test_the_default_follows_the_static_flag(self):
        assert Plain().approval_required({}) is False
        assert Plain().schema_revision() is None

        class Gated(Plain):
            requires_approval = True

        assert Gated().approval_required({"anything": 1}) is True

    def test_a_call_is_pending_only_when_the_tool_says_this_call_needs_approval(self):
        safe, safe_pending = run(self.executor._execute_tool("shifting", self.context, {}))
        risky, risky_pending = run(self.executor._execute_tool("shifting", self.context, {"risky": True}))

        assert (safe.data, safe_pending) == ("done", False)
        assert (risky.data, risky_pending) == ("preview", True)

    def test_confirming_is_refused_for_a_call_that_needed_no_approval(self):
        refused = run(self.executor.execute_tool_confirmed("shifting", self.context, {}))
        applied = run(self.executor.execute_tool_confirmed("shifting", self.context, {"risky": True}))

        assert refused.success is False and "does not require approval" in refused.error
        assert applied.data == "applied"

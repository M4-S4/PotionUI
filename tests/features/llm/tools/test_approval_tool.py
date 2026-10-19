import json

import pytest

from src.features.llm.tools.base import ApprovalTool, ToolApprovalPreview, ToolContext, ToolResult


class _SyncTool(ApprovalTool):
    name = "sync_tool"
    repositories = ("llm_memory_repository",)
    unavailable_error = "Widget manager not available"
    description = "d"
    parameters = {"type": "object", "properties": {}}

    def _preview(self, context, kwargs):
        if kwargs.get("bad"):
            raise ValueError("bad input")
        return self._proposal(ToolApprovalPreview(action="Do"), {"x": 1}, "confirm")

    def _apply(self, context, kwargs):
        if kwargs.get("bad"):
            raise ValueError("bad input")
        return ToolResult(success=True, data="applied")


class _AsyncTool(_SyncTool):
    name = "async_tool"

    async def _apply(self, context, kwargs):
        return ToolResult(success=True, data="applied async")


def _context(**overrides):
    return ToolContext(user_id="u", llm_memory_repository=overrides.get("repo", object()))


class TestApprovalTool:
    def test_requires_approval_by_default(self):
        assert _SyncTool().requires_approval is True

    @pytest.mark.asyncio
    async def test_preview_wraps_the_proposal(self):
        result = await _SyncTool().execute(_context())

        assert result.success is True
        assert result.preview.action == "Do"
        assert json.loads(result.data) == {"action": "sync_tool", "proposal": {"x": 1}, "message": "confirm"}

    @pytest.mark.asyncio
    async def test_value_error_becomes_a_failed_result_in_both_phases(self):
        tool = _SyncTool()

        for phase in (tool.execute, tool.execute_confirmed):
            result = await phase(_context(), bad=True)
            assert result.success is False
            assert result.error == "bad input"

    @pytest.mark.asyncio
    async def test_missing_repository_short_circuits_both_phases(self):
        tool = _SyncTool()

        for phase in (tool.execute, tool.execute_confirmed):
            result = await phase(_context(repo=None))
            assert result.success is False
            assert result.error == "Widget manager not available"

    @pytest.mark.asyncio
    async def test_apply_may_be_sync_or_async(self):
        assert (await _SyncTool().execute_confirmed(_context())).data == "applied"
        assert (await _AsyncTool().execute_confirmed(_context())).data == "applied async"

    @pytest.mark.asyncio
    async def test_unimplemented_steps_are_not_swallowed(self):
        class _Bare(ApprovalTool):
            name = "bare"
            description = "d"
            parameters = {}

        with pytest.raises(NotImplementedError):
            await _Bare().execute(_context())

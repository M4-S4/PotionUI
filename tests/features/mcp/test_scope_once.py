import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.features.llm.tools.base import ToolServices
from src.features.llm.tools.governance_repository import ToolGovernanceRepository
from src.features.llm.tools.registry import ToolRegistry
from src.features.mcp.protocol import McpToolCollaborators, handle_method
from tests.features.mcp.test_protocol import _EchoTool


class _CountingGovernance(ToolGovernanceRepository):
    def __init__(self):
        self.snapshot_calls = 0
        self.disabled_calls = 0

    def get_config_snapshot(self, llm_config_id, tool_names):
        self.snapshot_calls += 1
        return {}

    def get_user_disabled(self, user_id):
        self.disabled_calls += 1
        return set()


def _collaborators():
    registry = ToolRegistry()
    registry.register(_EchoTool())
    governance = _CountingGovernance()
    llm_repository = Mock()
    llm_repository.get_default_configuration.return_value = SimpleNamespace(id="cfg-default")
    collaborators = McpToolCollaborators(
        tool_registry=registry,
        tool_governance_repository=governance,
        services=ToolServices(llm_repository=llm_repository),
    )
    return collaborators, governance, llm_repository


@pytest.mark.asyncio
async def test_tools_list_resolves_governance_once():
    collaborators, governance, llm_repository = _collaborators()

    result = await handle_method(collaborators, "tools/list", {}, "user-1")

    assert [t["name"] for t in result["tools"]] == ["echo"]
    assert (governance.snapshot_calls, governance.disabled_calls) == (1, 1)
    assert llm_repository.get_default_configuration.call_count == 1


@pytest.mark.asyncio
async def test_tools_call_resolves_governance_and_the_default_config_once():
    collaborators, governance, llm_repository = _collaborators()

    result = await handle_method(
        collaborators, "tools/call", {"name": "echo", "arguments": {"text": "hi"}}, "user-1",
    )

    assert result["isError"] is False
    assert (governance.snapshot_calls, governance.disabled_calls) == (1, 1)
    assert llm_repository.get_default_configuration.call_count == 1

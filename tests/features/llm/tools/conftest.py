import pytest

from src.features.llm.tools.executor import ToolExecutor
from src.features.llm.tools.registry import ToolRegistry


def _executor_for(tool):
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(tool_registry=registry)


@pytest.fixture
def run_tool():
    async def run(tool, context, **arguments):
        result, _pending = await _executor_for(tool)._execute_tool(tool.name, context, arguments)
        return result

    return run


@pytest.fixture
def confirm_tool():
    async def confirm(tool, context, **arguments):
        return await _executor_for(tool).execute_tool_confirmed(tool.name, context, arguments)

    return confirm

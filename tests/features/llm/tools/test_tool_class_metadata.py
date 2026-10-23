import inspect

import pytest

from src.features.llm.tools.builtin import register_builtin_tools
from src.features.llm.tools.registry import ToolRegistry

ATTRIBUTES = ("name", "group", "user_description", "requires_approval")


def _builtin_tool_classes():
    registry = ToolRegistry()
    register_builtin_tools(registry)
    return sorted({type(tool) for tool in registry.get_all()}, key=lambda cls: cls.__name__)


@pytest.mark.parametrize("cls", _builtin_tool_classes(), ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("attribute", ATTRIBUTES)
def test_tool_metadata_is_a_plain_class_attribute(cls, attribute):
    assert not isinstance(inspect.getattr_static(cls, attribute), property)

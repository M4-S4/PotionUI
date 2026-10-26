from src.features.llm.tools.base import BaseTool
from src.features.llm.tools.builtin import register_builtin_tools
from src.features.llm.tools.registry import ToolRegistry

READ_ONLY = {
    "get_active_models",
    "get_current_segments",
    "get_form_state",
    "get_model_info",
    "get_music_director",
    "get_phrasebook_values",
    "get_preset_info",
    "get_prompt",
    "get_prompt_variables",
    "get_saved_segments",
    "get_segment_templates",
    "get_video_director",
    "list_library_items",
    "list_models",
    "list_phrasebook_categories",
    "list_presets",
    "list_prompts",
    "list_segment_categories",
    "list_tab_drafts",
    "read_memory",
    "search_gallery",
    "search_model_prompts",
    "search_models",
}


def builtin_tools():
    registry = ToolRegistry()
    register_builtin_tools(registry)
    return registry.get_all()


def test_a_tool_is_not_read_only_unless_it_says_so():
    assert BaseTool.read_only is False


def test_exactly_the_audited_read_tools_are_flagged_read_only():
    assert {tool.name for tool in builtin_tools() if tool.read_only} == READ_ONLY


def test_a_read_only_tool_never_asks_for_approval():
    assert [tool.name for tool in builtin_tools() if tool.read_only and tool.requires_approval] == []


def test_every_tool_that_writes_stays_sequential():
    writers = [tool for tool in builtin_tools() if tool.name.split("_")[0] in {"create", "update", "delete", "write", "remove", "add", "edit", "set", "manage", "organize", "run", "start"}]
    assert writers
    assert [tool.name for tool in writers if tool.read_only] == []

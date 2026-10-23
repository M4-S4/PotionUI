import re
from unittest.mock import Mock

from src.features.chat.context_builder import ChatContextBuilder
from src.features.chat.modes import (
    ChatMode,
    ChatModeRegistry,
    build_generation_mode,
    build_history_mode,
    build_phrasebook_mode,
    build_prompts_mode,
)
from src.features.llm import context_budget
from src.features.llm.tools.builtin import register_builtin_tools
from src.features.llm.tools.registry import ToolRegistry

_COMPACT = context_budget.TOOL_TIER_COMPACT
_STANDARD = context_budget.TOOL_TIER_STANDARD
_LARGE = context_budget.TOOL_TIER_LARGE


def _builder(*extra_modes):
    tools = ToolRegistry()
    register_builtin_tools(tools)
    modes = ChatModeRegistry()
    for mode in (build_generation_mode(), build_history_mode(), build_phrasebook_mode(), build_prompts_mode(), *extra_modes):
        modes.register(mode)
    manager = Mock()
    manager.chat_mode_registry = modes
    manager.tool_executor = Mock()
    manager.tool_executor.tool_registry = tools
    manager.tool_governance_repository = None
    return ChatContextBuilder(manager)


def _session(mode_id="generation", metadata=None):
    session = Mock()
    session.mode = mode_id
    session.metadata = metadata or {}
    session.user_id = None
    session.llm_config_id = None
    return session


def _resolve(builder, tier, mode_id="generation", **kwargs):
    return builder.resolve_session_prompt_and_tools(_session(mode_id, kwargs.pop("metadata", None)), tool_tier=tier, **kwargs)


class TestGenerationTiers:
    def test_large_offers_every_tool_and_defers_nothing(self):
        _prompt, allowed, _mode, deferred = _resolve(_builder(), _LARGE)
        assert deferred == {}
        assert {"list_segment_categories", "read_memory", "create_phrasebook_values"} <= set(allowed)

    def test_standard_defers_only_the_library(self):
        _prompt, allowed, _mode, deferred = _resolve(_builder(), _STANDARD)
        assert set(deferred) == {"library"}
        assert {"list_segment_categories", "list_presets", "start_generation"} <= set(deferred["library"])
        assert not set(deferred["library"]) & set(allowed)
        assert {"read_memory", "create_phrasebook_values", "set_prompt_relay_timeline"} <= set(allowed)

    def test_compact_keeps_only_the_core(self):
        _prompt, allowed, _mode, deferred = _resolve(_builder(), _COMPACT)
        assert set(deferred) == {"library", "memory_maintenance", "phrasebook_write", "director"}
        assert {"write_memory", "get_form_state", "search_gallery", "get_phrasebook_values"} <= set(allowed)
        assert "update_memory" not in allowed

    def test_compact_prompt_never_names_a_deferred_tool(self):
        prompt, _allowed, _mode, deferred = _resolve(_builder(), _COMPACT)
        for names in deferred.values():
            for name in names:
                assert not re.search(rf"\b{name}\b", prompt)

    def test_a_loaded_group_comes_back_into_the_offered_set(self):
        _prompt, allowed, _mode, deferred = _resolve(_builder(), _COMPACT, loaded_groups={"memory_maintenance"})
        assert "memory_maintenance" not in deferred
        assert {"read_memory", "update_memory", "delete_memory"} <= set(allowed)

    def test_tiers_and_loaded_groups_never_share_a_cache_entry(self):
        builder = _builder()
        compact = _resolve(builder, _COMPACT)
        large = _resolve(builder, _LARGE)
        loaded = _resolve(builder, _COMPACT, loaded_groups={"library"})
        assert compact[1] != large[1]
        assert compact[3] != loaded[3]

    def test_a_session_disabled_tool_is_neither_offered_nor_deferred(self):
        _prompt, allowed, _mode, deferred = _resolve(
            _builder(), _COMPACT, metadata={"disabled_tools": ["read_memory"]},
        )
        assert "read_memory" not in allowed
        assert "read_memory" not in deferred["memory_maintenance"]


class TestPageModesKeepTheirOwnTools:
    def test_phrasebook_mode_keeps_its_write_tools_on_compact(self):
        _prompt, allowed, _mode, deferred = _resolve(_builder(), _COMPACT, "phrasebook")
        assert {"create_phrasebook_values", "remove_phrasebook_values", "get_phrasebook_values"} <= set(allowed)
        assert set(deferred) == {"memory_maintenance"}

    def test_history_and_prompts_keep_their_library_tools_on_compact(self):
        builder = _builder()
        _p, history, _m, _d = _resolve(builder, _COMPACT, "history")
        _p, prompts, _m, _d = _resolve(builder, _COMPACT, "prompts")
        assert {"organize_gallery", "manage_collections", "list_library_items"} <= set(history)
        assert {"add_prompt", "edit_prompt", "delete_prompt", "get_prompt"} <= set(prompts)

    def test_a_plugin_mode_keeps_the_builtin_tools_it_borrows(self):
        plugin_mode = ChatMode(id="plugin-mode", name="Plugin", tool_names=["read_memory"], source="plugin-x")
        _prompt, allowed, _mode, deferred = _resolve(_builder(plugin_mode), _COMPACT, "plugin-mode")
        assert "read_memory" in allowed
        assert set(deferred) == {"memory_maintenance"}
        assert set(deferred["memory_maintenance"]) == {"update_memory", "delete_memory"}

import re

import pytest

from src.features.chat.context_builder import PARTIAL_TOOLS_BLOCK, ChatContextBuilder
from src.features.chat.modes import builtin
from src.features.llm.tools.tool_conditionals import render_tool_conditionals
from src.features.llm.tools.governance import ToolGovernanceRepository
from src.features.llm.tools.errors import unavailable_tool_error
from tests.fixtures.chat_engine import typed_history
from tests.features.chat.test_tool_availability_transparency import (
    _make_builder,
    _session,
    governance_db,
)

BUILTIN_BUILDERS = [
    getattr(builtin, n) for n in dir(builtin) if n.startswith("build_") and n.endswith("_mode")
]


def _segments(count):
    return [
        {"index": i, "id": f"s{i}", "content": "x" * 400, "name": f"Seg {i}", "type": "text", "enabled": True}
        for i in range(count)
    ]


def _model_facing_text(builder, session):
    form_state = {"preset": "native/SDXL", "mode": "image", "form_data": {}, "video_director": {"active": False}}
    system_prompt, allowed, mode, deferred = builder.resolve_session_prompt_and_tools(session, form_state=form_state)
    withheld = builder.withheld_tools_for_session(session, form_state=form_state)
    offered = set(allowed or ()).union(*(deferred or {}).values())
    history = [{"role": "user", "content": "improve my prompt"}]
    metadata = {"form_state": form_state, "segments": _segments(30)}
    typed_history(builder.inject_workspace_block)(history, metadata, offered_tools=offered)
    typed_history(builder.inject_prompt_state_block)(history, metadata, offered_tools=offered)
    typed_history(builder.inject_reply_contract_reminder_block)(history, mode, "large")
    typed_history(builder.inject_tool_availability_block)(history, allowed, withheld)
    text = system_prompt + "\n" + "\n".join(m["content"] for m in history)
    return text, withheld, allowed


def _case_toggle_off(governance_db):
    return _make_builder(ToolGovernanceRepository()), _session(metadata={"tools_enabled": False})


def _case_disabled_in_mode(governance_db):
    return _make_builder(ToolGovernanceRepository()), _session(
        metadata={"disabled_tools": ["get_active_models", "get_current_segments"]}
    )


def _case_disabled_by_admin(governance_db):
    repo = ToolGovernanceRepository()
    repo.upsert_config("cfg-a", "get_active_models", enabled=False)
    repo.upsert_config("cfg-a", "get_current_segments", enabled=False)
    return _make_builder(repo), _session()


def _case_opted_out(governance_db):
    repo = ToolGovernanceRepository()
    repo.set_user_disabled("user-1", "get_active_models", True)
    repo.set_user_disabled("user-1", "get_current_segments", True)
    return _make_builder(repo), _session(user_id="user-1")


@pytest.mark.parametrize(
    "case,reason",
    [
        (_case_toggle_off, "off_by_toggle"),
        (_case_disabled_in_mode, "disabled_in_mode"),
        (_case_disabled_by_admin, "disabled_by_admin"),
        (_case_opted_out, "opted_out"),
    ],
)
def test_withheld_tool_names_never_reach_the_model(governance_db, case, reason):
    builder, session = case(governance_db)
    text, withheld, allowed = _model_facing_text(builder, session)

    assert withheld
    assert set(withheld.values()) == {reason}
    for name in withheld:
        assert name not in text, name
    for name in allowed:
        assert name in text or name == "get_current_segments", name
    if allowed:
        assert PARTIAL_TOOLS_BLOCK in text
    else:
        assert "Tools are unavailable in this conversation" in text


def test_partial_block_is_one_fixed_line_without_names():
    history = [{"role": "user", "content": "hi"}]
    typed_history(ChatContextBuilder.inject_tool_availability_block)(
        history, ["get_form_state"], {"get_active_models": "disabled_in_mode", "list_models": "opted_out"}
    )
    assert PARTIAL_TOOLS_BLOCK in history[0]["content"]
    assert "get_active_models" not in history[0]["content"]
    assert "list_models" not in history[0]["content"]


def test_trace_still_records_withheld_names_and_reasons(governance_db):
    builder, session = _case_disabled_by_admin(governance_db)
    withheld = builder.withheld_tools_for_session(session)
    assert withheld == {"get_active_models": "disabled_by_admin", "get_current_segments": "disabled_by_admin"}


def test_ungated_hints_follow_offered_tools():
    segments = _segments(30)
    metadata = {"segments": segments}

    with_tool = [{"role": "user", "content": "hi"}]
    typed_history(ChatContextBuilder.inject_prompt_state_block)(with_tool, metadata, offered_tools={"get_current_segments"})
    assert "get_current_segments" in with_tool[0]["content"]

    without_tool = [{"role": "user", "content": "hi"}]
    typed_history(ChatContextBuilder.inject_prompt_state_block)(without_tool, metadata, offered_tools={"get_form_state"})
    assert "get_current_segments" not in without_tool[0]["content"]
    assert "more segments" in without_tool[0]["content"]


def _workspace_text(offered):
    from unittest.mock import Mock
    from src.features.chat.modes import ChatModeRegistry, build_generation_mode
    from src.features.chat.runtime import ChatRuntime
    from src.features.llm.tools.base import ToolServices

    registry = ChatModeRegistry()
    registry.register(build_generation_mode())
    manager = ChatRuntime(
        chat_repository=Mock(), llm_service=Mock(), response_processor=Mock(), plugin_registry=Mock(),
        chat_mode_registry=registry,
        tool_services=ToolServices(model_index_manager=Mock(), preset_collaborators=None),
    )
    history = [{"role": "user", "content": "hello"}]
    metadata = {"form_state": {"preset": "native/SDXL", "mode": "image", "form_data": {}, "video_director": {"active": False}}}
    typed_history(manager._context.inject_workspace_block)(history, metadata, offered_tools=offered)
    return history[0]["content"]


def test_workspace_steering_names_only_offered_tools():
    assert "update_form_settings" in _workspace_text({"update_form_settings"})
    block = _workspace_text({"get_form_state"})
    assert "update_form_settings" not in block
    assert "Video Director: no document active." in block


_TOOL_WORD = re.compile(
    r"\b(?:get|list|search|create|update|delete|write|read|manage|edit|add|remove|enhance|start|run|organize|set)"
    r"_[a-z_]+\b"
)


@pytest.mark.parametrize("build", BUILTIN_BUILDERS)
def test_builtin_mode_prompts_name_no_tool_when_none_offered(build):
    mode = build()
    rendered = render_tool_conditionals(mode.resolve_prompt_template(), [])
    assert not [w for w in _TOOL_WORD.findall(rendered) if w != "update_director_segment"]


class TestUnavailableToolError:
    def test_names_allowed_tools_and_says_to_stop(self):
        assert unavailable_tool_error("echo", ["fail", "other"]) == (
            "Tool 'echo' is unavailable; do not call it again. Use: fail, other."
        )

    def test_empty_allowed_set(self):
        assert unavailable_tool_error("echo", []) == (
            "No tools are available in this conversation; answer without calling tools."
        )

    def test_unrestricted_set_has_no_list(self):
        assert unavailable_tool_error("echo", None) == "Tool 'echo' is unavailable; do not call it again."

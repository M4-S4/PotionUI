import json
from unittest.mock import Mock

import pytest

from src.features.chat.context_builder import ChatContextBuilder
from src.features.chat.modes import (
    ChatModeRegistry,
    build_generation_mode,
    build_history_mode,
    build_models_mode,
    build_phrasebook_mode,
    build_prompts_mode,
)
from src.features.llm.clients.ollama_wire import build_prompt_tools_text
from src.features.llm.tools.builtin import register_builtin_tools
from src.features.llm.tools.registry import ToolRegistry

_TOLERANCE = 0.03

_FIXED_OVERHEAD_CHARS = {
    "generation": 54866,
    "history": 13261,
    "models": 9124,
    "phrasebook": 15011,
    "prompts": 13340,
}


def _builder():
    tools = ToolRegistry()
    register_builtin_tools(tools)
    modes = ChatModeRegistry()
    for build in (build_generation_mode, build_history_mode, build_models_mode, build_phrasebook_mode, build_prompts_mode):
        modes.register(build())
    manager = Mock()
    manager.chat_mode_registry = modes
    manager.tool_executor = Mock()
    manager.tool_executor.tool_registry = tools
    manager.tool_governance_repository = None
    return ChatContextBuilder(manager), tools


def _session(mode_id):
    session = Mock()
    session.mode = mode_id
    session.metadata = {}
    session.user_id = None
    session.llm_config_id = None
    return session


def _wire(builder, tools, mode_id, form_state=None):
    prompt, allowed, _ = builder.resolve_session_prompt_and_tools(_session(mode_id), form_state=form_state)
    schemas = tools.get_schemas(allowed)
    return prompt, json.dumps(schemas), build_prompt_tools_text(schemas)


@pytest.mark.parametrize("mode_id", sorted(_FIXED_OVERHEAD_CHARS))
def test_fixed_overhead_stays_at_its_pinned_size(mode_id):
    builder, tools = _builder()
    prompt, schemas, _ = _wire(builder, tools, mode_id)
    measured = len(prompt) + len(schemas)
    pinned = _FIXED_OVERHEAD_CHARS[mode_id]
    assert abs(measured - pinned) <= pinned * _TOLERANCE, (
        f"{mode_id}: system prompt + tool schemas is {measured} chars, pinned {pinned}; "
        "re-pin only when the change is meant to move it"
    )


def test_system_prompt_and_tools_are_byte_identical_across_turns():
    first_turn = {"preset": "sdxl", "form_data": {"prompt": "a fox", "steps": 20}}
    second_turn = {"preset": "sdxl", "form_data": {"prompt": "a castle at dusk", "steps": 32, "cfg": 5}}
    first_builder, first_tools = _builder()
    second_builder, second_tools = _builder()

    first = _wire(first_builder, first_tools, "generation", first_turn)
    second = _wire(second_builder, second_tools, "generation", second_turn)

    assert first == second

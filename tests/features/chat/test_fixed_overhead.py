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
from src.features.llm import context_budget
from src.features.llm.clients.ollama_wire import build_prompt_tools_text
from src.features.llm.tools.builtin import register_builtin_tools
from src.features.llm.tools.registry import ToolRegistry
from src.features.llm.tools.use_tools import use_tools_schema

_TOLERANCE = 0.03

_COMPACT = context_budget.TOOL_TIER_COMPACT
_STANDARD = context_budget.TOOL_TIER_STANDARD
_LARGE = context_budget.TOOL_TIER_LARGE

_FIXED_OVERHEAD_CHARS = {
    ("generation", _LARGE): 45091,
    ("generation", _STANDARD): 32241,
    ("generation", _COMPACT): 22247,
    ("history", _LARGE): 13261,
    ("history", _STANDARD): 13261,
    ("history", _COMPACT): 10460,
    ("models", _LARGE): 9124,
    ("models", _STANDARD): 9124,
    ("models", _COMPACT): 6323,
    ("phrasebook", _LARGE): 13992,
    ("phrasebook", _STANDARD): 13992,
    ("phrasebook", _COMPACT): 11191,
    ("prompts", _LARGE): 13544,
    ("prompts", _STANDARD): 13544,
    ("prompts", _COMPACT): 10743,
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


def _wire(builder, tools, mode_id, tier, form_state=None):
    prompt, allowed, _, deferred = builder.resolve_session_prompt_and_tools(
        _session(mode_id), form_state=form_state, tool_tier=tier,
    )
    schemas = tools.get_schemas(allowed) + ([use_tools_schema(deferred)] if deferred else [])
    return prompt, json.dumps(schemas), build_prompt_tools_text(schemas)


@pytest.mark.parametrize("mode_id,tier", sorted(_FIXED_OVERHEAD_CHARS))
def test_fixed_overhead_stays_at_its_pinned_size(mode_id, tier):
    builder, tools = _builder()
    prompt, schemas, _ = _wire(builder, tools, mode_id, tier)
    measured = len(prompt) + len(schemas)
    pinned = _FIXED_OVERHEAD_CHARS[(mode_id, tier)]
    assert abs(measured - pinned) <= pinned * _TOLERANCE, (
        f"{mode_id}/{tier}: system prompt + tool schemas is {measured} chars, pinned {pinned}; "
        "re-pin only when the change is meant to move it"
    )


@pytest.mark.parametrize("tier", [_COMPACT, _STANDARD, _LARGE])
def test_system_prompt_and_tools_are_byte_identical_across_turns(tier):
    first_turn = {"preset": "sdxl", "form_data": {"prompt": "a fox", "steps": 20}}
    second_turn = {"preset": "sdxl", "form_data": {"prompt": "a castle at dusk", "steps": 32, "cfg": 5}}
    first_builder, first_tools = _builder()
    second_builder, second_tools = _builder()

    first = _wire(first_builder, first_tools, "generation", tier, first_turn)
    second = _wire(second_builder, second_tools, "generation", tier, second_turn)

    assert first == second

"""Chat feature wiring.

`ChatRuntime` reaches almost every other domain through its tool context, so it
used to be constructed early with ``None`` placeholders and completed by
seventeen attribute assignments at the bottom of `build_container`. Nothing
built between those two points consumes a chat component, so the placeholders
were an ordering artefact, not a cycle: `build_chat` is called once every
collaborator exists and passes them all to the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

from src.features.chat import ChatRuntime, ResponseProcessor
from src.features.chat.pre_chat_actions import PreChatActionRegistry
from src.features.chat.repository import ChatRepository
from src.features.chat.routes import ChatController
from src.features.chat.turns import ChatTurnRegistry
from src.features.llm.gateway import LLMGateway
from src.features.llm.repository import LLMRepository
from src.features.llm.tools.base import ToolServices
from src.features.llm_memory import operations as memory_operations
from src.features.sessions.hooks import SESSION_HOOKS
from src.platform.plugins import PluginRegistry
from src.platform.plugins.hooks import HookContext
from src.platform.settings.settings import Settings

if TYPE_CHECKING:
    from src.features.chat.modes import ChatModeRegistry
    from src.features.generation.model_repository import GenerationModelRepository
    from src.features.generation.parameter_repository import GenerationParameterRepository
    from src.features.generation.repository import GenerationRepository
    from src.features.llm.harness.engine import TurnEngine
    from src.features.llm.harness.ports import ReplaySelector
    from src.features.llm.tools.executor import ToolExecutor
    from src.platform.resources import ResourceRegistry


DEFAULT_TURN_TIMEOUT_SECONDS = 1800
SESSION_MEMORY_CLEANUP_HANDLER_ID = "core.session_memory_cleanup"


@dataclass(frozen=True)
class ChatDeps:
    """Everything chat needs from the rest of the process."""

    llm_service: LLMGateway
    llm_repository: LLMRepository
    plugin_registry: PluginRegistry
    settings: Settings
    chat_mode_registry: "ChatModeRegistry"
    resource_registry: "ResourceRegistry"
    tool_executor: "ToolExecutor"
    tool_governance_repository: Any
    tool_services: ToolServices
    phrasebook_search: Callable[..., Any]
    preset_lookup: Callable[..., Any]
    preset_form_schema_lookup: Callable[..., Any]
    generation_repository: "GenerationRepository"
    generation_parameter_repository: "GenerationParameterRepository"
    generation_model_repository: "GenerationModelRepository"
    turn_engine: "TurnEngine"
    replay_selector: "ReplaySelector"


@dataclass(frozen=True)
class ChatComponents:
    """What chat contributes back to the container."""

    chat_repository: ChatRepository
    response_processor: ResponseProcessor
    pre_chat_action_registry: PreChatActionRegistry
    chat_runtime: ChatRuntime
    chat_turn_registry: ChatTurnRegistry
    chat_controller: ChatController


def _turn_timeout_seconds(settings: Settings) -> int:
    try:
        return int(settings.get_setting("chat_turn_timeout_seconds", DEFAULT_TURN_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_TURN_TIMEOUT_SECONDS


def register_session_memory_cleanup(plugin_registry: PluginRegistry, llm_memory_repository: Any) -> None:
    if llm_memory_repository is None:
        return

    def _delete_session_notes(context: HookContext) -> HookContext:
        session_id = context.data.get("session_id")
        user_id = context.data.get("user_id")
        if session_id and user_id:
            memory_operations.delete_notes_for_scope_ref(
                llm_memory_repository, user_id=user_id, scope="session", scope_ref=session_id,
            )
        return context

    plugin_registry.hook_chain.register(
        SESSION_HOOKS.after_delete, SESSION_MEMORY_CLEANUP_HANDLER_ID, _delete_session_notes,
    )


def build_chat(deps: ChatDeps) -> ChatComponents:
    chat_repository = ChatRepository()
    response_processor = ResponseProcessor(plugin_registry=deps.plugin_registry)
    pre_chat_action_registry = PreChatActionRegistry(
        plugin_registry=deps.plugin_registry,
        llm_repository=deps.llm_repository,
    )
    chat_runtime = ChatRuntime(
        chat_repository=chat_repository,
        llm_service=deps.llm_service,
        response_processor=response_processor,
        plugin_registry=deps.plugin_registry,
        chat_mode_registry=deps.chat_mode_registry,
        tool_executor=deps.tool_executor,
        preset_lookup=deps.preset_lookup,
        preset_form_schema_lookup=deps.preset_form_schema_lookup,
        phrasebook_search=deps.phrasebook_search,
        pre_chat_action_registry=pre_chat_action_registry,
        resource_registry=deps.resource_registry,
        tool_governance_repository=deps.tool_governance_repository,
        tool_services=deps.tool_services,
        generation_repository=deps.generation_repository,
        generation_parameter_repository=deps.generation_parameter_repository,
        generation_model_repository=deps.generation_model_repository,
        turn_engine=deps.turn_engine,
        replay_selector=deps.replay_selector,
    )

    # Turns are owned by a per-process registry so a client disconnect (page
    # reload) can't kill an in-flight response.
    chat_turn_registry = ChatTurnRegistry(turn_timeout_seconds=_turn_timeout_seconds(deps.settings))
    return ChatComponents(
        chat_repository=chat_repository,
        response_processor=response_processor,
        pre_chat_action_registry=pre_chat_action_registry,
        chat_runtime=chat_runtime,
        chat_turn_registry=chat_turn_registry,
        chat_controller=ChatController(chat_runtime, chat_turn_registry),
    )

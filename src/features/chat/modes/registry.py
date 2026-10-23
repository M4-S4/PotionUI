"""Registry for chat modes."""

import logging
from typing import List, Optional

from src.features.chat.exceptions import UnknownChatModeException
from src.features.chat.reply_contract import REPLY_CONTRACT_PROMPT_BLOCK
from src.features.llm.tools.tool_conditionals import render_tool_conditionals
from src.platform.plugins.chat_modes import ChatMode, DuplicateChatModeError
from src.platform.plugins.sourced_registry import SourcedRegistry

logger = logging.getLogger(__name__)


class ChatModeRegistry(SourcedRegistry[ChatMode]):
    """Registry of available chat modes.

    The builtin 'generation' mode is registered at startup; plugins register
    additional modes when enabled and unregister them (by source) on disable.
    """

    key_attr = "id"
    label = "Chat mode"
    duplicate_error = DuplicateChatModeError

    def _duplicate_message(self, key: str) -> str:
        return f"Chat mode '{key}' is already registered (by '{self._entries[key].source}')"

    def unregister(self, mode_id: str) -> bool:
        """Unregister a mode by id. Returns True if it was found and removed."""
        if mode_id in self._entries:
            del self._entries[mode_id]
            logger.debug(f"Unregistered chat mode: {mode_id}")
            return True
        return False

    def unregister_source(self, source: str) -> int:
        """Unregister all modes registered by the given source (plugin id).

        Returns the number of modes removed.
        """
        removed = super().unregister_source(source)
        if removed:
            logger.info(f"Unregistered {removed} chat modes for source '{source}'")
        return removed

    def get_all(self) -> List[ChatMode]:
        """Get all registered modes."""
        return self.all()

    def require(self, mode_id: str) -> ChatMode:
        """Get a mode by id or raise UnknownChatModeException."""
        mode = self._entries.get(mode_id)
        if not mode:
            raise UnknownChatModeException(f"Unknown chat mode '{mode_id}'")
        return mode

    def resolve_system_prompt(
        self,
        mode: ChatMode,
        tool_hints_text: str,
        allowed_names: Optional[List[str]] = None,
    ) -> str:
        """Build the mode's system prompt for the session.

        Resolves any ``{{#if}}`` / ``{{#ifany}}`` tool-conditional blocks against
        ``allowed_names`` (so no rule references a tool the session can't call —
        available to plugin mode prompts too, since every mode's template flows
        through here), then substitutes ``{{TOOL_HINTS}}``.

        ``allowed_names=None`` keeps every conditional block (markers stripped),
        preserving the old behavior for callers that don't thread the allowed set.

        When ``mode.structured_reply`` is true (the default), the
        ``## improved`` / ``## questions`` reply-contract block is appended
        last, after tool-conditional resolution and ``{{TOOL_HINTS}}``
        substitution.
        """
        template = mode.resolve_prompt_template()
        template = render_tool_conditionals(template, allowed_names)
        resolved = template.replace("{{TOOL_HINTS}}", tool_hints_text or "")
        if mode.structured_reply:
            resolved += REPLY_CONTRACT_PROMPT_BLOCK
        return resolved

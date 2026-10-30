"""
LLM response processing pipeline with plugin hook support.

This module provides a processing pipeline for LLM responses that:
1. Runs built-in transformations (remove thinking tags)
2. Executes CHAT_RESPONSE_TRANSFORM hook for plugin transformations
3. Returns cleaned content and parsed data

Example plugin use case:
- Plugin registers hook to detect custom tags like "<jump_to_sky>"
- Hook transforms it to {"actions": ["jump"]} in parsed_content
"""

import re
import logging
from typing import Dict, Any, Tuple, Optional
from src.features.chat.hooks import CHAT_RESPONSE_HOOKS
from src.features.chat.reply_contract import parse_reply_contract
from src.platform.plugins.hooks import execute_hook


logger = logging.getLogger(__name__)


class ResponseProcessor:
    """Process LLM responses through a pipeline with plugin hooks.

    Pipeline:
    1. Remove thinking/thought tags (built-in)
    2. Execute CHAT_RESPONSE_TRANSFORM hook (plugins can register transformations)
    3. Return cleaned content and parsed data

    Example plugin use case:
    - Plugin registers hook to detect "<action:jump>" tag
    - Hook transforms it to {"actions": ["jump"]} in parsed_content
    """

    def __init__(self, plugin_registry: Optional['PluginRegistry'] = None):
        """Initialize ResponseProcessor.

        Args:
            plugin_registry: Optional plugin registry for hook execution.
                            If None, hooks are skipped.
        """
        self.plugins = plugin_registry

    def process(
        self,
        content: str,
        mode: str = 'generation'
    ) -> Tuple[str, Dict[str, Any]]:
        """Process LLM response through the full pipeline.

        Args:
            content: Raw LLM response content
            mode: Chat mode id of the session (for context in hooks)

        Returns:
            Tuple of (cleaned_content, parsed_content)
        """
        from src.features.llm.tools.builtin.condition_text import repair_condition_actions

        # Step 1: Built-in transformations
        cleaned = repair_condition_actions(self._tidy(content))

        # Step 2: Split the structured reply contract (## improved / ##
        # questions) out of the prose, before hooks run, so plugin
        # transformations see the same cleaned prose the user's bubble gets
        # rather than re-parsing or duplicating the structured sections.
        cleaned, reply_contract = parse_reply_contract(cleaned)

        # Step 3: Plugin transformations via hook
        # Plugins can modify content and add to parsed_content
        parsed_content: Dict[str, Any] = {'raw': cleaned}
        if reply_contract:
            parsed_content['reply_contract'] = reply_contract

        if self.plugins:
            hook_data, _ = execute_hook(self.plugins,
                CHAT_RESPONSE_HOOKS.transform,
                {
                    'content': cleaned,
                    'mode': mode,
                    'parsed_content': parsed_content
                }
            )
            # Plugins can modify content and add structured data
            cleaned = hook_data.get('content', cleaned)
            parsed_content = hook_data.get('parsed_content', parsed_content)
            # A hook that replaces parsed_content wholesale (rather than
            # mutating the dict it was given) must not drop the contract.
            if reply_contract:
                parsed_content.setdefault('reply_contract', reply_contract)

        return cleaned, parsed_content

    @staticmethod
    def _tidy(content: str) -> str:
        return re.sub(r'\n\s*\n\s*\n', '\n\n', content or "").strip()

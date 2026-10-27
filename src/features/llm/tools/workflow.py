import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.features.llm.tools import tool_call_rescue

logger = logging.getLogger(__name__)


# Regex to match <tool_call>...</tool_call> blocks (some models emit XML instead of structured calls)
_TOOL_CALL_XML_RE = re.compile(
    r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL
)


def parse_xml_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse <tool_call> XML blocks from content into structured tool call dicts.

    Some LLMs (especially via Ollama) output tool calls as XML in the content
    field instead of using the structured tool_calls API field. This function
    detects and parses those into the same format as native tool calls.

    Returns a list of tool call dicts with {"function": {"name": ..., "arguments": ...}}.
    """
    tool_calls = []
    for match in _TOOL_CALL_XML_RE.finditer(content):
        raw = match.group(1).strip()
        try:
            call_data = json.loads(raw)
        except json.JSONDecodeError:
            # The same tokenizer artifact tool_call_rescue works around for
            # <tool_action> tags (quote characters round-tripped as
            # <|"|>) shows up inside a well-formed <tool_call> block too — the
            # call is complete, not a near-miss, so it's demangled and retried
            # here rather than being dropped into the rescue path.
            try:
                call_data = json.loads(tool_call_rescue.demangle_quote_tokens(raw))
            except json.JSONDecodeError:
                logger.warning(f"[ToolWorkflow] Failed to parse XML tool_call JSON: {raw[:200]}")
                continue

        # Normalise to {"function": {"name": ..., "arguments": ...}}
        if "function" in call_data:
            tool_calls.append(call_data)
        elif "name" in call_data:
            tool_calls.append({
                "function": {
                    "name": call_data["name"],
                    "arguments": call_data.get("arguments") or call_data.get("parameters") or {},
                }
            })
        else:
            logger.warning(f"[ToolWorkflow] XML tool_call missing 'name'/'function': {call_data}")
    return tool_calls


def strip_tool_call_xml(content: str) -> str:
    """Remove any <tool_call>...</tool_call> blocks from content."""
    return _TOOL_CALL_XML_RE.sub('', content).strip()


class ToolCallGuard:
    """Per-turn tracker for calls that already failed, and a running per-tool
    failure count.

    A model that retries the exact same failing call burns loop iterations
    without learning anything new; once (tool_name, canonical arguments) has
    failed once this turn, the identical call is refused without re-executing
    the tool. A call whose arguments differ, or that previously succeeded, is
    never blocked.
    """

    ALREADY_APPLIED_RESULT = "Already applied earlier in this turn."

    def __init__(self, applied_calls: Optional[set] = None) -> None:
        self._failed_calls: Dict[str, str] = {}
        self.tool_failures: Dict[str, int] = {}
        self._applied_calls = set(applied_calls or ())

    @staticmethod
    def call_key(tool_name: str, arguments: Dict[str, Any]) -> str:
        try:
            args_json = json.dumps(arguments, sort_keys=True, default=str)
        except TypeError:
            args_json = str(arguments)
        return f"{tool_name}:{args_json}"

    def blocked_repeat_error(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """The teaching error to return without re-executing, or None to proceed."""
        original_error = self._failed_calls.get(self.call_key(tool_name, arguments))
        if original_error is None:
            return None
        return (
            f"You already called {tool_name} with these exact arguments and it "
            f"failed: {original_error}. Change the arguments or take a different approach."
        )

    def record(self, tool_name: str, arguments: Dict[str, Any], result) -> None:
        """Track *result* for repeat-detection and failure accounting."""
        if result.success:
            return
        self.tool_failures[tool_name] = self.tool_failures.get(tool_name, 0) + 1
        key = self.call_key(tool_name, arguments)
        if key not in self._failed_calls:
            self._failed_calls[key] = result.error or "unknown error"

    def already_applied(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        return self.call_key(tool_name, arguments) in self._applied_calls

    def failed_call_keys(self) -> List[str]:
        return list(self._failed_calls)

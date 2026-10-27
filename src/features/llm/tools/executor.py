import inspect
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from src.features.llm.tools.base import (
    ToolContext,
    ToolExecution,
    ToolResult,
    serialize_approval_preview,
)
from src.features.llm.tools.registry import ToolRegistry
from src.features.llm.tools import tool_call_rescue
from src.features.llm.tools.errors import unavailable_tool_error, unexpected
from src.features.llm.tools.use_tools import USE_TOOLS_NAME, deferred_group_of, load_tool_group, use_tools_schema
from src.features.llm.tools.workflow import ToolCallGuard

logger = logging.getLogger(__name__)

_MAX_TOOL_RESULT_CHARS = 8000

_VISION_UNSUPPORTED_NOTE = (
    "This model cannot see images — switch the chat to a vision-capable LLM "
    "(one with vision support enabled) to describe them."
)


def bound_tool_result_content(content: str) -> str:
    if content is None or len(content) <= _MAX_TOOL_RESULT_CHARS:
        return content
    return (
        content[:_MAX_TOOL_RESULT_CHARS]
        + f"\n\n[Result truncated at {_MAX_TOOL_RESULT_CHARS} characters. "
        "Refine the call (smaller limit, narrower filters) if you need more.]"
    )


class ToolExecutor:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def registered_allowed(
        self, allowed_tools: Optional[List[str]], deferred_tools: Optional[Dict[str, List[str]]] = None,
    ) -> set:
        """Registered tool names, narrowed to the session's allowed set."""
        names = {tool.name for tool in self.tool_registry.get_all()}
        if allowed_tools is not None:
            names &= set(allowed_tools)
        if deferred_tools:
            names |= {USE_TOOLS_NAME, *(name for group in deferred_tools.values() for name in group)}
        return names

    def turn_tool_schemas(
        self, allowed_tools: Optional[List[str]], deferred_tools: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict]:
        schemas = self.tool_registry.get_schemas(allowed_tools)
        if deferred_tools:
            return [*schemas, use_tools_schema(deferred_tools)]
        return schemas

    async def gate_tool_image(self, tool_context: ToolContext, result: ToolResult) -> Optional[str]:
        """Strip a tool result's image when the session's LLM config can't see
        it, returning a note to append to that tool's message content instead
        (or ``None`` when the image should be attached normally).

        Vision support can't be determined without both ``llm_repository``
        and ``llm_id`` on the context — that's treated as "attach it", the
        prior behavior, rather than a reason to block.
        """
        if not result.image_data:
            return None
        repo = tool_context.llm_repository
        llm_id = tool_context.llm_id
        if not repo or not llm_id:
            return None
        try:
            config = repo.get_configuration(llm_id)
            if inspect.isawaitable(config):
                config = await config
        except Exception:
            return None
        if not config or getattr(config, "supports_vision", True):
            return None
        result.image_data = None
        return _VISION_UNSUPPORTED_NOTE

    def rescue_final_content(
        self, content: str, allowed_tools: Optional[List[str]],
        deferred_tools: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str], List[Dict[str, Any]], str]:
        """Scan a would-be-final assistant message for near-miss tool invocations.

        Returns ``(repaired_calls, ambiguous_names, problems, records,
        cleaned_content)``: near-misses whose arguments parse and satisfy the
        tool schema become real tool_call dicts; the rest are named for a
        corrective retry, carrying the parse failures that made them ambiguous.
        Every detected span is stripped from ``cleaned_content`` so no markup
        ever surfaces.
        """
        registered = self.registered_allowed(allowed_tools, deferred_tools)
        near = tool_call_rescue.find_near_miss_invocations(content or "", registered)
        if not near:
            return [], [], [], [], content

        repaired: List[Dict[str, Any]] = []
        ambiguous: List[str] = []
        problems: List[str] = []
        records: List[Dict[str, Any]] = []
        spans: List[Tuple[int, int]] = []
        for index, nm in enumerate(near):
            spans.append(nm.span)
            tool = self.tool_registry.get(nm.tool_name)
            schema = tool.parameters if tool else {}
            if nm.arguments is not None and tool_call_rescue.validate_arguments(nm.arguments, schema):
                repaired.append({
                    "id": f"rescue_{index}",
                    "type": "function",
                    # Canonical in-process shape: object arguments (see
                    # clients.tool_call_shape). Each client re-serializes to its wire form.
                    "function": {"name": nm.tool_name, "arguments": nm.arguments},
                })
                records.append({
                    "tool_name": nm.tool_name,
                    "repaired": True,
                    "original_format": nm.original_format,
                })
                logger.debug(
                    f"[ToolExecutor] Repaired near-miss {nm.original_format} call to "
                    f"'{nm.tool_name}' into a real tool call"
                )
            else:
                ambiguous.append(nm.tool_name)
                if nm.problem:
                    problems.append(nm.problem)
        return repaired, ambiguous, problems, records, tool_call_rescue.strip_spans(content, spans)

    async def tool_result_text(self, tool_context: ToolContext, result: ToolResult) -> str:
        vision_note = await self.gate_tool_image(tool_context, result)
        content = bound_tool_result_content(result.data if result.success else f"Error: {result.error}")
        if vision_note:
            content = f"{content}\n\n{vision_note}"
        return content

    @staticmethod
    def tool_end_payload(execution: ToolExecution) -> Dict[str, Any]:
        result = execution.result
        data: Dict[str, Any] = {
            "tool_name": execution.tool_name,
            "success": result.success,
            "duration_ms": execution.duration_ms,
            "approval_status": execution.approval_status,
        }
        if execution.approval_status == "pending":
            data["arguments"] = execution.arguments
            preview = serialize_approval_preview(result.preview)
            if preview:
                data["preview"] = preview
        if result.sources:
            data["sources"] = [asdict(source) for source in result.sources]
        return data

    async def _execute_tool(
        self, tool_name: str, context: ToolContext, arguments: Dict[str, Any],
        allowed_tools: Optional[List[str]] = None,
        deferred_tools: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[ToolResult, bool]:
        """Execute a single tool by name.

        Returns:
            Tuple of (ToolResult, pending flag).
            The flag is True only when the tool has requires_approval=True
            AND the execution succeeded (i.e. produced a valid preview).
        """
        if deferred_tools and allowed_tools is not None:
            if tool_name == USE_TOOLS_NAME:
                return load_tool_group(arguments.get("group"), allowed_tools, deferred_tools), False
            group = deferred_group_of(tool_name, deferred_tools)
            if group is not None:
                load_tool_group(group, allowed_tools, deferred_tools)

        if allowed_tools is not None and tool_name not in allowed_tools:
            logger.warning(f"[ToolExecutor] Tool '{tool_name}' not in allowed_tools, rejecting")
            return ToolResult(success=False, data="", error=unavailable_tool_error(tool_name, allowed_tools)), False

        tool = self.tool_registry.get(tool_name)
        if not tool:
            logger.error(f"[ToolExecutor] Tool not found: {tool_name}")
            return ToolResult(
                success=False,
                data="",
                error=f"Tool '{tool_name}' not found",
            ), False

        try:
            logger.debug(f"[ToolExecutor] Executing tool: {tool_name} with args: {arguments}")
            result = await tool.execute(context, **arguments)
            logger.debug(f"[ToolExecutor] Tool {tool_name} completed: success={result.success}")
            pending = tool.requires_approval and result.success
            return result, pending
        except Exception as e:
            logger.error(f"[ToolExecutor] Tool {tool_name} failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data="",
                error=unexpected(tool_name, "run", e, context.is_admin),
            ), False

    async def execute_tool_guarded(
        self,
        tool_name: str,
        context: ToolContext,
        arguments: Dict[str, Any],
        allowed_tools: Optional[List[str]],
        guard: ToolCallGuard,
        deferred_tools: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[ToolResult, bool]:
        """`_execute_tool`, short-circuiting an exact repeat of a call that
        already failed this turn (see `_ToolCallGuard`) instead of re-running it."""
        if guard.already_applied(tool_name, arguments):
            return ToolResult(success=True, data=guard.ALREADY_APPLIED_RESULT), False
        blocked = guard.blocked_repeat_error(tool_name, arguments)
        if blocked is not None:
            result = ToolResult(success=False, data="", error=blocked)
            guard.record(tool_name, arguments, result)
            return result, False
        result, is_pending = await self._execute_tool(
            tool_name, context, arguments, allowed_tools=allowed_tools, deferred_tools=deferred_tools,
        )
        guard.record(tool_name, arguments, result)
        return result, is_pending

    async def execute_tool_confirmed(
        self, tool_name: str, context: ToolContext, arguments: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool's confirmed action after user approval.

        This delegates to `tool.execute_confirmed()` which applies the action
        that was previewed during the initial `execute()` call.

        Args:
            tool_name: Name of the tool to confirm
            context: Tool execution context
            arguments: Original arguments passed to `execute()`

        Returns:
            ToolResult from the confirmed execution
        """
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                data="",
                error=f"Tool '{tool_name}' not found",
            )
        if not tool.requires_approval:
            return ToolResult(
                success=False,
                data="",
                error=f"Tool '{tool_name}' does not require approval",
            )
        try:
            logger.debug(f"[ToolExecutor] Executing confirmed tool: {tool_name} with args: {arguments}")
            result = await tool.execute_confirmed(context, **arguments)
            logger.debug(f"[ToolExecutor] Confirmed tool {tool_name} completed: success={result.success}")
            return result
        except Exception as e:
            logger.error(f"[ToolExecutor] Tool {tool_name} execute_confirmed failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data="",
                error=unexpected(tool_name, "confirm", e, context.is_admin),
            )

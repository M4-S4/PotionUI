import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set
from weakref import WeakValueDictionary

from src.features.chat import tool_transcript
from src.features.chat.exceptions import (
    ContinuationConflictException,
    MessageCreationFailedException,
    ToolExecutionPendingException,
)
from src.features.llm.tools.base import ToolContext, ToolResult, serialize_approval_preview
from src.features.llm.tools.errors import unexpected
from src.features.llm.tools.use_tools import USE_TOOLS_NAME

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Continuation:
    message_id: str
    tool_index: int
    tool_name: str
    tool_call_id: str
    rounds_used: int
    applied_calls: Set[str]
    context_metadata: Optional[Dict[str, Any]]
    loaded_groups: FrozenSet[str] = frozenset()
    turn_message_ids: FrozenSet[str] = frozenset()


class ToolCallDispatcher:
    def __init__(self, manager):
        self._m = manager
        self._locks: "WeakValueDictionary[str, asyncio.Lock]" = WeakValueDictionary()

    def _lock(self, message_id: str) -> asyncio.Lock:
        lock = self._locks.get(message_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[message_id] = lock
        return lock

    def tool_context(
        self,
        session: Any,
        user_id: str,
        is_admin: bool,
        context_metadata: Optional[Dict[str, Any]],
        mode_id: Optional[str] = None,
    ) -> ToolContext:
        return ToolContext.for_session(
            self._m.tool_services,
            user_id,
            mode_id=mode_id or session.mode,
            is_admin=is_admin,
            session_metadata=context_metadata or {},
            llm_repository=self._m.llm_service.repository,
            plugin_registry=self._m.plugins,
            llm_id=session.llm_config_id,
        )

    def _load_execution(self, session_id: str, message_id: str, tool_index: int):
        message = self._m.chat_repository.get_message(message_id)
        if not message or message.session_id != session_id:
            raise MessageCreationFailedException(f"Message {message_id} not found")
        metadata = dict(message.metadata or {})
        executions = metadata.get("tool_executions") or []
        if tool_index < 0 or tool_index >= len(executions):
            raise MessageCreationFailedException(
                f"Tool index {tool_index} out of range (message has {len(executions)} tool executions)"
            )
        return message, metadata, executions

    @staticmethod
    def _resolution(status: str, message_id: str, tool_index: int, execution: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": status, "execution": execution, "message_id": message_id, "tool_index": tool_index}

    async def resolve_tool_execution(
        self,
        session_id: str,
        user_id: str,
        message_id: str,
        tool_index: int,
        approved: bool,
        reason: Optional[str] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
        is_admin: bool = False,
    ) -> Dict[str, Any]:
        session = self._m._get_session_or_raise(session_id)
        self._m._verify_ownership(session, user_id)

        async with self._lock(message_id):
            _message, metadata, executions = self._load_execution(session_id, message_id, tool_index)
            execution = executions[tool_index]
            status = execution.get("approval_status")
            allowed = (tool_transcript.PENDING, tool_transcript.FAILED) if approved else (tool_transcript.PENDING,)
            if status not in allowed:
                return self._resolution("already_resolved", message_id, tool_index, execution)

            if not approved:
                execution["approval_status"] = tool_transcript.REJECTED
                execution["approval_reason"] = (reason or "")[:tool_transcript.MAX_REASON_CHARS] or None
            else:
                stored_context = metadata.get("context_metadata") or {}
                if (
                    status == tool_transcript.PENDING
                    and not force
                    and context_metadata is not None
                    and await self._is_stale(
                        execution, self.tool_context(session, user_id, is_admin, context_metadata)
                    )
                ):
                    return self._resolution("stale", message_id, tool_index, execution)

                result = await self._run_confirmed(
                    execution, self.tool_context(session, user_id, is_admin, stored_context)
                )
                result_data = {"success": result.success, "data": result.data, "error": result.error}
                execution["result"] = result_data
                execution["approval_status"] = tool_transcript.APPROVED if result.success else tool_transcript.FAILED
                execution["approval_reason"] = None
                self._apply_action_to_context_metadata(result_data, stored_context)
                if stored_context:
                    metadata["context_metadata"] = stored_context

            tool_transcript.set_transcript_result(
                metadata.get("tool_transcript"),
                execution.get("tool_call_id"),
                tool_transcript.resolved_result(execution),
            )
            metadata["tool_executions"] = executions
            if not self._m.chat_repository.transition_tool_execution(message_id, tool_index, status, metadata):
                _message, _metadata, current = self._load_execution(session_id, message_id, tool_index)
                return self._resolution("already_resolved", message_id, tool_index, current[tool_index])

        return self._resolution(execution["approval_status"], message_id, tool_index, execution)

    async def _run_confirmed(self, execution: Dict[str, Any], context: ToolContext) -> ToolResult:
        tool_name = execution.get("tool_name", "")
        if not self._m.tool_executor:
            return ToolResult(success=False, data="", error="Tools are not available for this session")
        try:
            return await self._m.tool_executor.execute_tool_confirmed(
                tool_name, context, execution.get("arguments") or {}
            )
        except Exception as e:
            logger.exception("Confirmed tool %s failed", tool_name)
            return ToolResult(success=False, data="", error=unexpected(tool_name, "confirm", e, context.is_admin))

    @staticmethod
    def _before_values(preview: Optional[Dict[str, Any]]) -> List[Any]:
        if not preview:
            return []
        values: List[Any] = []
        for row in preview.get("fields") or []:
            if isinstance(row, dict) and "old" in row:
                values.append(("field", row.get("label"), row.get("old")))
        for block in preview.get("text_blocks") or []:
            if isinstance(block, dict) and "old_text" in block:
                values.append(("text", block.get("label"), block.get("old_text")))
        for index, change in enumerate(preview.get("changes") or []):
            if isinstance(change, dict) and change.get("before") is not None:
                values.append(("change", index, change.get("before")))
        return values

    async def _is_stale(self, execution: Dict[str, Any], context: ToolContext) -> bool:
        before = self._before_values(execution.get("preview"))
        if not before or not self._m.tool_executor:
            return False
        tool = self._m.tool_executor.tool_registry.get(execution.get("tool_name", ""))
        if tool is None:
            return False
        try:
            result = await tool.execute(context, **(execution.get("arguments") or {}))
        except Exception:
            logger.debug("Stale check preview failed for %s", execution.get("tool_name"), exc_info=True)
            return False
        if not result.success or result.preview is None:
            return False
        current = self._before_values(serialize_approval_preview(result.preview))
        return json.dumps(current, default=str, sort_keys=True) != json.dumps(before, default=str, sort_keys=True)

    async def reject_pending_for_reply(self, session_id: str, content: str) -> Optional[Dict[str, Any]]:
        latest = self._m.chat_repository.get_latest_message(session_id)
        if latest is None or latest.role != "assistant":
            return None
        async with self._lock(latest.id):
            _message, metadata, executions = self._load_execution_or_none(session_id, latest.id)
            if executions is None:
                return None
            index = tool_transcript.pending_index(executions)
            if index is None:
                return None
            execution = executions[index]
            execution["approval_status"] = tool_transcript.REJECTED
            execution["approval_reason"] = (content or "")[:tool_transcript.MAX_REASON_CHARS] or None
            tool_transcript.set_transcript_result(
                metadata.get("tool_transcript"),
                execution.get("tool_call_id"),
                tool_transcript.REPLIED_INSTEAD_RESULT,
            )
            metadata["tool_executions"] = executions
            if not self._m.chat_repository.transition_tool_execution(
                latest.id, index, tool_transcript.PENDING, metadata
            ):
                return None
        return {"id": latest.id, "tool_executions": executions}

    def _load_execution_or_none(self, session_id: str, message_id: str):
        message = self._m.chat_repository.get_message(message_id)
        if not message or message.session_id != session_id:
            return None, None, None
        metadata = dict(message.metadata or {})
        return message, metadata, metadata.get("tool_executions") or []

    def continuation(self, session_id: str, continue_from: Dict[str, Any]) -> Continuation:
        message_id = continue_from.get("message_id") or ""
        tool_index = int(continue_from.get("tool_index", 0))
        message, metadata, executions = self._load_execution(session_id, message_id, tool_index)
        status = executions[tool_index].get("approval_status")
        if status == tool_transcript.PENDING:
            raise ToolExecutionPendingException(f"Tool execution {tool_index} on message {message_id} is still pending")
        if status not in (tool_transcript.APPROVED, tool_transcript.REJECTED, tool_transcript.FAILED):
            raise MessageCreationFailedException(
                f"Tool execution {tool_index} on message {message_id} did not need approval"
            )
        latest = self._m.chat_repository.get_latest_message(session_id)
        if latest is None or latest.id != message_id:
            raise ContinuationConflictException(f"Message {message_id} was already continued")
        chain = self._m.chat_repository.get_turn_chain(session_id, message_id)
        return Continuation(
            message_id=message_id,
            tool_index=tool_index,
            tool_name=executions[tool_index].get("tool_name") or "",
            tool_call_id=executions[tool_index].get("tool_call_id") or "",
            rounds_used=int(metadata.get("turn_rounds") or 0),
            applied_calls=tool_transcript.applied_call_keys(chain),
            context_metadata=metadata.get("context_metadata"),
            loaded_groups=self._loaded_tool_groups(chain),
            turn_message_ids=frozenset(row.id for row in chain),
        )

    def _loaded_tool_groups(self, chain: List[Any]) -> FrozenSet[str]:
        registry = self._m.tool_executor.tool_registry if self._m.tool_executor else None
        groups: Set[str] = set()
        for row in chain:
            for record in (row.metadata or {}).get("tool_executions") or []:
                name = record.get("tool_name", "")
                if name == USE_TOOLS_NAME:
                    group = (record.get("arguments") or {}).get("group")
                    if (record.get("result") or {}).get("success") and isinstance(group, str):
                        groups.add(group)
                    continue
                tool = registry.get(name) if registry else None
                group = getattr(tool, "context_group", None)
                if isinstance(group, str):
                    groups.add(group)
        return frozenset(groups)

    @staticmethod
    def _apply_action_to_context_metadata(
        result_data: Dict[str, Any],
        context_metadata: Dict[str, Any],
    ) -> None:
        if not result_data.get("success") or not isinstance(context_metadata, dict):
            return
        data = result_data.get("data")
        if not isinstance(data, str) or not data.strip():
            return
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(payload, dict):
            return

        form_state = context_metadata.get("form_state")
        if not isinstance(form_state, dict):
            return

        action = payload.get("action")
        if action == "apply_form_changes":
            form_data = dict(form_state.get("form_data") or {})
            for change in payload.get("applied_changes") or []:
                field_name = change.get("field_name") if isinstance(change, dict) else None
                if isinstance(field_name, str) and field_name:
                    form_data[field_name] = change.get("new_value")
            form_state["form_data"] = form_data

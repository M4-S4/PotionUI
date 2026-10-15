from typing import Any, Dict, Iterable, List, Optional, Set

from src.features.llm.tools.base import ToolExecution, serialize_approval_preview
from src.features.llm.tools.executor import _bound_tool_result_content
from src.features.llm.tools.workflow import ToolCallGuard

MAX_REASON_CHARS = 2000

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
FAILED = "failed"

REPLIED_INSTEAD_RESULT = "User rejected this and replied instead."
_MISSING_RESULT = "No result was recorded for this call."


def execution_record(execution: ToolExecution) -> Dict[str, Any]:
    return {
        "tool_name": execution.tool_name,
        "arguments": execution.arguments,
        "result": {
            "success": execution.result.success,
            "data": execution.result.data,
            "error": execution.result.error,
        },
        "duration_ms": execution.duration_ms,
        "approval_status": execution.approval_status,
        "approval_reason": None,
        "skipped": execution.skipped,
        "preview": serialize_approval_preview(execution.result.preview),
        "tool_call_id": execution.tool_call_id,
    }


def tool_metadata(
    tool_executions: List[ToolExecution],
    transcript: Optional[List[Dict[str, Any]]],
    turn_rounds: int,
    context_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "tool_transcript": transcript or [],
        "turn_rounds": turn_rounds,
    }
    if tool_executions:
        metadata["tool_executions"] = [execution_record(te) for te in tool_executions]
        if context_metadata:
            metadata["context_metadata"] = context_metadata
    return metadata


def rejected_result(reason: Optional[str]) -> str:
    if reason:
        return f"User rejected this. Reason: {reason}"
    return "User rejected this."


def resolved_result(record: Dict[str, Any]) -> Optional[str]:
    status = record.get("approval_status")
    result = record.get("result") or {}
    if status == APPROVED:
        return _bound_tool_result_content(result.get("data") or "")
    if status == FAILED:
        return f"Error: {result.get('error') or 'unknown error'}"
    if status == REJECTED:
        return rejected_result(record.get("approval_reason"))
    return None


def set_transcript_result(
    transcript: Optional[List[Dict[str, Any]]], tool_call_id: Optional[str], content: str
) -> None:
    if not transcript or not tool_call_id:
        return
    for round_ in transcript:
        if not any(call.get("id") == tool_call_id for call in round_.get("calls") or []):
            continue
        results = [r for r in round_.get("results") or [] if r.get("tool_call_id") != tool_call_id]
        results.append({"tool_call_id": tool_call_id, "content": content})
        round_["results"] = results
        return


def replay_messages(content: str, metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metadata = metadata or {}
    transcript = metadata.get("tool_transcript") or []
    if not transcript:
        return [{"role": "assistant", "content": content}]

    records = {
        record.get("tool_call_id"): record
        for record in metadata.get("tool_executions") or []
        if record.get("tool_call_id")
    }
    messages: List[Dict[str, Any]] = []
    for round_ in transcript:
        calls = round_.get("calls") or []
        if not calls:
            continue
        results = {r.get("tool_call_id"): r.get("content", "") for r in round_.get("results") or []}
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call.get("name", ""), "arguments": call.get("arguments") or {}},
                }
                for call in calls
            ],
        })
        for call in calls:
            result = results.get(call["id"])
            if result is None:
                result = resolved_result(records.get(call["id"]) or {}) or _MISSING_RESULT
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call.get("name", ""),
                "content": result,
            })
    if (content or "").strip():
        paused = any(record.get("approval_status") for record in metadata.get("tool_executions") or [])
        if paused and messages:
            messages[0]["content"] = content
        else:
            messages.append({"role": "assistant", "content": content})
    return messages


def pending_index(executions: Iterable[Dict[str, Any]]) -> Optional[int]:
    for index, record in enumerate(executions):
        if record.get("approval_status") == PENDING:
            return index
    return None


def applied_call_keys(rows: Iterable[Any]) -> Set[str]:
    keys: Set[str] = set()
    for row in rows:
        for record in (row.metadata or {}).get("tool_executions") or []:
            if record.get("approval_status") == APPROVED:
                keys.add(ToolCallGuard.call_key(record.get("tool_name", ""), record.get("arguments") or {}))
    return keys

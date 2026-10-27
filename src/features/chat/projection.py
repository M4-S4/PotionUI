from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.features.chat.dto import MessageResponse, NoticeResponse, ReasoningBlock, UsageResponse
from src.features.llm.harness.events import Usage
from src.features.llm.harness.messages import ReasoningPart, Role, ToolCallPart, ToolResultPart
from src.features.llm.harness.turn_state import StepRecord, TurnState
from src.features.llm.tools.base import ToolExecution, serialize_approval_preview

STEP_RANGE_KEY = "step_range"

_COUNT_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "cached_input_tokens", "cache_write_tokens")


def step_range(metadata: Optional[Mapping[str, Any]]) -> Optional[Tuple[int, int]]:
    window = (metadata or {}).get(STEP_RANGE_KEY)
    if not isinstance(window, Mapping):
        return None
    first, last = window.get("first"), window.get("last")
    if isinstance(first, int) and isinstance(last, int) and not isinstance(first, bool) and not isinstance(last, bool):
        return first, last
    return None


def segment(steps: Sequence[StepRecord], first: int, last: int) -> List[StepRecord]:
    return sorted((step for step in steps if first <= step.seq <= last), key=lambda step: step.seq)


def segment_text(steps: Sequence[StepRecord]) -> str:
    text = ""
    for step in steps:
        piece = step.message.text() if step.message.role is Role.ASSISTANT else ""
        if not piece:
            continue
        if text and not text[-1].isspace() and not piece[0].isspace():
            text += "\n\n"
        text += piece
    return text


def reasoning_block(part: ReasoningPart) -> ReasoningBlock:
    if part.text:
        kind, text = "raw", part.text
    elif part.summary:
        kind, text = "summary", part.summary
    else:
        kind, text = "omitted", ""
    return ReasoningBlock(kind=kind, text=text, redacted=part.redacted, duration_ms=part.duration_ms)


def reasoning_blocks(steps: Sequence[StepRecord]) -> List[ReasoningBlock]:
    return [
        reasoning_block(part)
        for step in steps
        if step.message.role is Role.ASSISTANT
        for part in step.message.reasoning()
    ]


def segment_usage(steps: Sequence[StepRecord]) -> Optional[UsageResponse]:
    usages: List[Usage] = [step.usage for step in steps if step.usage is not None]
    if not usages:
        return None
    totals: Dict[str, Any] = {}
    for name in _COUNT_FIELDS:
        counted = [getattr(usage, name) for usage in usages if getattr(usage, name) is not None]
        totals[name] = sum(counted) if counted else None
    costs = [usage.cost for usage in usages if usage.cost is not None]
    totals["cost"] = sum(costs) if costs else None
    return UsageResponse(**totals)


def segment_notices(state: Optional[TurnState], first: int, last: int, closes_turn: bool) -> List[NoticeResponse]:
    if state is None:
        return []
    return [
        NoticeResponse(kind=notice.kind.value, message=notice.message, step_seq=notice.step_seq)
        for notice in state.notices
        if (notice.step_seq is None and closes_turn)
        or (notice.step_seq is not None and first <= notice.step_seq <= last)
    ]


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


def tool_executions(
    steps: Sequence[StepRecord],
    state: Optional[TurnState],
    records: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    records = records or {}
    results: Dict[str, ToolResultPart] = {}
    for step in steps:
        if step.message.role is Role.TOOL:
            for result in step.message.tool_results():
                results[result.call_id] = result
    pending = {}
    if state is not None:
        for result in state.gathered:
            results.setdefault(result.call_id, result)
        pending = {call.call_id: call for call in state.pending}
    executions: List[Dict[str, Any]] = []
    for step in steps:
        if step.message.role is not Role.ASSISTANT:
            continue
        for call in step.message.tool_calls():
            executions.append(_execution(call, results.get(call.call_id), pending.get(call.call_id), records.get(call.call_id)))
    return executions


def _execution(
    call: ToolCallPart,
    result: Optional[ToolResultPart],
    pending: Any,
    record: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    record = record or {}
    if pending is not None:
        outcome = {"success": True, "data": "", "error": None}
        status = "pending"
        preview = pending.preview
    else:
        failed = result.is_error if result is not None else False
        text = result.text() if result is not None else ""
        outcome = record.get("result") or {
            "success": not failed,
            "data": "" if failed else text,
            "error": text if failed else None,
        }
        status = record.get("approval_status")
        preview = record.get("preview")
    return {
        "tool_name": call.name,
        "arguments": call.args or {},
        "result": outcome,
        "duration_ms": record.get("duration_ms"),
        "approval_status": status,
        "approval_reason": record.get("approval_reason"),
        "skipped": bool(record.get("skipped")),
        "preview": preview,
        "tool_call_id": call.call_id,
    }


def assistant_metadata(
    steps: Sequence[StepRecord],
    state: TurnState,
    *,
    records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    context_metadata: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    ordered = sorted(steps, key=lambda step: step.seq)
    metadata: Dict[str, Any] = dict(extra or {})
    metadata[STEP_RANGE_KEY] = {"first": ordered[0].seq, "last": ordered[-1].seq}
    metadata["turn_rounds"] = state.rounds_used
    usage = segment_usage(ordered)
    if usage is not None:
        total = usage.input_tokens + usage.output_tokens if None not in (usage.input_tokens, usage.output_tokens) else None
        last = next((step.usage for step in reversed(ordered) if step.usage is not None), None)
        metadata["tokens_used"] = total
        metadata["prompt_tokens"] = last.input_tokens if last is not None else None
        metadata["completion_tokens"] = usage.output_tokens
    executions = tool_executions(ordered, state, records)
    if executions:
        metadata["tool_executions"] = executions
        if context_metadata:
            metadata["context_metadata"] = dict(context_metadata)
    return metadata


def metadata_without_reasoning(
    metadata: Optional[Dict[str, Any]], keep_visibility_flag: bool = False,
) -> Optional[Dict[str, Any]]:
    if not metadata:
        return metadata
    transcript = metadata.get("tool_transcript")
    trace = metadata.get("behavior_trace")
    carries_round = isinstance(transcript, list) and any(isinstance(item, dict) and "reasoning" in item for item in transcript)
    carries_trace = isinstance(trace, dict) and "thinking_visible" in trace and not keep_visibility_flag
    if "thinking" not in metadata and not carries_round and not carries_trace:
        return metadata
    cleaned = {key: value for key, value in metadata.items() if key != "thinking"}
    if carries_round:
        cleaned["tool_transcript"] = [
            {key: value for key, value in item.items() if key != "reasoning"} if isinstance(item, dict) else item
            for item in transcript
        ]
    if carries_trace:
        cleaned["behavior_trace"] = {key: value for key, value in trace.items() if key != "thinking_visible"}
    return cleaned


def present(
    messages: Sequence[MessageResponse],
    steps_by_turn: Mapping[str, Sequence[StepRecord]],
    states: Mapping[str, TurnState],
    *,
    show_thinking: bool,
) -> List[MessageResponse]:
    presented: List[MessageResponse] = []
    for item in messages:
        update: Dict[str, Any] = {}
        cleaned = metadata_without_reasoning(item.metadata, keep_visibility_flag=show_thinking)
        if cleaned is not item.metadata:
            update["metadata"] = cleaned
        window = step_range(item.metadata) if item.role == "assistant" and item.turn_id else None
        if window is not None:
            turn_steps = steps_by_turn.get(item.turn_id, ())
            steps = segment(turn_steps, *window)
            if steps:
                state = states.get(item.turn_id)
                closes_turn = steps[-1].seq >= max(step.seq for step in turn_steps)
                update["usage"] = segment_usage(steps)
                notices = segment_notices(state, steps[0].seq, steps[-1].seq, closes_turn)
                if notices:
                    update["notices"] = notices
                if show_thinking:
                    update["reasoning"] = reasoning_blocks(steps)
        presented.append(item.model_copy(update=update) if update else item)
    return presented

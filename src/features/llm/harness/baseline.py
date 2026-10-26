import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Union

from src.features.llm.harness.errors import HarnessError
from src.features.llm.harness.messages import ImagePart, TextPart, ToolCallPart, ToolResultPart
from src.features.llm.harness.turn_state import (
    DecisionKind,
    PendingCall,
    StepRecord,
    ToolFinished,
    ToolStarted,
    TurnState,
    TurnStatus,
)
from src.features.llm.harness.build import JPEG
from src.features.llm.tools.base import ToolExecution, ToolResult, serialize_approval_preview

class InMemoryTurnStore:
    def __init__(self) -> None:
        self._steps: Dict[str, List[StepRecord]] = {}
        self._states: Dict[str, TurnState] = {}

    def steps(self, session_id: str) -> List[StepRecord]:
        return list(self._steps.get(session_id, []))

    def next_seq(self, session_id: str) -> int:
        return len(self._steps.get(session_id, []))

    def append(self, step: StepRecord) -> None:
        recorded = self._steps.setdefault(step.session_id, [])
        if step.seq != len(recorded):
            raise ValueError(f"step seq {step.seq} is not the next seq {len(recorded)} of session {step.session_id}")
        recorded.append(step)

    def save_state(self, state: TurnState) -> None:
        self._states[state.turn_id] = state

    def load_state(self, turn_id: str) -> Optional[TurnState]:
        return self._states.get(turn_id)

    def open_turn(self, session_id: str) -> Optional[TurnState]:
        waiting = [
            state for state in self._states.values()
            if state.session_id == session_id and state.status is TurnStatus.AWAITING_DECISION
        ]
        return waiting[-1] if waiting else None


class NoRetry:
    def delay(self, error: HarnessError, attempt: int) -> Optional[float]:
        return None


def _pending_note(name: str) -> str:
    return f"Not run: waiting for approval of {name}. Call it again if still needed."


class SequentialToolRunner:
    def __init__(self, tools: Any, clock: Callable[[], float] = time.monotonic) -> None:
        self._tools = tools
        self._clock = clock

    async def run(
        self, calls: Sequence[ToolCallPart], state: TurnState, context: Any,
    ) -> AsyncIterator[Union[ToolStarted, ToolFinished, PendingCall]]:
        for position, call in enumerate(calls):
            arguments = dict(call.args or {})
            yield ToolStarted(call)
            started = self._clock()
            result, pending = await self._tools.execute_tool_guarded(
                call.name, context.tool_context, arguments, context.allowed_tools, context.guard, context.deferred_tools,
            )
            execution = ToolExecution(
                tool_name=call.name,
                arguments=arguments,
                result=result,
                duration_ms=int((self._clock() - started) * 1000),
                approval_status="pending" if pending else None,
                tool_call_id=call.call_id,
            )
            context.executions.append(execution)
            if pending:
                yield ToolFinished(ToolResultPart(call_id=call.call_id, name=call.name, content=()), record=execution)
                yield PendingCall(
                    call_id=call.call_id, name=call.name, decision=DecisionKind.APPROVAL,
                    preview=serialize_approval_preview(result.preview),
                )
                for skipped in calls[position + 1:]:
                    yield self._skipped(skipped, call.name, context)
                return
            yield ToolFinished(await self._result_part(call, result, context), record=execution)

    def _skipped(self, call: ToolCallPart, pending_name: str, context: Any) -> ToolFinished:
        execution = ToolExecution(
            tool_name=call.name,
            arguments=dict(call.args or {}),
            result=ToolResult(success=False, data="", error=f"Not run: waiting for approval of {pending_name}."),
            duration_ms=0,
            skipped=True,
            tool_call_id=call.call_id,
        )
        context.executions.append(execution)
        part = ToolResultPart(
            call_id=call.call_id, name=call.name, content=(TextPart(text=_pending_note(pending_name)),), is_error=True,
        )
        return ToolFinished(part, record=execution)

    async def _result_part(self, call: ToolCallPart, result: ToolResult, context: Any) -> ToolResultPart:
        text = await self._tools.tool_result_text(context.tool_context, result)
        content: List[Any] = [TextPart(text=text)]
        if result.image_data:
            content.append(ImagePart(media_type=JPEG, data=result.image_data))
        return ToolResultPart(call_id=call.call_id, name=call.name, content=tuple(content), is_error=not result.success)

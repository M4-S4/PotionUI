import asyncio
import logging
import time
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from src.features.llm.harness.build import JPEG
from src.features.llm.harness.messages import ImagePart, TextPart, ToolCallPart, ToolResultPart
from src.features.llm.harness.turn_state import (
    Decision,
    DecisionKind,
    PendingCall,
    ToolFinished,
    ToolStarted,
    TurnState,
)
from src.features.llm.tools.base import ToolExecution, ToolResult, serialize_approval_preview
from src.features.llm.tools.errors import unexpected

logger = logging.getLogger(__name__)

MAX_CONCURRENCY = 8
MAX_REASON_CHARS = 2000

APPROVED = "approved"
REJECTED = "rejected"
FAILED = "failed"
PENDING = "pending"

RunnerEvent = Union[ToolStarted, ToolFinished, PendingCall]


class IncompleteBatch(ValueError):
    pass


@dataclass
class _Done:
    result: ToolResult
    duration_ms: int = 0
    pending: bool = False
    skipped: bool = False
    status: Optional[str] = None


def error_result(call: ToolCallPart, text: str) -> ToolResultPart:
    return ToolResultPart(call_id=call.call_id, name=call.name, content=(TextPart(text),), is_error=True)


def unparseable_call_error(call: ToolCallPart) -> str:
    return (
        f"The call to {call.name} was cut off or its arguments were not valid JSON, so it was not run. "
        "Send the call again with complete arguments."
    )


def rejection_text(reason: Optional[str]) -> str:
    reason = (reason or "").strip()[:MAX_REASON_CHARS]
    return f"User rejected this. Reason: {reason}" if reason else "User rejected this."


def waiting_note(name: str) -> str:
    return f"Not run: waiting for approval of {name}. Call it again if still needed."


def batch_results(calls: Sequence[ToolCallPart], results: Iterable[ToolResultPart]) -> Tuple[ToolResultPart, ...]:
    by_id: Dict[str, ToolResultPart] = {result.call_id: result for result in results}
    missing = [call.call_id for call in calls if call.call_id not in by_id]
    if missing:
        raise IncompleteBatch(f"no result for call(s) {', '.join(missing)}")
    return tuple(by_id[call.call_id] for call in calls)


class ParallelToolRunner:
    def __init__(
        self,
        tools: Any,
        clock: Callable[[], float] = time.monotonic,
        max_concurrency: int = MAX_CONCURRENCY,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._tools = tools
        self._clock = clock
        self._max_concurrency = max_concurrency

    async def run(
        self,
        calls: Sequence[ToolCallPart],
        state: TurnState,
        context: Any,
        decisions: Sequence[Decision] = (),
    ) -> AsyncIterator[RunnerEvent]:
        waiting = {item.call_id: item for item in state.pending}
        approvable = {key for key, item in waiting.items() if item.decision is DecisionKind.APPROVAL}
        verdicts = {item.call_id: item for item in decisions}
        stray = sorted(set(verdicts) - approvable)
        if stray:
            raise ValueError(f"no approval is pending for call(s) {', '.join(stray)}")
        done = {part.call_id for part in state.gathered}
        todo = [call for call in calls if call.call_id not in done]
        pending_name: Optional[str] = None
        index = 0
        while index < len(todo):
            call = todo[index]
            if call.call_id in waiting:
                index += 1
                verdict = verdicts.get(call.call_id)
                if verdict is None:
                    pending_name = pending_name or call.name
                    yield waiting[call.call_id]
                    continue
                stream = self._decide(call, verdict, context)
            elif self._concurrent(call):
                end = index + 1
                while end < len(todo) and todo[end].call_id not in waiting and self._concurrent(todo[end]):
                    end += 1
                stream = self._run_group(todo[index:end], context)
                index = end
            else:
                index += 1
                stream = self._skip(call, pending_name, context) if pending_name else self._run_one(call, context)
            async with aclosing(stream) as events:
                async for event in events:
                    if isinstance(event, PendingCall):
                        pending_name = pending_name or event.name
                    yield event

    def _concurrent(self, call: ToolCallPart) -> bool:
        if call.args is None:
            return True
        tool = self._tools.tool_registry.get(call.name)
        return tool is not None and (tool.read_only or tool.requires_approval)

    async def _run_one(self, call: ToolCallPart, context: Any) -> AsyncIterator[RunnerEvent]:
        yield ToolStarted(call)
        for event in await self._settle(call, await self._execute(call, None, context), context):
            yield event

    async def _run_group(self, group: Sequence[ToolCallPart], context: Any) -> AsyncIterator[RunnerEvent]:
        for call in group:
            yield ToolStarted(call)
        gate = asyncio.Semaphore(self._max_concurrency)
        tasks: List[asyncio.Future] = [asyncio.ensure_future(self._execute(call, gate, context)) for call in group]
        try:
            for call, task in zip(group, tasks):
                for event in await self._settle(call, await task, context):
                    yield event
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _skip(self, call: ToolCallPart, pending_name: str, context: Any) -> AsyncIterator[RunnerEvent]:
        note = waiting_note(pending_name)
        result = ToolResult(success=False, data="", error=f"Not run: waiting for approval of {pending_name}.")
        record = self._record(call, _Done(result, skipped=True), context)
        yield ToolFinished(error_result(call, note), record=record)

    async def _decide(self, call: ToolCallPart, verdict: Decision, context: Any) -> AsyncIterator[RunnerEvent]:
        if not verdict.approved:
            text = rejection_text(verdict.reason)
            done = _Done(ToolResult(success=False, data="", error=text), status=REJECTED)
            yield ToolFinished(error_result(call, text), record=self._record(call, done, context))
            return
        yield ToolStarted(call)
        started = self._clock()
        try:
            result = await self._tools.execute_tool_confirmed(call.name, context.tool_context, dict(call.args or {}))
        except Exception as error:
            logger.exception("tool %s raised while confirming", call.name)
            result = ToolResult(
                success=False, data="", error=unexpected(call.name, "confirm", error, _is_admin(context)),
            )
        done = _Done(result, int((self._clock() - started) * 1000), status=APPROVED if result.success else FAILED)
        yield ToolFinished(await self._part(call, result, context), record=self._record(call, done, context))

    async def _execute(self, call: ToolCallPart, gate: Optional[asyncio.Semaphore], context: Any) -> _Done:
        if call.args is None:
            return _Done(ToolResult(success=False, data="", error=unparseable_call_error(call)))
        if gate is None:
            return await self._invoke(call, context)
        async with gate:
            return await self._invoke(call, context)

    async def _invoke(self, call: ToolCallPart, context: Any) -> _Done:
        started = self._clock()
        try:
            result, pending = await self._tools.execute_tool_guarded(
                call.name, context.tool_context, dict(call.args), context.allowed_tools, context.guard,
                context.deferred_tools,
            )
        except Exception as error:
            logger.exception("tool %s raised while running", call.name)
            result = ToolResult(success=False, data="", error=unexpected(call.name, "run", error, _is_admin(context)))
            pending = False
        return _Done(result, int((self._clock() - started) * 1000), pending=pending, status=PENDING if pending else None)

    async def _settle(self, call: ToolCallPart, done: _Done, context: Any) -> List[RunnerEvent]:
        record = self._record(call, done, context)
        if done.pending:
            placeholder = ToolResultPart(call_id=call.call_id, name=call.name, content=())
            preview = serialize_approval_preview(done.result.preview)
            return [
                ToolFinished(placeholder, record=record),
                PendingCall(call_id=call.call_id, name=call.name, decision=DecisionKind.APPROVAL, preview=preview),
            ]
        if call.args is None:
            return [ToolFinished(error_result(call, unparseable_call_error(call)), record=record)]
        return [ToolFinished(await self._part(call, done.result, context), record=record)]

    @staticmethod
    def _record(call: ToolCallPart, done: _Done, context: Any) -> ToolExecution:
        execution = ToolExecution(
            tool_name=call.name,
            arguments=dict(call.args or {}),
            result=done.result,
            duration_ms=done.duration_ms,
            approval_status=done.status,
            skipped=done.skipped,
            tool_call_id=call.call_id,
        )
        context.executions.append(execution)
        return execution

    async def _part(self, call: ToolCallPart, result: ToolResult, context: Any) -> ToolResultPart:
        text = await self._tools.tool_result_text(context.tool_context, result)
        content: List[Any] = [TextPart(text=text)]
        if result.image_data:
            content.append(ImagePart(media_type=JPEG, data=result.image_data))
        return ToolResultPart(call_id=call.call_id, name=call.name, content=tuple(content), is_error=not result.success)


def _is_admin(context: Any) -> bool:
    return bool(getattr(context.tool_context, "is_admin", False))

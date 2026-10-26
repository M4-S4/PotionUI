import asyncio
import logging
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union

from src.features.llm.harness.messages import TextPart, ToolCallPart, ToolResultPart
from src.features.llm.harness.turn_state import Decision, PendingCall, DecisionKind, ToolFinished, ToolStarted, TurnState

logger = logging.getLogger(__name__)

MAX_CONCURRENCY = 8
MAX_REASON_CHARS = 2000

RunnerEvent = Union[ToolStarted, ToolFinished, PendingCall]


class IncompleteBatch(ValueError):
    pass


@dataclass(frozen=True)
class ToolOutcome:
    result: ToolResultPart
    record: Any = None
    pending: bool = False
    preview: Any = None


class ToolBackend(Protocol):
    def parallel_safe(self, name: str) -> bool: ...

    async def execute(self, call: ToolCallPart, context: Any, state: TurnState) -> ToolOutcome: ...

    async def confirm(self, call: ToolCallPart, context: Any, state: TurnState) -> ToolOutcome: ...


def error_result(call: ToolCallPart, text: str) -> ToolResultPart:
    return ToolResultPart(call_id=call.call_id, name=call.name, content=(TextPart(text),), is_error=True)


def truncated_call_result(call: ToolCallPart) -> ToolResultPart:
    return error_result(
        call,
        f"The call to {call.name} was cut off or its arguments were not valid JSON, so it was not run. "
        "Send the call again with complete arguments.",
    )


def denied_result(call: ToolCallPart, reason: Optional[str]) -> ToolResultPart:
    reason = (reason or "").strip()[:MAX_REASON_CHARS]
    return error_result(call, f"User rejected this. Reason: {reason}" if reason else "User rejected this.")


def skipped_result(call: ToolCallPart, reason: str) -> ToolResultPart:
    return error_result(call, f"Not run: {reason}")


def skipped_results(calls: Sequence[ToolCallPart], resolved: Iterable[str], reason: str) -> Tuple[ToolResultPart, ...]:
    done = set(resolved)
    return tuple(skipped_result(call, reason) for call in calls if call.call_id not in done)


def batch_results(calls: Sequence[ToolCallPart], results: Iterable[ToolResultPart]) -> Tuple[ToolResultPart, ...]:
    by_id: Dict[str, ToolResultPart] = {result.call_id: result for result in results}
    missing = [call.call_id for call in calls if call.call_id not in by_id]
    if missing:
        raise IncompleteBatch(f"no result for call(s) {', '.join(missing)}")
    return tuple(by_id[call.call_id] for call in calls)


def unresolved_calls(calls: Sequence[ToolCallPart], state: TurnState) -> Tuple[ToolCallPart, ...]:
    done = {part.call_id for part in state.gathered}
    return tuple(call for call in calls if call.call_id not in done)


class ParallelToolRunner:
    def __init__(self, backend: ToolBackend, max_concurrency: int = MAX_CONCURRENCY) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._backend = backend
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
        todo = unresolved_calls(calls, state)
        blocked = False
        index = 0
        while index < len(todo):
            call = todo[index]
            if call.call_id in waiting:
                verdict = verdicts.get(call.call_id)
                index += 1
                if verdict is None:
                    blocked = True
                    yield waiting[call.call_id]
                    continue
                async with aclosing(self._decide(call, verdict, state, context)) as stream:
                    async for event in stream:
                        yield event
                continue
            if self._concurrent(call):
                end = index + 1
                while end < len(todo) and todo[end].call_id not in waiting and self._concurrent(todo[end]):
                    end += 1
                group = todo[index:end]
                index = end
                stream = self._run_group(group, state, context)
            else:
                index += 1
                if blocked:
                    continue
                stream = self._run_one(call, state, context)
            async with aclosing(stream) as events:
                async for event in events:
                    blocked = blocked or isinstance(event, PendingCall)
                    yield event

    def _concurrent(self, call: ToolCallPart) -> bool:
        return call.args is None or self._backend.parallel_safe(call.name)

    async def _run_one(self, call: ToolCallPart, state: TurnState, context: Any) -> AsyncIterator[RunnerEvent]:
        yield ToolStarted(call)
        outcome = await self._guarded(self._backend.execute, call, state, context)
        yield _settle(call, outcome)

    async def _run_group(
        self, group: Sequence[ToolCallPart], state: TurnState, context: Any,
    ) -> AsyncIterator[RunnerEvent]:
        for call in group:
            yield ToolStarted(call)
        gate = asyncio.Semaphore(self._max_concurrency)
        tasks: List[asyncio.Future] = [
            asyncio.ensure_future(self._execute(call, gate, state, context)) for call in group
        ]
        try:
            for call, task in zip(group, tasks):
                yield _settle(call, await task)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute(
        self, call: ToolCallPart, gate: asyncio.Semaphore, state: TurnState, context: Any,
    ) -> ToolOutcome:
        if call.args is None:
            return ToolOutcome(truncated_call_result(call))
        async with gate:
            return await self._guarded(self._backend.execute, call, state, context)

    async def _decide(
        self, call: ToolCallPart, verdict: Decision, state: TurnState, context: Any,
    ) -> AsyncIterator[RunnerEvent]:
        yield ToolStarted(call)
        if not verdict.approved:
            yield ToolFinished(denied_result(call, verdict.reason))
            return
        outcome = await self._guarded(self._backend.confirm, call, state, context)
        yield ToolFinished(outcome.result, outcome.record)

    @staticmethod
    async def _guarded(run, call: ToolCallPart, state: TurnState, context: Any) -> ToolOutcome:
        try:
            return await run(call, context, state)
        except Exception:
            logger.exception("tool %s raised while running", call.name)
            return ToolOutcome(error_result(
                call,
                f"{call.name} failed unexpectedly on the server. "
                "This is not something you can fix by changing your call - tell the user, or try once more.",
            ))


def _settle(call: ToolCallPart, outcome: ToolOutcome) -> RunnerEvent:
    if outcome.pending:
        return PendingCall(call.call_id, call.name, DecisionKind.APPROVAL, outcome.preview)
    return ToolFinished(outcome.result, outcome.record)

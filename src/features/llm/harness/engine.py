import asyncio
import logging
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Awaitable, Callable, ClassVar, Dict, FrozenSet, List, Mapping, Optional, Tuple

from src.features.llm.harness.adapter import Settings, ToolSpec
from src.features.llm.harness.assembler import StepAssembler, parse_args
from src.features.llm.harness.build import context_message, new_id
from src.features.llm.harness.call_filter import TEXT, ToolCallTextFilter
from src.features.llm.harness.errors import HarnessError
from src.features.llm.harness.events import FinishReason, ReasoningDelta, TextDelta, Usage
from src.features.llm.harness.messages import Message, Part, ReasoningPart, Role, TextPart, ToolCallPart, ToolResultPart
from src.features.llm.harness.ports import RetryPolicy, ToolRunner, TurnStore
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.harness.recovery import CANCELLED_REASON, FAILED_REASON, interrupted_tool_step
from src.features.llm.harness.retry import retry_notice
from src.features.llm.harness.truncation import TruncationAction, decide_truncation
from src.features.llm.harness.turn_state import (
    Notice,
    NoticeRaised,
    PendingCall,
    StepCommitted,
    StepRecord,
    StepStatus,
    ToolFinished,
    TurnSnapshot,
    TurnState,
    TurnStatus,
)
from src.features.llm.tools import tool_call_rescue
from src.features.llm.tools.base import ToolExecution
from src.features.llm.tools.workflow import ToolCallGuard, parse_xml_tool_calls, strip_tool_call_xml

logger = logging.getLogger(__name__)

TOOL_BUDGET_EXHAUSTED_MESSAGE = (
    "Tool budget for this turn is exhausted — answer now with what you "
    "have; say plainly what remains undone."
)
MAX_RESCUE_RETRIES = 2
EMPTY_RESPONSE_ATTEMPTS = 3

ANSWER = "answer"
TRUNCATED = "truncated"
AMBIGUOUS = "ambiguous"
BUDGET = "budget"
PENDING = "pending"


@dataclass(frozen=True)
class TurnInput:
    session_id: str
    llm_id: str
    system: str
    history: Tuple[Message, ...]
    new_messages: Tuple[Message, ...] = ()
    settings: Settings = field(default_factory=Settings)
    tool_context: Any = None
    allowed_tools: Optional[Tuple[str, ...]] = None
    deferred_tools: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    max_rounds: int = 20
    iteration_nudge: Optional[str] = None
    applied_calls: FrozenSet[str] = frozenset()
    mode: Optional[str] = None
    turn_id: Optional[str] = None


@dataclass
class ToolRound:
    tool_context: Any
    allowed_tools: Optional[List[str]]
    deferred_tools: Dict[str, List[str]]
    guard: ToolCallGuard
    executions: List[ToolExecution] = field(default_factory=list)


@dataclass(frozen=True)
class BudgetExhausted:
    TYPE: ClassVar[str] = "budget_exhausted"
    max_rounds: int


@dataclass(frozen=True)
class TurnOutcome:
    TYPE: ClassVar[str] = "turn_outcome"
    content: str
    reason: str
    state: TurnState
    executions: Tuple[ToolExecution, ...]
    transcript: Tuple[Dict[str, Any], ...]
    rescues: Optional[List[Dict[str, Any]]]
    tool_failures: Optional[Dict[str, int]]
    rounds_used: int
    usage: Optional[Usage]
    finish: Optional[FinishReason]
    finish_raw: Optional[str]
    steps: Tuple[StepRecord, ...]

    @property
    def pending(self) -> bool:
        return self.reason == PENDING

    @property
    def model_generated(self) -> bool:
        return self.reason in (ANSWER, BUDGET)

    @property
    def completion(self) -> Optional[Dict[str, Any]]:
        if not self.model_generated or self.finish is None:
            return None
        return {"reason": self.finish.value, "raw": self.finish_raw}

    @property
    def thinking_mode(self) -> Optional[Dict[str, Any]]:
        raw = self.usage.raw if self.usage is not None else None
        value = raw.get("thinking_mode") if isinstance(raw, dict) else None
        return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class _Dispatch:
    calls: Tuple[ToolCallPart, ...]
    text: str


@dataclass(frozen=True)
class _Finish:
    content: str
    reason: str
    text: str


@dataclass(frozen=True)
class _Steer:
    text: str
    nudge: str


class _StepBox:
    def __init__(self) -> None:
        self.assembler: Optional[StepAssembler] = None
        self.message: Optional[Message] = None
        self.profile: Optional[CapabilityProfile] = None


class TurnEngine:
    def __init__(
        self,
        gateway: Any,
        tools: Any,
        runner: ToolRunner,
        store: TurnStore,
        retry: RetryPolicy,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self._gateway = gateway
        self._tools = tools
        self._runner = runner
        self._store = store
        self._retry = retry
        self._sleep = sleep

    async def run(self, turn: TurnInput) -> AsyncIterator[Any]:
        run = _Run(self, turn)
        async with aclosing(run.events()) as events:
            async for event in events:
                yield event


class _Run:
    def __init__(self, engine: TurnEngine, turn: TurnInput) -> None:
        self.engine = engine
        self.turn = turn
        self.turn_id = turn.turn_id or new_id("turn")
        self.working: List[Message] = list(turn.history)
        self.steps: List[StepRecord] = []
        self.first_seq: Optional[int] = None
        self.round = ToolRound(
            tool_context=turn.tool_context,
            allowed_tools=list(turn.allowed_tools) if turn.allowed_tools is not None else None,
            deferred_tools={group: list(names) for group, names in turn.deferred_tools.items()},
            guard=ToolCallGuard(set(turn.applied_calls)),
        )
        self.settings = turn.settings
        self.rounds_used = 0
        self.rescue_retries = 0
        self.rescues: List[Dict[str, Any]] = []
        self.notices: List[Notice] = []
        self.nudged = False
        self.profile: Optional[CapabilityProfile] = None
        self.initial_tools: Tuple[ToolSpec, ...] = ()
        self.last_usage: Optional[Usage] = None
        self.last_finish: Optional[FinishReason] = None
        self.last_finish_raw: Optional[str] = None
        self.pending: Tuple[PendingCall, ...] = ()
        self.gathered: Tuple[ToolResultPart, ...] = ()
        self.box: Optional[_StepBox] = None
        self.finished = False

    @property
    def tools_on(self) -> bool:
        return self.turn.tool_context is not None

    async def events(self) -> AsyncIterator[Any]:
        try:
            self.engine._store.save_state(self._state(TurnStatus.RUNNING))
            for message in self.turn.new_messages:
                yield self._commit(message, StepStatus.COMPLETED)
            if self.tools_on:
                self.initial_tools = self._tool_specs()
                async for event in self._tool_loop():
                    yield event
            else:
                async for event in self._single_answer():
                    yield event
        except (asyncio.CancelledError, GeneratorExit):
            self._abort(StepStatus.CANCELLED, TurnStatus.CANCELLED, None)
            raise
        except Exception as error:
            self._abort(StepStatus.FAILED, TurnStatus.FAILED, error)
            raise

    async def _single_answer(self) -> AsyncIterator[Any]:
        box = _StepBox()
        async for event in self._step(box, (), filtered=False, retry_empty=False):
            yield event
        message = box.message
        cut_off = self._cut_off(message)
        if cut_off:
            yield cut_off
        yield self._commit(message, StepStatus.INCOMPLETE if cut_off else StepStatus.COMPLETED, final=True)
        yield self._outcome(message.text(), ANSWER, TurnStatus.COMPLETED)

    async def _tool_loop(self) -> AsyncIterator[Any]:
        for _ in range(self.turn.max_rounds):
            self.rounds_used += 1
            box = _StepBox()
            async for event in self._step(box, self._tool_specs(), filtered=True, retry_empty=True):
                yield event
            decision = self._decide(box.message)
            if isinstance(decision, _Steer):
                self.rescue_retries += 1
                yield self._commit(self._rebuilt(box.message, decision.text, ()), StepStatus.COMPLETED)
                yield self._commit(context_message("rescue", decision.nudge), StepStatus.COMPLETED)
                continue
            if isinstance(decision, _Finish):
                cut_off = self._cut_off(box.message) if decision.reason == ANSWER else None
                if cut_off:
                    yield cut_off
                status = StepStatus.COMPLETED if decision.reason == ANSWER and not cut_off else StepStatus.INCOMPLETE
                yield self._commit(self._rebuilt(box.message, decision.text, ()), status, final=True)
                yield self._outcome(decision.content, decision.reason, TurnStatus.COMPLETED)
                return
            yield self._commit(self._rebuilt(box.message, decision.text, decision.calls), StepStatus.COMPLETED, final=True)
            results: List[ToolResultPart] = []
            pending: List[PendingCall] = []
            async with aclosing(self.engine._runner.run(decision.calls, self._state(TurnStatus.RUNNING), self.round)) as items:
                async for item in items:
                    if isinstance(item, PendingCall):
                        pending.append(item)
                        continue
                    if isinstance(item, ToolFinished):
                        results.append(item.result)
                        self.gathered = tuple(result for result in results if result.content)
                    yield item
            if pending:
                pending_ids = {call.call_id for call in pending}
                self.pending = tuple(pending)
                self.gathered = tuple(result for result in results if result.call_id not in pending_ids)
                yield self._outcome("", PENDING, TurnStatus.AWAITING_DECISION)
                return
            yield self._commit(
                Message(id=new_id("msg"), role=Role.TOOL, parts=tuple(results)), StepStatus.COMPLETED,
            )
            self.gathered = ()
            if self.turn.iteration_nudge and not self.nudged:
                self.nudged = True
                yield self._commit(context_message("nudge", self.turn.iteration_nudge), StepStatus.COMPLETED)

        logger.warning("[TurnEngine] tool budget of %s round(s) reached, making a final call without tools", self.turn.max_rounds)
        yield BudgetExhausted(self.turn.max_rounds)
        yield self._commit(context_message("budget", TOOL_BUDGET_EXHAUSTED_MESSAGE), StepStatus.COMPLETED)
        box = _StepBox()
        async for event in self._step(box, (), filtered=True, retry_empty=False):
            yield event
        text = strip_tool_call_xml(box.message.text())
        cut_off = self._cut_off(box.message)
        if cut_off:
            yield cut_off
        yield self._commit(
            self._rebuilt(box.message, text, ()), StepStatus.INCOMPLETE if cut_off else StepStatus.COMPLETED, final=True,
        )
        yield self._outcome(text, BUDGET, TurnStatus.COMPLETED)

    async def _step(
        self, box: _StepBox, tools: Tuple[ToolSpec, ...], *, filtered: bool, retry_empty: bool,
    ) -> AsyncIterator[Any]:
        self.box = box
        retry_attempt = 0
        empty_attempt = 0
        truncation_attempt = 0
        while True:
            prepared = await self.engine._gateway.prepare(
                self.turn.llm_id, tuple(self.working),
                system=self.turn.system, settings=self.settings, tools=tools,
            )
            if self.profile is None:
                self.profile = prepared.profile
            box.assembler = StepAssembler(new_id("msg"), prepared.origin)
            text_filter = ToolCallTextFilter(self._registered()) if filtered else None
            received = False
            try:
                async with aclosing(self.engine._gateway.run(prepared)) as stream:
                    async for event in stream:
                        received = True
                        box.assembler.feed(event)
                        if isinstance(event, ReasoningDelta):
                            yield event
                        elif isinstance(event, TextDelta):
                            for visible in self._visible(event, text_filter):
                                yield visible
            except HarnessError as error:
                if received:
                    raise
                delay = self.engine._retry.delay(error, retry_attempt)
                if delay is None:
                    raise
                notice = retry_notice(error, retry_attempt, delay)
                retry_attempt += 1
                self.notices.append(notice)
                yield NoticeRaised(notice)
                await self.engine._sleep(delay)
                continue
            if text_filter is not None:
                trailing = text_filter.flush()
                if trailing:
                    yield TextDelta(block=1, text=trailing)
            message = box.assembler.message()
            finish = box.assembler.finish
            truncation = decide_truncation(
                message, finish.reason if finish is not None else None, self.settings, prepared.profile, truncation_attempt,
            )
            if truncation.action in (TruncationAction.RETRY, TruncationAction.FAIL):
                self.notices.append(truncation.notice)
                yield NoticeRaised(truncation.notice)
                if truncation.action is TruncationAction.FAIL:
                    raise truncation.error
                truncation_attempt += 1
                self.settings = replace(self.settings, max_tokens=truncation.max_tokens)
                continue
            empty_attempt += 1
            if not retry_empty or message.text() or message.tool_calls() or empty_attempt >= EMPTY_RESPONSE_ATTEMPTS:
                break
            logger.warning("[TurnEngine] empty response (attempt %s/%s), retrying", empty_attempt, EMPTY_RESPONSE_ATTEMPTS)
        box.message = message
        self.last_usage = box.assembler.usage
        finish = box.assembler.finish
        self.last_finish = finish.reason if finish is not None else None
        self.last_finish_raw = finish.raw_reason if finish is not None else None

    def _cut_off(self, message: Message) -> Optional[NoticeRaised]:
        decision = decide_truncation(message, self.last_finish, self.settings, self.profile)
        if decision.action is not TruncationAction.INCOMPLETE:
            return None
        self.notices.append(decision.notice)
        return NoticeRaised(decision.notice)

    @staticmethod
    def _visible(event: TextDelta, text_filter: Optional[ToolCallTextFilter]) -> List[TextDelta]:
        if text_filter is None:
            return [event] if event.text else []
        return [
            TextDelta(block=event.block, text=value, phase=event.phase)
            for kind, value in text_filter.feed(event.text)
            if kind == TEXT and value
        ]

    def _registered(self) -> set:
        return self.engine._tools.registered_allowed(self.round.allowed_tools, self.round.deferred_tools)

    def _tool_specs(self) -> Tuple[ToolSpec, ...]:
        schemas = self.engine._tools.turn_tool_schemas(self.round.allowed_tools, self.round.deferred_tools)
        return tuple(ToolSpec.from_schema(schema) for schema in schemas)

    def _decide(self, message: Message) -> Any:
        text = message.text()
        structured = message.tool_calls()
        if structured:
            return _Dispatch(structured, strip_tool_call_xml(text))
        xml_calls = parse_xml_tool_calls(text)
        if xml_calls:
            return _Dispatch(tuple(_call_part(call) for call in xml_calls), strip_tool_call_xml(text))

        truncated = tool_call_rescue.find_truncated_tool_call(text)
        if truncated:
            cleaned = tool_call_rescue.strip_spans(text, [truncated.span])
            if self.rescue_retries < MAX_RESCUE_RETRIES:
                return _Steer(cleaned, tool_call_rescue.truncated_retry_nudge(truncated.tool_name))
            return _Finish(tool_call_rescue.truncated_fallback_message(truncated.tool_name), TRUNCATED, strip_tool_call_xml(cleaned))

        repaired, ambiguous, problems, records, cleaned = self.engine._tools.rescue_final_content(
            text, self.round.allowed_tools, self.round.deferred_tools,
        )
        if repaired:
            self.rescues.extend(records)
            return _Dispatch(tuple(_call_part(call) for call in repaired), strip_tool_call_xml(cleaned or ""))
        if ambiguous:
            if self.rescue_retries < MAX_RESCUE_RETRIES:
                return _Steer(cleaned, tool_call_rescue.retry_nudge(ambiguous, problems))
            return _Finish(tool_call_rescue.fallback_message(ambiguous), AMBIGUOUS, strip_tool_call_xml(cleaned or ""))
        stripped = strip_tool_call_xml(text)
        return _Finish(stripped, ANSWER, stripped)

    @staticmethod
    def _rebuilt(message: Message, text: str, calls: Tuple[ToolCallPart, ...]) -> Message:
        parts: List[Part] = [part for part in message.parts if isinstance(part, ReasoningPart)]
        if text:
            parts.append(TextPart(text=text))
        parts.extend(calls)
        return Message(id=message.id, role=Role.ASSISTANT, parts=tuple(parts), origin=message.origin, raw=message.raw)

    def _commit(self, message: Message, status: StepStatus, *, final: bool = False) -> StepCommitted:
        store = self.engine._store
        if self.first_seq is None:
            self.first_seq = store.next_seq(self.turn.session_id)
        step = StepRecord(
            id=new_id("step"),
            session_id=self.turn.session_id,
            turn_id=self.turn_id,
            seq=self.first_seq + len(self.steps),
            message=message,
            status=status,
            usage=self.last_usage if final and message.role is Role.ASSISTANT else None,
            finish=self.last_finish if final and message.role is Role.ASSISTANT else None,
            finish_raw=self.last_finish_raw if final and message.role is Role.ASSISTANT else None,
        )
        store.append(step)
        self.steps.append(step)
        self.working.append(message)
        return StepCommitted(step)

    def _abort(self, step_status: StepStatus, turn_status: TurnStatus, error: Optional[BaseException]) -> None:
        if self.finished:
            return
        self.finished = True
        try:
            self._persist_abort(step_status, turn_status, error)
        except Exception:
            logger.exception("[TurnEngine] could not persist the %s turn %s", turn_status.value, self.turn_id)

    def _persist_abort(self, step_status: StepStatus, turn_status: TurnStatus, error: Optional[BaseException]) -> None:
        box = self.box
        if box is not None and box.message is None and box.assembler is not None:
            partial = box.assembler.message()
            if partial.parts:
                self.last_usage = box.assembler.usage
                self._commit(partial, step_status, final=True)
        error_json = error.to_json() if isinstance(error, HarnessError) else (
            {"kind": "error", "message": str(error)} if error is not None else None
        )
        state = self._state(turn_status, error=error_json)
        closing = interrupted_tool_step(
            self.steps,
            state,
            step_id=new_id("step"),
            seq=self.first_seq + len(self.steps) if self.first_seq is not None else 0,
            status=step_status,
            reason=CANCELLED_REASON if turn_status is TurnStatus.CANCELLED else FAILED_REASON,
        )
        if closing is not None:
            self.engine._store.append(closing)
            self.steps.append(closing)
        self.engine._store.save_state(state)

    def _state(self, status: TurnStatus, error: Optional[Mapping[str, Any]] = None) -> TurnState:
        return TurnState(
            turn_id=self.turn_id,
            session_id=self.turn.session_id,
            status=status,
            snapshot=TurnSnapshot(
                llm_config_id=self.turn.llm_id,
                system=self.turn.system,
                tools=self.initial_tools,
                settings=self.turn.settings,
                profile=self.profile or CapabilityProfile(adapter="unknown", model=""),
                mode=self.turn.mode,
                deferred_tools={group: tuple(names) for group, names in self.turn.deferred_tools.items()},
            ),
            first_seq=self.first_seq if self.first_seq is not None else self.engine._store.next_seq(self.turn.session_id),
            rounds_used=self.rounds_used,
            max_rounds=self.turn.max_rounds,
            pending=self.pending,
            gathered=self.gathered,
            failed_calls=frozenset(self.round.guard.failed_call_keys()),
            tool_failures=dict(self.round.guard.tool_failures),
            applied_calls=frozenset(self.turn.applied_calls),
            loaded_tool_groups=self._loaded_groups(),
            rescue_retries=self.rescue_retries,
            notices=tuple(self.notices),
            error=error,
        )

    def _loaded_groups(self) -> Tuple[str, ...]:
        allowed = set(self.round.allowed_tools or ())
        return tuple(
            group for group, names in self.round.deferred_tools.items()
            if names and set(names) <= allowed and not set(names) <= set(self.turn.allowed_tools or ())
        )

    def _outcome(self, content: str, reason: str, status: TurnStatus) -> TurnOutcome:
        self.finished = True
        state = self._state(status)
        self.engine._store.save_state(state)
        return TurnOutcome(
            content=content,
            reason=reason,
            state=state,
            executions=tuple(self.round.executions),
            transcript=tuple(self._transcript()),
            rescues=list(self.rescues) or None,
            tool_failures=dict(self.round.guard.tool_failures) or None,
            rounds_used=self.rounds_used,
            usage=self.last_usage,
            finish=self.last_finish,
            finish_raw=self.last_finish_raw,
            steps=tuple(self.steps),
        )

    def _transcript(self) -> List[Dict[str, Any]]:
        rounds: List[Dict[str, Any]] = []
        for step in self.steps:
            message = step.message
            if message.role is Role.ASSISTANT and message.tool_calls():
                round_: Dict[str, Any] = {
                    "calls": [
                        {"id": call.call_id, "name": call.name, "arguments": dict(call.args or {})}
                        for call in message.tool_calls()
                    ],
                    "results": [],
                }
                reasoning = "".join(part.text for part in message.reasoning())
                if reasoning:
                    round_["reasoning"] = reasoning
                rounds.append(round_)
            elif message.role is Role.TOOL and rounds:
                rounds[-1]["results"].extend(_transcript_result(part) for part in message.tool_results())
        if self.gathered and rounds:
            rounds[-1]["results"].extend(_transcript_result(part) for part in self.gathered)
        return rounds


def _transcript_result(part: ToolResultPart) -> Dict[str, Any]:
    return {"tool_call_id": part.call_id, "content": part.text()}


def _call_part(call: Dict[str, Any]) -> ToolCallPart:
    function = call.get("function", {})
    return ToolCallPart(
        call_id=call.get("id") or new_id("call"),
        name=function.get("name", ""),
        args=_object_args(function.get("arguments")),
    )


def _object_args(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        return parse_args(arguments) or {}
    return {}

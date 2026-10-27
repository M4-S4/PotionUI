import json
from contextlib import aclosing
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

from src.features.llm import trace_collector
from src.features.llm.harness.adapter import RenderRequest, ToolSpec, WireRequest
from src.features.llm.harness.baseline import InMemoryTurnStore
from src.features.llm.harness.retry import BackoffRetry
from src.features.llm.harness.tool_runner import ParallelToolRunner
from src.features.llm.harness.context import user_text
from src.features.llm.harness.engine import BudgetExhausted, TurnEngine, TurnOutcome
from src.features.llm.harness.errors import HarnessError, InvalidRequest, WireFailure
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import ImagePart, Message, Role, ToolCallPart, ToolResultPart
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.harness.turn_state import ToolFinished, ToolStarted, TurnSnapshot, TurnState, TurnStatus
from src.features.llm.tools.base import ToolApprovalPreview, ToolExecution, ToolResult, ToolSource
from src.features.llm.tools.executor import ToolExecutor
from tests.fixtures.llm_engine import scripted_config, scripted_gateway

DEFAULT_TURN: Dict[str, Any] = {"content": "All done."}
SIDE_PURPOSES = frozenset({"title", "memory_reflection", "memory_compaction"})
SIDE_TURN: Dict[str, Any] = {"content": "[]"}
THINKING_PIECE_CHARS = 2000


def _purpose() -> Optional[str]:
    context = trace_collector._trace_context.get()
    return context.purpose if context is not None else None


def turn_events(turn: Dict[str, Any]) -> List[StreamEvent]:
    events: List[StreamEvent] = []
    thinking = turn.get("thinking") or ""
    for start in range(0, len(thinking), THINKING_PIECE_CHARS):
        events.append(ReasoningDelta(block=0, text=thinking[start:start + THINKING_PIECE_CHARS]))
    for piece in turn.get("chunks") or ([turn["content"]] if turn.get("content") else []):
        events.append(TextDelta(block=1, text=piece))
    for offset, call in enumerate(turn.get("tool_calls") or []):
        block = 2 + offset
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        events.append(ToolCallStart(block=block, call_id=call.get("id") or f"call_{offset}", name=function.get("name", ""), wire_id=call.get("id")))
        events.append(ToolArgsDelta(block=block, text=arguments if isinstance(arguments, str) else json.dumps(arguments)))
        events.append(ToolCallEnd(block=block))
    usage = turn.get("usage") or {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
    events.append(UsageReported(Usage(**usage)))
    finish = turn.get("finish") or (FinishReason.TOOL_CALLS if turn.get("tool_calls") else FinishReason.STOP)
    events.append(Finish(reason=finish, raw_reason=turn.get("finish_raw", finish.value)))
    return events


class _Replay:
    def __init__(self, events: Sequence[StreamEvent]) -> None:
        self._events = list(events)

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        return []

    def close(self) -> List[StreamEvent]:
        return self._events


class ScriptedTurns:
    name = "scripted"

    def __init__(self, turns: Sequence[Any] = (), default: Dict[str, Any] = DEFAULT_TURN) -> None:
        self.turns = list(turns)
        self.default = default
        self.requests: List[RenderRequest] = []
        self.side_requests: List[RenderRequest] = []
        self._current: Any = None

    def render(self, request: RenderRequest, profile: Any) -> WireRequest:
        if _purpose() in SIDE_PURPOSES:
            self.side_requests.append(request)
            self._current = SIDE_TURN
        else:
            self.requests.append(request)
            self._current = self.turns.pop(0) if self.turns else self.default
        return WireRequest(method="POST", url="scripted://turn", headers={}, body={"messages": legacy_dicts(request.messages)})

    def decoder(self, request: RenderRequest, profile: Any) -> _Replay:
        if isinstance(self._current, HarnessError):
            return _Replay([])
        if isinstance(self._current, list):
            return _Replay(self._current)
        return _Replay(turn_events(self._current))

    async def send(self, wire: WireRequest, request: RenderRequest) -> AsyncIterator[bytes]:
        if isinstance(self._current, HarnessError):
            raise self._current
        yield b""

    def classify_error(self, failure: WireFailure) -> HarnessError:
        return InvalidRequest(f"scripted failure {failure.status}", status=failure.status)

    async def discover(self, config: Any) -> ProfileLayer:
        return ProfileLayer(ProfileSource.DISCOVERED, {})

    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> list:
        return []

    @property
    def calls(self) -> List[Dict[str, Any]]:
        return [
            {
                "messages": legacy_dicts(request.messages),
                "tools": [tool.function_schema() for tool in request.tools] or None,
                "system": request.system,
                "settings": request.settings,
            }
            for request in self.requests
        ]


def legacy_dicts(messages: Sequence[Message]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for message in messages:
        if message.role is Role.TOOL:
            entries.extend(
                {"role": "tool", "content": part.text(), "tool_call_id": part.call_id, "name": part.name}
                for part in message.tool_results()
            )
            continue
        if message.role is Role.USER:
            entries.append({"role": "user", "content": user_text(message)})
            continue
        entry: Dict[str, Any] = {"role": message.role.value, "content": message.text()}
        reasoning = "".join(part.text for part in message.reasoning())
        if reasoning:
            entry["reasoning"] = reasoning
        calls = message.tool_calls()
        if calls:
            entry["tool_calls"] = [
                {"id": call.call_id, "type": "function", "function": {"name": call.name, "arguments": dict(call.args or {})}}
                for call in calls
            ]
        entries.append(entry)
    return entries


def scripted_llm(turns: Sequence[Any] = (), **config: Any) -> Any:
    adapter = ScriptedTurns(turns)
    values = {"id": "llm-1", "supports_vision": True}
    values.update(config)
    gateway = scripted_gateway(adapter, scripted_config(**values))
    gateway.adapter = adapter
    return gateway


def turn_engine_for(gateway: Any, tool_executor: ToolExecutor, store: Optional[InMemoryTurnStore] = None) -> TurnEngine:
    return TurnEngine(
        gateway=gateway,
        tools=tool_executor,
        runner=ParallelToolRunner(tool_executor),
        store=store or InMemoryTurnStore(),
        retry=BackoffRetry(max_retries=0),
        clock=lambda: 0.0,
    )


def set_config(gateway: Any, **updates: Any) -> None:
    configs = gateway.repository._configs
    configs["llm-1"] = configs["llm-1"].model_copy(update=updates)


def _legacy_usage(data: Dict[str, Any]) -> Optional[Usage]:
    keys = ("tokens_used", "prompt_tokens", "completion_tokens", "thinking_mode")
    if not any(data.get(key) is not None for key in keys):
        return None
    raw = {"thinking_mode": data["thinking_mode"]} if data.get("thinking_mode") is not None else None
    return Usage(
        input_tokens=data.get("prompt_tokens"),
        output_tokens=data.get("completion_tokens"),
        total_tokens=data.get("tokens_used"),
        raw=raw,
    )


def _legacy_finish(completion: Any) -> Tuple[Optional[FinishReason], Optional[str]]:
    if not isinstance(completion, dict) or not completion.get("reason"):
        return None, None
    return FinishReason(completion["reason"]), completion.get("raw")


class ScriptedTurnEngine:
    def __init__(self) -> None:
        self.inputs: List[Any] = []
        self._script: Any = None

    def script(self, source: Any) -> None:
        self._script = source

    def previous_tools(self, session_id: str) -> Tuple[ToolSpec, ...]:
        return ()

    @property
    def call_kwargs(self) -> Dict[str, Any]:
        turn = self.inputs[-1]
        messages = [*turn.history, *turn.new_messages]
        images = [part.data for message in messages for part in message.parts if isinstance(part, ImagePart)]
        settings = turn.settings
        options = {
            key: value
            for key, value in (
                ("temperature", settings.temperature), ("top_p", settings.top_p), ("top_k", settings.top_k),
                ("min_p", settings.min_p), ("max_tokens", settings.max_tokens), ("think", settings.reasoning),
            )
            if value is not None
        } or None
        return {
            "messages": legacy_dicts(messages),
            "llm_id": turn.llm_id,
            "image_data": images[-1] if images else None,
            "custom_system_message": turn.system,
            "system_message": turn.system or "",
            "options_override": options,
            "llm_options": options,
            "tool_context": turn.tool_context,
            "max_iterations": turn.max_rounds,
            "allowed_tools": list(turn.allowed_tools) if turn.allowed_tools is not None else None,
            "iteration_nudge": turn.iteration_nudge,
            "applied_calls": set(turn.applied_calls) or None,
            "deferred_tools": {group: list(names) for group, names in turn.deferred_tools.items()} or None,
        }

    async def run(self, turn: Any) -> AsyncIterator[Any]:
        self.inputs.append(turn)
        source = self._script if self._script is not None else []
        if isinstance(source, BaseException):
            raise source
        if callable(source):
            source = source(**self.call_kwargs)
        if not hasattr(source, "__aiter__"):
            source = _tokens(source)
        done: Optional[Dict[str, Any]] = None
        usage_data: Dict[str, Any] = {}
        async with aclosing(source) as stream:
            async for event in stream:
                kind = event.get("type")
                data = event.get("data") if isinstance(event.get("data"), dict) else event
                if kind == "token":
                    yield TextDelta(block=1, text=data.get("content", ""))
                elif kind == "thinking":
                    yield ReasoningDelta(block=0, text=data.get("content", ""))
                elif kind == "tool_start":
                    yield ToolStarted(ToolCallPart(call_id="call_scripted", name=data.get("tool_name", ""), args=dict(data.get("arguments") or {})))
                elif kind == "tool_end":
                    record = _record_from_payload(data)
                    yield ToolFinished(ToolResultPart(call_id=record.tool_call_id or "", name=record.tool_name, content=()), record=record)
                elif kind == "status" and data.get("step") == "tool_budget_exhausted":
                    yield BudgetExhausted((data.get("detail") or {}).get("max_iterations", 0))
                elif kind == "usage":
                    usage_data = data
                elif kind == "done":
                    done = data
        yield _outcome(turn, done, usage_data)


async def _tokens(chunks: Sequence[str]) -> AsyncIterator[Dict[str, Any]]:
    for chunk in chunks:
        yield {"type": "token", "content": chunk}


def _outcome(turn: Any, done: Optional[Dict[str, Any]], usage_data: Dict[str, Any]) -> TurnOutcome:
    summary = done if done is not None else usage_data
    finish, finish_raw = _legacy_finish(summary.get("completion"))
    pending = bool((done or {}).get("pending_tool_approval"))
    state = TurnState(
        turn_id="turn-scripted",
        session_id=turn.session_id,
        status=TurnStatus.AWAITING_DECISION if pending else TurnStatus.COMPLETED,
        snapshot=TurnSnapshot(
            llm_config_id=turn.llm_id, system=turn.system or "", tools=(), settings=turn.settings,
            profile=CapabilityProfile(adapter="scripted", model="scripted-model"),
        ),
        first_seq=0,
    )
    return TurnOutcome(
        content=(done or {}).get("full_content", ""),
        reason="pending" if pending else "answer",
        state=state,
        executions=tuple((done or {}).get("tool_executions") or ()),
        transcript=tuple((done or {}).get("tool_transcript") or ()),
        rescues=(done or {}).get("rescues"),
        tool_failures=(done or {}).get("tool_failures"),
        rounds_used=(done or {}).get("rounds_used") or 0,
        usage=_legacy_usage(summary),
        last_usage=_legacy_usage(summary),
        finish=finish,
        finish_raw=finish_raw,
        steps=(),
    )


def _record_from_payload(data: Dict[str, Any]) -> ToolExecution:
    preview = data.get("preview")
    return ToolExecution(
        tool_name=data.get("tool_name", ""),
        arguments=data.get("arguments") or {},
        result=ToolResult(
            success=bool(data.get("success", True)),
            data="",
            sources=[ToolSource(**source) for source in data["sources"]] if data.get("sources") else None,
            preview=ToolApprovalPreview(**preview) if isinstance(preview, dict) else None,
        ),
        duration_ms=data.get("duration_ms") or 0,
        approval_status=data.get("approval_status"),
        tool_call_id=data.get("tool_call_id"),
    )


def _plain(value: Any) -> Any:
    return value if isinstance(value, (dict, int, str, list, type(None))) else None


def buffered_reply(response: Any, executions: Sequence[Any] = ()) -> Any:
    def build(**_: Any) -> AsyncIterator[Dict[str, Any]]:
        async def events() -> AsyncIterator[Dict[str, Any]]:
            content = _plain(getattr(response, "content", "")) or ""
            done = {
                "tool_executions": list(executions),
                "full_content": content,
                "tokens_used": _plain(getattr(response, "tokens_used", None)),
                "prompt_tokens": _plain(getattr(response, "prompt_tokens", None)),
                "completion_tokens": _plain(getattr(response, "completion_tokens", None)),
                "thinking_mode": _plain(getattr(response, "thinking_mode", None)),
                "completion": _plain(getattr(response, "completion", None)),
                "rescues": _plain(getattr(response, "rescues", None)),
                "tool_failures": _plain(getattr(response, "tool_failures", None)),
                "tool_transcript": _plain(getattr(response, "tool_transcript", None)),
                "rounds_used": _plain(getattr(response, "rounds_used", None)),
            }
            if content:
                yield {"type": "token", "content": content}
            yield {"type": "done", "data": done}

        return events()

    return build


def typed_history(inject: Any) -> Any:
    def call(history: List[Dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
        from src.features.llm.harness.legacy import messages_from_dicts

        messages = messages_from_dicts(history)[0]
        result = inject(messages, *args, **kwargs)
        if hasattr(result, "__await__"):
            async def finish() -> Any:
                value = await result
                history[:] = legacy_dicts(messages)
                return value

            return finish()
        history[:] = legacy_dicts(messages)
        return result

    return call


def legacy_turn(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    turn: Dict[str, Any] = {"chunks": [], "tool_calls": []}
    for event in events:
        if event.get("type") == "token":
            turn["chunks"].append(event.get("content", ""))
        elif event.get("type") == "thinking":
            turn["thinking"] = turn.get("thinking", "") + event.get("content", "")
        elif event.get("type") == "tool_calls":
            turn["tool_calls"].extend(event.get("tool_calls") or [])
        elif event.get("type") == "usage" and event.get("tokens_used") is not None:
            turn["usage"] = {
                "input_tokens": event.get("prompt_tokens"),
                "output_tokens": event.get("completion_tokens"),
                "total_tokens": event.get("tokens_used"),
            }
    if not turn["chunks"] and not turn["tool_calls"] and "thinking" not in turn:
        return {"finish": FinishReason.STOP}
    return turn


def scripted_llm_from_events(scripts: Sequence[Sequence[Dict[str, Any]]], **config: Any) -> Any:
    turns = [legacy_turn(events) for events in scripts]
    gateway = scripted_llm(turns, **config)
    gateway.adapter.default = turns[-1]
    return gateway

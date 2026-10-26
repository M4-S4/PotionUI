import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.features.llm.harness.adapter import RenderRequest, StreamDecoder, ToolChoice, ToolSpec, WireRequest
from src.features.llm.harness.context import user_text
from src.features.llm.harness.errors import (
    AuthFailed,
    ConnectionFailed,
    ContextOverflow,
    HarnessError,
    InvalidRequest,
    MalformedStream,
    OpaqueViolation,
    Overloaded,
    QuotaExhausted,
    RateLimited,
    ServerFailed,
    WireFailure,
)
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    OpaqueDelta,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import (
    CompactionPart,
    FilePart,
    ImagePart,
    JsonPart,
    Message,
    Origin,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    credential_fingerprint,
)
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource, ReasoningKind
from src.features.llm.harness.turn_state import Notice, NoticeKind
from src.features.llm.model_listing import (
    CONTEXT_WINDOW_PROBE_TIMEOUT_SECONDS,
    DiscoveredModel,
    ModelListingError,
    fetch_json,
    positive_int,
    sorted_models,
)

ENCRYPTED_REASONING = "reasoning.encrypted_content"
WINDOW_KEYS = ("context_window", "context_length", "max_model_len")
SKELETON_FIELDS = {
    "reasoning": ("id", "encrypted_content"),
    "text": ("text", "phase"),
    "call": ("call_id", "name", "arguments"),
}
SCHEMA_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _data_url(media_type: str, data: Optional[str], what: str) -> str:
    if data is None:
        raise InvalidRequest(f"{what} has to be loaded before it can be sent")
    return f"data:{media_type};base64,{data}"


def _output_text(content: Any) -> str:
    return "".join(
        entry.get("text") or entry.get("refusal") or ""
        for entry in (content if isinstance(content, list) else [])
        if isinstance(entry, dict)
    )


def _user_item(message: Message) -> Optional[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, CompactionPart):
            content.append({"type": "input_text", "text": part.summary})
    text = user_text(message)
    if text:
        content.append({"type": "input_text", "text": text})
    for part in message.parts:
        if isinstance(part, ImagePart):
            content.append({"type": "input_image", "image_url": _data_url(part.media_type, part.data, "an image")})
        elif isinstance(part, FilePart):
            content.append({
                "type": "input_file",
                "filename": part.name or "file",
                "file_data": _data_url(part.media_type, part.data, "a file"),
            })
    return {"role": "user", "content": content} if content else None


def _result_item(part: ToolResultPart, profile: CapabilityProfile) -> Dict[str, Any]:
    texts: List[str] = []
    images: List[Dict[str, Any]] = []
    for entry in part.content:
        if isinstance(entry, TextPart):
            texts.append(entry.text)
        elif isinstance(entry, JsonPart):
            texts.append(_compact(entry.value))
        elif profile.tool_result_media and entry.data is not None:
            images.append({"type": "input_image", "image_url": _data_url(entry.media_type, entry.data, "an image")})
    text = "".join(texts)
    output: Any = [{"type": "input_text", "text": text}, *images] if images else text
    return {"type": "function_call_output", "call_id": part.call_id, "output": output}


def _call_arguments(part: ToolCallPart) -> str:
    if part.args_raw is not None:
        return part.args_raw
    return _compact(part.args) if part.args is not None else "{}"


def _raw_skeleton(raw: Sequence[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    skeleton: List[Tuple[Any, ...]] = []
    for item in raw:
        kind = item.get("type")
        if kind == "reasoning":
            skeleton.append(("reasoning", item.get("id"), item.get("encrypted_content")))
        elif kind == "message":
            text = _output_text(item.get("content"))
            if text:
                skeleton.append(("text", text, item.get("phase")))
        elif kind == "function_call":
            skeleton.append(("call", item.get("call_id"), item.get("name"), item.get("arguments") or ""))
    return skeleton


def _parts_skeleton(message: Message) -> List[Tuple[Any, ...]]:
    skeleton: List[Tuple[Any, ...]] = []
    for part in message.parts:
        if isinstance(part, ReasoningPart):
            opaque = part.opaque or {}
            skeleton.append(("reasoning", opaque.get("id"), opaque.get("encrypted_content")))
        elif isinstance(part, TextPart):
            skeleton.append(("text", part.text, part.phase))
        elif isinstance(part, ToolCallPart):
            skeleton.append(("call", part.call_id, part.name, part.args_raw or ""))
    return skeleton


def _difference(expected: Sequence[Tuple[Any, ...]], actual: Sequence[Tuple[Any, ...]]) -> str:
    for position, (want, got) in enumerate(zip(expected, actual)):
        if want == got:
            continue
        if want[0] != got[0]:
            return f"item {position} is a {want[0]} in the output and a {got[0]} in the parts"
        names = SKELETON_FIELDS[want[0]]
        differing = [name for name, left, right in zip(names, want[1:], got[1:]) if left != right]
        return f"{want[0]} {position} differs in {', '.join(differing)}"
    return f"{len(expected)} output items against {len(actual)} parts"


def _violation(message: Message, reason: str) -> OpaqueViolation:
    return OpaqueViolation(
        f"Stored OpenAI output for message {message.id} cannot be replayed: {reason}",
        detail={"message_id": message.id, "adapter": "openai-responses"},
    )


def _verified_output(message: Message) -> List[Dict[str, Any]]:
    raw = message.raw
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise _violation(message, "the stored payload is not an output item list")
    for item in raw:
        if item.get("type") == "reasoning" and not (item.get("id") and item.get("encrypted_content")):
            raise _violation(message, "a reasoning item lost its id or encrypted_content")
    expected = _raw_skeleton(raw)
    actual = _parts_skeleton(message)
    if expected != actual:
        raise _violation(message, _difference(expected, actual))
    return list(raw)


def _rebuilt_items(message: Message) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            item: Dict[str, Any] = {"role": "assistant", "content": [{"type": "output_text", "text": part.text}]}
            if part.phase:
                item["phase"] = part.phase
            items.append(item)
        elif isinstance(part, CompactionPart):
            items.append({"role": "assistant", "content": [{"type": "output_text", "text": part.summary}]})
        elif isinstance(part, ToolCallPart):
            items.append({
                "type": "function_call",
                "call_id": part.call_id,
                "name": part.name,
                "arguments": _call_arguments(part),
            })
    return items


def _dropped_notice(message: Message, reason: str, text: str) -> Notice:
    return Notice(NoticeKind.REASONING_DROPPED, text, detail={"message_id": message.id, "reason": reason})


def _tool_definition(spec: ToolSpec, profile: CapabilityProfile) -> Dict[str, Any]:
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.parameters,
        "strict": bool(spec.strict and profile.strict_tools),
    }


def _tool_choice(choice: Optional[ToolChoice], profile: CapabilityProfile, reasoning_active: bool) -> Any:
    if choice is None or choice.mode == "auto":
        return None
    allowed = profile.tool_choice_with_reasoning if reasoning_active else profile.tool_choice
    if choice.mode not in allowed:
        return None
    if choice.mode == "named":
        if not choice.name:
            raise InvalidRequest("A named tool choice needs the tool name")
        return {"type": "function", "name": choice.name}
    return choice.mode


def _effort(setting: Any, profile: CapabilityProfile) -> Optional[str]:
    if setting is False:
        return "none" if "none" in profile.reasoning_levels else None
    if isinstance(setting, str) and (not profile.reasoning_levels or setting in profile.reasoning_levels):
        return setting
    return None


def _reasoning(request: RenderRequest, profile: CapabilityProfile) -> Optional[Dict[str, Any]]:
    if profile.reasoning is ReasoningKind.NONE:
        return None
    reasoning: Dict[str, Any] = {}
    effort = _effort(request.settings.reasoning, profile)
    if effort:
        reasoning["effort"] = effort
    if request.config.show_thinking and effort != "none":
        reasoning["summary"] = "auto"
    return reasoning or None


def _text_format(schema: Any) -> Dict[str, Any]:
    title = schema.get("title") if isinstance(schema, dict) else None
    name = SCHEMA_NAME_UNSAFE.sub("_", title)[:64] if isinstance(title, str) and title else "response"
    return {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}


@dataclass
class _Slot:
    opened: bool = False
    done: bool = False
    text: bool = False
    summary: bool = False
    reasoning_text: bool = False
    args: bool = False
    phase: Optional[str] = None
    phase_sent: bool = False
    summary_index: int = 0


def _index(event: Dict[str, Any]) -> int:
    index = event.get("output_index")
    if not isinstance(index, int) or isinstance(index, bool):
        raise MalformedStream(f"{event.get('type')} carried no output index")
    return index


def _item(event: Dict[str, Any]) -> Dict[str, Any]:
    item = event.get("item")
    if not isinstance(item, dict):
        raise MalformedStream(f"{event.get('type')} carried no item")
    return item


def _response(event: Dict[str, Any]) -> Dict[str, Any]:
    response = event.get("response")
    return response if isinstance(response, dict) else {}


def _usage(data: Any) -> Optional[Usage]:
    if not isinstance(data, dict):
        return None
    inputs = data.get("input_tokens_details")
    outputs = data.get("output_tokens_details")
    inputs = inputs if isinstance(inputs, dict) else {}
    outputs = outputs if isinstance(outputs, dict) else {}
    return Usage(
        input_tokens=data.get("input_tokens"),
        output_tokens=data.get("output_tokens"),
        total_tokens=data.get("total_tokens"),
        reasoning_tokens=outputs.get("reasoning_tokens"),
        cached_input_tokens=inputs.get("cached_tokens"),
        cache_write_tokens=inputs.get("cache_write_tokens"),
        raw=data,
    )


class _Decoder:
    def __init__(self, classify: Callable[[WireFailure], HarnessError], buffered: bool) -> None:
        self._classify = classify
        self._buffered = buffered
        self._pending = b""
        self._slots: Dict[int, _Slot] = {}
        self._items: Dict[int, Dict[str, Any]] = {}
        self._refused = False
        self._finished = False
        self._handlers: Dict[str, Callable[[Dict[str, Any]], List[StreamEvent]]] = {
            "response.output_item.added": self._on_added,
            "response.output_item.done": self._on_done,
            "response.output_text.delta": self._on_text,
            "response.refusal.delta": self._on_refusal,
            "response.reasoning_summary_text.delta": self._on_summary,
            "response.reasoning_text.delta": self._on_reasoning_text,
            "response.function_call_arguments.delta": self._on_arguments,
            "response.completed": lambda event: self._finish(_response(event), "completed"),
            "response.incomplete": lambda event: self._finish(_response(event), "incomplete"),
            "response.failed": lambda event: self._finish(_response(event), "failed"),
            "error": self._on_error,
        }

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        if self._buffered:
            self._pending += chunk
            return []
        self._pending = (self._pending + chunk).replace(b"\r\n", b"\n")
        events: List[StreamEvent] = []
        while True:
            end = self._pending.find(b"\n\n")
            if end < 0:
                return events
            frame, self._pending = self._pending[:end], self._pending[end + 2:]
            events.extend(self._frame(frame))

    def close(self) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        if self._buffered:
            events = self._body(self._pending)
        elif self._pending.strip():
            events = self._frame(self._pending)
        self._pending = b""
        if not self._finished:
            raise MalformedStream("The stream ended before the response completed")
        return events

    def _body(self, body: bytes) -> List[StreamEvent]:
        try:
            response = json.loads(body)
        except ValueError:
            raise MalformedStream("The response body is not valid JSON") from None
        if not isinstance(response, dict):
            raise MalformedStream("The response body is not an object")
        status = response.get("status") or "completed"
        if status not in ("completed", "incomplete", "failed"):
            raise MalformedStream(f"The response ended with status {status}")
        return self._finish(response, status)

    def _frame(self, frame: bytes) -> List[StreamEvent]:
        try:
            lines = frame.decode("utf-8").split("\n")
        except UnicodeDecodeError:
            raise MalformedStream("An event is not valid UTF-8") from None
        data = [line[6:] if line.startswith("data: ") else line[5:] for line in lines if line.startswith("data:")]
        text = "\n".join(data)
        if not text or text == "[DONE]" or self._finished:
            return []
        try:
            payload = json.loads(text)
        except ValueError:
            raise MalformedStream("An event carried invalid JSON") from None
        if not isinstance(payload, dict):
            raise MalformedStream("An event carried a payload that is not an object")
        handler = self._handlers.get(payload.get("type"))
        return handler(payload) if handler else []

    def _slot(self, index: int) -> _Slot:
        return self._slots.setdefault(index, _Slot())

    def _on_added(self, event: Dict[str, Any]) -> List[StreamEvent]:
        return self._open(_index(event), _item(event))

    def _on_done(self, event: Dict[str, Any]) -> List[StreamEvent]:
        return self._close(_index(event), _item(event))

    def _open(self, index: int, item: Dict[str, Any]) -> List[StreamEvent]:
        slot = self._slot(index)
        if slot.opened:
            return []
        slot.opened = True
        kind = item.get("type")
        if kind == "reasoning":
            return [ReasoningDelta(block=index, text="")]
        if kind == "message":
            slot.phase = item.get("phase")
        elif kind == "function_call":
            return [ToolCallStart(
                block=index,
                call_id=str(item.get("call_id") or ""),
                name=str(item.get("name") or ""),
                wire_id=item.get("id"),
            )]
        return []

    def _close(self, index: int, item: Dict[str, Any]) -> List[StreamEvent]:
        events = self._open(index, item)
        slot = self._slot(index)
        if slot.done:
            return events
        slot.done = True
        self._items[index] = item
        kind = item.get("type")
        if kind == "reasoning":
            events.extend(self._close_reasoning(index, slot, item))
        elif kind == "message":
            events.extend(self._close_message(index, slot, item))
        elif kind == "function_call":
            arguments = item.get("arguments")
            if arguments and not slot.args:
                events.append(ToolArgsDelta(block=index, text=arguments))
            if item.get("status") != "incomplete":
                events.append(ToolCallEnd(block=index))
        return events

    @staticmethod
    def _close_reasoning(index: int, slot: _Slot, item: Dict[str, Any]) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        if not slot.summary:
            summaries = item.get("summary") if isinstance(item.get("summary"), list) else []
            summary = "\n\n".join(
                entry["text"] for entry in summaries if isinstance(entry, dict) and entry.get("text")
            )
            if summary:
                events.append(ReasoningDelta(block=index, text=summary, summary=True))
        if not slot.reasoning_text:
            text = _output_text(item.get("content"))
            if text:
                events.append(ReasoningDelta(block=index, text=text))
        for key in ("id", "encrypted_content"):
            if item.get(key):
                events.append(OpaqueDelta(block=index, key=key, value=item[key]))
        return events

    def _close_message(self, index: int, slot: _Slot, item: Dict[str, Any]) -> List[StreamEvent]:
        content = item.get("content") if isinstance(item.get("content"), list) else []
        if any(isinstance(entry, dict) and entry.get("type") == "refusal" for entry in content):
            self._refused = True
        text = _output_text(content)
        phase = item.get("phase") or slot.phase
        if text and not slot.text:
            slot.phase_sent = slot.phase_sent or bool(phase)
            return [TextDelta(block=index, text=text, phase=phase)]
        if phase and not slot.phase_sent:
            return [TextDelta(block=index, text="", phase=phase)]
        return []

    def _on_text(self, event: Dict[str, Any]) -> List[StreamEvent]:
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return []
        index = _index(event)
        slot = self._slot(index)
        slot.text = True
        slot.phase_sent = slot.phase_sent or bool(slot.phase)
        return [TextDelta(block=index, text=delta, phase=slot.phase)]

    def _on_refusal(self, event: Dict[str, Any]) -> List[StreamEvent]:
        self._refused = True
        return self._on_text(event)

    def _on_summary(self, event: Dict[str, Any]) -> List[StreamEvent]:
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return []
        index = _index(event)
        slot = self._slot(index)
        events: List[StreamEvent] = []
        position = event.get("summary_index") or 0
        if slot.summary and position != slot.summary_index:
            events.append(ReasoningDelta(block=index, text="\n\n", summary=True))
        slot.summary = True
        slot.summary_index = position
        events.append(ReasoningDelta(block=index, text=delta, summary=True))
        return events

    def _on_reasoning_text(self, event: Dict[str, Any]) -> List[StreamEvent]:
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return []
        index = _index(event)
        self._slot(index).reasoning_text = True
        return [ReasoningDelta(block=index, text=delta)]

    def _on_arguments(self, event: Dict[str, Any]) -> List[StreamEvent]:
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            return []
        index = _index(event)
        self._slot(index).args = True
        return [ToolArgsDelta(block=index, text=delta)]

    def _on_error(self, event: Dict[str, Any]) -> List[StreamEvent]:
        nested = event.get("error")
        error = nested if isinstance(nested, dict) else {key: event.get(key) for key in ("code", "message", "param")}
        raise self._failure(error)

    def _failure(self, error: Any) -> HarnessError:
        body = json.dumps({"error": error if isinstance(error, dict) else {}}).encode("utf-8")
        return self._classify(WireFailure(status=None, body=body, in_stream=True))

    def _finish(self, response: Dict[str, Any], status: str) -> List[StreamEvent]:
        if status == "failed":
            raise self._failure(response.get("error"))
        events: List[StreamEvent] = []
        output = response.get("output") if isinstance(response.get("output"), list) else []
        for index, item in enumerate(output):
            if isinstance(item, dict):
                events.extend(self._close(index, item))
        usage = _usage(response.get("usage"))
        if usage is not None:
            events.append(UsageReported(usage))
        events.append(self._finish_event(response, status))
        self._finished = True
        return events

    def _finish_event(self, response: Dict[str, Any], status: str) -> Finish:
        complete = all(slot.done for slot in self._slots.values())
        raw = [self._items[index] for index in sorted(self._items)] if complete else None
        if status == "incomplete":
            details = response.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, dict) else None
            mapped = {"max_output_tokens": FinishReason.LENGTH, "content_filter": FinishReason.CONTENT_FILTER}
            return Finish(reason=mapped.get(reason, FinishReason.UNKNOWN), raw_reason=reason or status, raw=raw)
        if any(item.get("type") == "function_call" for item in self._items.values()):
            return Finish(reason=FinishReason.TOOL_CALLS, raw_reason=status, raw=raw)
        return Finish(reason=FinishReason.REFUSAL if self._refused else FinishReason.STOP, raw_reason=status, raw=raw)


def _error_object(body: bytes) -> Dict[str, Any]:
    try:
        data = json.loads(body)
    except ValueError:
        data = None
    if isinstance(data, dict):
        nested = data.get("error")
        if isinstance(nested, dict):
            return nested
        if isinstance(nested, str):
            return {"message": nested}
        return data
    return {"message": body.decode("utf-8", errors="replace")[:500]}


class OpenAIResponsesAdapter:
    name = "openai-responses"

    def origin(self, config: Any) -> Origin:
        return Origin(
            adapter=self.name,
            provider=config.type,
            model=config.model,
            credential=credential_fingerprint(config.base_url, config.api_key),
        )

    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest:
        config = request.config
        settings = request.settings
        items, _ = self._input(request, profile)
        reasoning = _reasoning(request, profile)
        reasoning_active = reasoning is not None and reasoning.get("effort") != "none"
        body: Dict[str, Any] = {"model": config.model}
        if request.system:
            body["instructions"] = request.system
        body["input"] = items
        if request.tools:
            body["tools"] = [_tool_definition(spec, profile) for spec in request.tools]
            choice = _tool_choice(settings.tool_choice, profile, reasoning_active)
            if choice is not None:
                body["tool_choice"] = choice
            if not profile.parallel_tools:
                body["parallel_tool_calls"] = False
        body["store"] = False
        body["stream"] = request.stream
        if profile.reasoning is not ReasoningKind.NONE:
            body["include"] = [ENCRYPTED_REASONING]
        if reasoning is not None:
            body["reasoning"] = reasoning
        if settings.max_tokens is not None:
            ceiling = profile.max_output_tokens
            body["max_output_tokens"] = min(settings.max_tokens, ceiling) if ceiling else settings.max_tokens
        for name, value in (("temperature", settings.temperature), ("top_p", settings.top_p)):
            if value is None or name in profile.sampling_rejected:
                continue
            if reasoning_active and name in profile.sampling_ignored_with_reasoning:
                continue
            body[name] = value
        if settings.cache_key:
            body["prompt_cache_key"] = settings.cache_key
        if settings.response_schema is not None:
            body["text"] = _text_format(settings.response_schema)
        headers = {"Content-Type": "application/json"}
        if request.stream:
            headers["Accept"] = "text/event-stream"
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        url = f"{config.base_url.strip().rstrip('/')}/responses"
        return WireRequest(method="POST", url=url, headers=headers, body=body, stream=request.stream)

    def reasoning_notices(self, request: RenderRequest, profile: CapabilityProfile) -> Tuple[Notice, ...]:
        return tuple(self._input(request, profile)[1])

    def decoder(self, request: RenderRequest, profile: CapabilityProfile) -> StreamDecoder:
        return _Decoder(self.classify_error, buffered=not request.stream)

    def classify_error(self, failure: WireFailure) -> HarnessError:
        error = _error_object(failure.body)
        code = str(error.get("code") or "")
        kind = str(error.get("type") or "")
        status = failure.status
        message = str(error.get("message") or "") or f"OpenAI answered HTTP {status}"
        lowered = message.lower()
        detail = {key: error[key] for key in ("code", "type", "param") if error.get(key)}
        request_id = failure.header("x-request-id")
        if request_id:
            detail["request_id"] = request_id
        if failure.in_stream:
            detail["in_stream"] = True
        fields: Dict[str, Any] = {
            "status": status,
            "code": code or kind or None,
            "retry_after": failure.retry_after_seconds(),
            "detail": detail,
        }
        if status is None and failure.cause is not None and not failure.body:
            return ConnectionFailed(str(failure.cause) or "The connection to OpenAI failed", **fields)
        if code == "invalid_encrypted_content" or (
            "encrypted content" in lowered and "not supported" not in lowered
        ) or "without its required" in lowered:
            return OpaqueViolation(message, **fields)
        if code == "context_length_exceeded" or "context length" in lowered or "context window" in lowered:
            return ContextOverflow(message, **fields)
        if "insufficient_quota" in (code, kind) or "exceeded your current quota" in lowered:
            return QuotaExhausted(message, **fields)
        if status == 429 or code == "rate_limit_exceeded":
            return RateLimited(message, **fields)
        if status in (401, 403) or code in ("invalid_api_key", "invalid_organization", "account_deactivated"):
            return AuthFailed(message, **fields)
        if status == 503 or code in ("server_is_overloaded", "overloaded", "slow_down"):
            return Overloaded(message, **fields)
        if (status is not None and status >= 500) or code in ("server_error", "internal_error"):
            return ServerFailed(message, **fields)
        return InvalidRequest(message, **fields)

    async def discover(self, config: Any) -> ProfileLayer:
        payload = await fetch_json(
            "OpenAI",
            config.base_url,
            "/models",
            self._listing_headers(config.api_key),
            timeout=CONTEXT_WINDOW_PROBE_TIMEOUT_SECONDS,
        )
        entries = payload.get("data") if isinstance(payload, dict) else None
        fields: Dict[str, Any] = {}
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and entry.get("id") == config.model:
                tokens = next((value for value in (positive_int(entry.get(key)) for key in WINDOW_KEYS) if value), None)
                if tokens:
                    fields["context_window"] = tokens
        return ProfileLayer(ProfileSource.DISCOVERED, fields)

    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> List[DiscoveredModel]:
        payload = await fetch_json("OpenAI", base_url, "/models", self._listing_headers(api_key))
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ModelListingError("OpenAI returned no model list.")
        models = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            owner = entry.get("owned_by")
            models.append(DiscoveredModel(id=str(entry["id"]), details={"owned_by": owner} if owner else None))
        return sorted_models(models)

    @staticmethod
    def _listing_headers(api_key: Optional[str]) -> Dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _input(self, request: RenderRequest, profile: CapabilityProfile) -> Tuple[List[Dict[str, Any]], List[Notice]]:
        target = self.origin(request.config)
        items: List[Dict[str, Any]] = []
        notices: List[Notice] = []
        for message in request.messages:
            if message.role is Role.SYSTEM:
                items.append({"role": "developer", "content": message.text()})
            elif message.role is Role.USER:
                item = _user_item(message)
                if item is not None:
                    items.append(item)
            elif message.role is Role.TOOL:
                items.extend(_result_item(part, profile) for part in message.tool_results())
            else:
                rendered, notice = self._assistant(message, target)
                items.extend(rendered)
                if notice is not None:
                    notices.append(notice)
        return items, notices

    def _assistant(self, message: Message, target: Origin) -> Tuple[List[Dict[str, Any]], Optional[Notice]]:
        if message.replays_raw_to(target):
            items = _verified_output(message)
            cut = len(items)
            while cut and items[cut - 1].get("type") == "reasoning":
                cut -= 1
            if cut == len(items):
                return items, None
            return items[:cut], _dropped_notice(
                message, "incomplete", "Reasoning that was cut off before its answer was left out."
            )
        notice = None
        origin = message.origin
        if origin is not None and origin.adapter == self.name and any(
            isinstance(part, ReasoningPart) and part.opaque for part in message.parts
        ):
            if origin.credential != target.credential:
                notice = _dropped_notice(message, "credential", "Reasoning made with a different API key was left out.")
            else:
                notice = _dropped_notice(message, "model", "Reasoning made by a different model was left out.")
        return _rebuilt_items(message), notice

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type
from urllib.parse import quote

from src.features.llm.harness.adapter import RenderRequest, StreamDecoder, ToolSpec, WireRequest
from src.features.llm.harness.assembler import parse_args
from src.features.llm.harness.context import user_text
from src.features.llm.harness.errors import (
    AuthFailed,
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
)
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource, ReasoningKind
from src.features.llm.harness.sse import DONE_SENTINEL, SSEFramer
from src.features.llm.harness.transport import HttpAdapter
from src.features.llm.model_listing import DiscoveredModel, ModelListingError, fetch_json, positive_int, sorted_models

if TYPE_CHECKING:
    from src.features.llm.repository import LLMConfig

ADAPTER_NAME = "gemini-interactions"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
PLACEHOLDER_SIGNATURE = "skip_thought_signature_validator"
THINKING_LEVELS = ("minimal", "low", "medium", "high")
TERMINAL_STATUSES = frozenset({"completed", "requires_action", "incomplete", "failed", "cancelled", "budget_exceeded"})
LISTING_PAGE_SIZE = 1000
LISTING_MAX_PAGES = 10
GENERATION_METHODS = frozenset({"generateContent", "createInteraction"})
SIGNATURE_MARKERS = ("thought_signature", "thought signature", "thoughtsignature")
OVERFLOW_MARKERS = ("input token count", "exceeds the maximum number of tokens", "token limit", "context length")
SUMMARY_SEPARATOR = "\n\n"

CODE_CLASSES: Dict[str, Type[HarnessError]] = {
    "rate_limit_exceeded": RateLimited,
    "too_many_requests": RateLimited,
    "quota_exceeded": QuotaExhausted,
    "payment_required": QuotaExhausted,
    "authentication": AuthFailed,
    "permission_denied": AuthFailed,
    "service_unavailable": Overloaded,
    "invalid_request": InvalidRequest,
    "failed_precondition": InvalidRequest,
    "parameter_unknown": InvalidRequest,
    "out_of_range": InvalidRequest,
    "not_found": InvalidRequest,
    "model_not_found": InvalidRequest,
    "unimplemented": InvalidRequest,
}


def api_root(base_url: Optional[str]) -> str:
    url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    for suffix in ("/v1beta", "/v1"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def model_id(model: str) -> str:
    return model[len("models/"):] if model.startswith("models/") else model


def auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    return {"x-goog-api-key": api_key} if api_key else {}


def thought_step(signature: str, summary: Optional[List[Any]] = None) -> Dict[str, Any]:
    step: Dict[str, Any] = {"type": "thought", "signature": signature}
    if summary:
        step["summary"] = summary
    return step


def text_step(text: str) -> Dict[str, Any]:
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def user_step(message: Message) -> Optional[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, ImagePart) and part.data is not None:
            content.append({"type": "image", "data": part.data, "mime_type": part.media_type})
        elif isinstance(part, FilePart) and part.data is not None:
            content.append({"type": "document", "data": part.data, "mime_type": part.media_type})
        elif isinstance(part, CompactionPart):
            content.append({"type": "text", "text": part.summary})
    text = user_text(message)
    if text:
        content.append({"type": "text", "text": text})
    return {"type": "user_input", "content": content} if content else None


def result_content(part: ToolResultPart, profile: CapabilityProfile) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in part.content:
        if isinstance(item, TextPart):
            items.append({"type": "text", "text": item.text})
        elif isinstance(item, JsonPart):
            items.append({"type": "text", "text": json.dumps(item.value, ensure_ascii=False)})
        elif isinstance(item, ImagePart) and item.data is not None and profile.tool_result_media:
            items.append({"type": "image", "data": item.data, "mime_type": item.media_type})
    return items or [{"type": "text", "text": ""}]


def result_step(part: ToolResultPart, wire_ids: Mapping[str, str], profile: CapabilityProfile) -> Dict[str, Any]:
    step: Dict[str, Any] = {
        "type": "function_result",
        "call_id": wire_ids.get(part.call_id, part.call_id),
        "name": part.name,
        "result": result_content(part, profile),
    }
    if part.is_error:
        step["is_error"] = True
    return step


def stored_thought(part: ReasoningPart) -> Dict[str, Any]:
    opaque = part.opaque or {}
    signature = opaque.get("signature")
    if not signature:
        raise OpaqueViolation("A stored Gemini thought has lost its signature, so the history cannot be sent.")
    return thought_step(signature, opaque.get("summary"))


def flagged_for_placeholder(part: ToolCallPart) -> bool:
    return bool((part.opaque or {}).get("placeholder_signature"))


def part_steps(message: Message, profile: CapabilityProfile, native: bool) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    signed = False
    derive = profile.reasoning is not ReasoningKind.NONE
    for part in message.parts:
        if isinstance(part, ReasoningPart):
            if native:
                steps.append(stored_thought(part))
                signed = True
        elif isinstance(part, TextPart):
            steps.append(text_step(part.text))
        elif isinstance(part, CompactionPart):
            steps.append(text_step(part.summary))
        elif isinstance(part, ToolCallPart):
            if not signed and (derive or flagged_for_placeholder(part)):
                steps.append(thought_step(PLACEHOLDER_SIGNATURE))
                signed = True
            steps.append({
                "type": "function_call",
                "id": part.wire_id or part.call_id,
                "name": part.name,
                "arguments": part.args if part.args is not None else {},
            })
    return steps


def call_differs(step: Mapping[str, Any], part: ToolCallPart) -> bool:
    return (
        step.get("id") != (part.wire_id or part.call_id)
        or step.get("name") != part.name
        or (part.args is not None and step.get("arguments") != part.args)
    )


def is_subsequence(wanted: Sequence[Any], available: Sequence[Any]) -> bool:
    remaining = iter(available)
    return all(item in remaining for item in wanted)


def raw_steps(message: Message) -> List[Dict[str, Any]]:
    raw = message.raw
    if not isinstance(raw, list) or not all(isinstance(step, dict) for step in raw):
        raise OpaqueViolation("The stored Gemini steps are not a list of steps, so the history cannot be sent.")
    for step in raw:
        if step.get("type") == "thought" and not step.get("signature"):
            raise OpaqueViolation("A stored Gemini thought step has lost its signature, so the history cannot be sent.")
    calls = [step for step in raw if step.get("type") == "function_call"]
    parts = message.tool_calls()
    if len(calls) != len(parts) or any(call_differs(step, part) for step, part in zip(calls, parts)):
        raise OpaqueViolation("The stored tool calls no longer match the Gemini steps they came from.")
    signatures = [step["signature"] for step in raw if step.get("type") == "thought"]
    carried = [(part.opaque or {}).get("signature") for part in message.reasoning()]
    if not is_subsequence(carried, signatures):
        raise OpaqueViolation("The stored reasoning no longer matches the Gemini thoughts it came from.")
    return list(raw)


def assistant_steps(message: Message, target: Origin, profile: CapabilityProfile) -> List[Dict[str, Any]]:
    if message.replays_raw_to(target):
        return raw_steps(message)
    return part_steps(message, profile, native=message.origin is not None and message.origin.adapter == ADAPTER_NAME)


def input_steps(messages: Sequence[Message], target: Origin, profile: CapabilityProfile) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    wire_ids: Dict[str, str] = {}
    for message in messages:
        for call in message.tool_calls():
            wire_ids[call.call_id] = call.wire_id or call.call_id
        if message.role is Role.USER:
            step = user_step(message)
            if step is not None:
                steps.append(step)
        elif message.role is Role.ASSISTANT:
            steps.extend(assistant_steps(message, target, profile))
        elif message.role is Role.TOOL:
            steps.extend(result_step(part, wire_ids, profile) for part in message.tool_results())
        else:
            raise InvalidRequest("Gemini takes the system prompt as system_instruction, not as a message.")
    return steps


def thinking_config(request: RenderRequest, profile: CapabilityProfile) -> Tuple[Dict[str, Any], bool]:
    if profile.reasoning is ReasoningKind.NONE:
        return {}, False
    levels = profile.reasoning_levels
    reasoning = request.settings.reasoning
    config: Dict[str, Any] = {}
    active = True
    if reasoning is False:
        if not levels or profile.reasoning is ReasoningKind.ALWAYS_ON:
            return {}, True
        config["thinking_level"] = levels[0]
        active = levels[0] != "minimal"
    elif isinstance(reasoning, str) and reasoning in (levels or THINKING_LEVELS):
        config["thinking_level"] = reasoning
    if active and request.config.show_thinking:
        config["thinking_summaries"] = "auto"
    return config, active


def tool_choice(request: RenderRequest, profile: CapabilityProfile, thinking: bool) -> Any:
    choice = request.settings.tool_choice
    if choice is None or not request.tools:
        return None
    allowed = profile.tool_choice_with_reasoning if thinking else profile.tool_choice
    if choice.mode not in allowed:
        return None
    if choice.mode == "named":
        if not choice.name:
            raise InvalidRequest("A named tool choice needs the tool's name.")
        return {"allowed_tools": {"mode": "any", "tools": [choice.name]}}
    return {"none": "none", "required": "any"}.get(choice.mode)


def generation_config(request: RenderRequest, profile: CapabilityProfile) -> Dict[str, Any]:
    settings, config = request.settings, request.config
    generation: Dict[str, Any] = {}
    temperature = settings.temperature if settings.temperature is not None else config.temperature
    if temperature is not None and "temperature" not in profile.sampling_rejected:
        generation["temperature"] = temperature
    if settings.top_p is not None and "top_p" not in profile.sampling_rejected:
        generation["top_p"] = settings.top_p
    max_tokens = settings.max_tokens if settings.max_tokens is not None else config.max_tokens
    if max_tokens:
        generation["max_output_tokens"] = max_tokens
    thinking, active = thinking_config(request, profile)
    generation.update(thinking)
    choice = tool_choice(request, profile, active)
    if choice is not None:
        generation["tool_choice"] = choice
    return generation


def tool_declaration(spec: ToolSpec) -> Dict[str, Any]:
    return {"type": "function", "name": spec.name, "description": spec.description, "parameters": spec.parameters}


def usage_from(raw: Mapping[str, Any]) -> Usage:
    output = raw.get("total_output_tokens")
    return Usage(
        input_tokens=raw.get("total_input_tokens"),
        output_tokens=None if output is None else output + (raw.get("total_thought_tokens") or 0),
        total_tokens=raw.get("total_tokens"),
        reasoning_tokens=raw.get("total_thought_tokens"),
        cached_input_tokens=raw.get("total_cached_tokens"),
        raw=dict(raw),
    )


def finish_reason(status: Optional[str], has_calls: bool) -> FinishReason:
    if status == "requires_action":
        return FinishReason.TOOL_CALLS
    if status == "incomplete":
        return FinishReason.LENGTH
    if status == "completed":
        return FinishReason.TOOL_CALLS if has_calls else FinishReason.STOP
    return FinishReason.UNKNOWN


def upgraded_by_message(cls: Type[HarnessError], message: str) -> Type[HarnessError]:
    if cls is not InvalidRequest:
        return cls
    lowered = message.lower()
    if any(marker in lowered for marker in SIGNATURE_MARKERS):
        return OpaqueViolation
    if any(marker in lowered for marker in OVERFLOW_MARKERS):
        return ContextOverflow
    return cls


def stream_failure(error: Any, fallback: str) -> HarnessError:
    error = error if isinstance(error, dict) else {}
    code = error.get("code") if isinstance(error.get("code"), str) else None
    message = error.get("message") if isinstance(error.get("message"), str) and error.get("message") else fallback
    cls = upgraded_by_message(CODE_CLASSES.get(code or "", ServerFailed), message)
    return cls(message, code=code)


def error_object(body: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        return {}
    if isinstance(payload, list) and payload:
        payload = payload[0]
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, dict) else {}


def detail_items(error: Mapping[str, Any]) -> List[Dict[str, Any]]:
    details = error.get("details")
    return [item for item in details if isinstance(item, dict)] if isinstance(details, list) else []


def is_quota(error: Mapping[str, Any]) -> bool:
    if error.get("code") == "quota_exceeded":
        return True
    for item in detail_items(error):
        for violation in item.get("violations") or []:
            if isinstance(violation, dict) and (
                "PerDay" in str(violation.get("quotaId", "")) or str(violation.get("quotaValue")) == "0"
            ):
                return True
    return False


def retry_delay(error: Mapping[str, Any]) -> Optional[float]:
    for item in detail_items(error):
        delay = item.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


@dataclass
class StepState:
    kind: str
    raw: Dict[str, Any]
    texts: List[str] = field(default_factory=list)
    summary: List[Any] = field(default_factory=list)
    signature: str = ""
    arguments: List[str] = field(default_factory=list)
    start_arguments: Dict[str, Any] = field(default_factory=dict)
    stopped: bool = False


class GeminiDecoder:
    def __init__(self) -> None:
        self._framer = SSEFramer(ADAPTER_NAME)
        self._steps: Dict[int, StepState] = {}
        self._status: Optional[str] = None
        self._usage: Optional[Dict[str, Any]] = None
        self._finished = False

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        return self._records(self._framer.feed(chunk))

    def close(self) -> List[StreamEvent]:
        events = self._records(self._framer.close())
        if not self._finished:
            if self._status not in TERMINAL_STATUSES:
                raise MalformedStream("The Gemini stream ended before the interaction completed.")
            events.extend(self._finish(self._status, None))
        return events

    def _records(self, records: Sequence[Any]) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for record in records:
            if record.data.strip() == DONE_SENTINEL:
                continue
            try:
                payload = json.loads(record.data)
            except ValueError as exc:
                raise MalformedStream("The Gemini stream carried a record that is not JSON.") from exc
            if not isinstance(payload, dict):
                raise MalformedStream("The Gemini stream carried a record that is not a JSON object.")
            if not self._finished:
                events.extend(self._event(payload))
        return events

    def _event(self, payload: Dict[str, Any]) -> List[StreamEvent]:
        kind = payload.get("event_type")
        if kind == "step.start":
            return self._start(payload)
        if kind == "step.delta":
            return self._delta(payload)
        if kind == "step.stop":
            return self._stop(payload)
        if kind == "interaction.status_update":
            self._status = payload.get("status") or self._status
            self._remember_usage(payload.get("usage"))
        elif kind == "interaction.completed":
            interaction = payload.get("interaction")
            interaction = interaction if isinstance(interaction, dict) else {}
            self._remember_usage(interaction.get("usage"))
            return self._finish(interaction.get("status") or self._status, interaction.get("errors"))
        elif kind == "error":
            raise stream_failure(payload.get("error"), "The Gemini stream reported an error.")
        return []

    def _remember_usage(self, usage: Any) -> None:
        if isinstance(usage, dict) and usage:
            self._usage = usage

    def _step(self, payload: Mapping[str, Any]) -> Tuple[int, Optional[StepState]]:
        index = payload.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise MalformedStream("A Gemini step event has no step index.")
        return index, self._steps.get(index)

    def _start(self, payload: Dict[str, Any]) -> List[StreamEvent]:
        index, existing = self._step(payload)
        step = payload.get("step")
        if existing is not None or not isinstance(step, dict):
            raise MalformedStream(f"Gemini step {index} was started twice or without a step.")
        kind = step.get("type")
        if kind == "thought":
            state = StepState(kind, {"type": "thought"}, signature=step.get("signature") or "")
            self._steps[index] = state
            events: List[StreamEvent] = [ReasoningDelta(block=index, text="", summary=True)]
            for item in step.get("summary") or []:
                events.extend(self._summary(index, state, item))
            return events
        if kind == "model_output":
            state = StepState(kind, {"type": "model_output"})
            self._steps[index] = state
            return [event for item in step.get("content") or [] for event in self._text(index, state, item)]
        if kind == "function_call":
            call_id, name = step.get("id"), step.get("name")
            if not call_id or not name:
                raise MalformedStream(f"Gemini function call step {index} has no id or name.")
            arguments = step.get("arguments")
            self._steps[index] = StepState(
                kind,
                {"type": "function_call", "id": call_id, "name": name, "arguments": {}},
                start_arguments=dict(arguments) if isinstance(arguments, dict) else {},
            )
            return [ToolCallStart(block=index, call_id=call_id, name=name, wire_id=call_id)]
        raise MalformedStream(f"Gemini sent a {kind!r} step, which this adapter cannot replay.")

    def _text(self, index: int, state: StepState, item: Any) -> List[StreamEvent]:
        if not isinstance(item, dict) or item.get("type") != "text":
            raise MalformedStream(f"Gemini sent non-text content in step {index}.")
        text = item.get("text") or ""
        state.texts.append(text)
        return [TextDelta(block=index, text=text)] if text else []

    def _summary(self, index: int, state: StepState, item: Any) -> List[StreamEvent]:
        if not isinstance(item, dict):
            raise MalformedStream(f"Gemini sent a thought summary item that is not an object in step {index}.")
        separator = SUMMARY_SEPARATOR if state.summary else ""
        state.summary.append(item)
        if item.get("type") != "text":
            return []
        return [ReasoningDelta(block=index, text=separator + (item.get("text") or ""), summary=True)]

    def _delta(self, payload: Dict[str, Any]) -> List[StreamEvent]:
        index, state = self._step(payload)
        delta = payload.get("delta")
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            self._remember_usage(metadata.get("total_usage"))
        if state is None or state.stopped or not isinstance(delta, dict):
            raise MalformedStream(f"Gemini sent a delta for step {index}, which is not open.")
        kind = delta.get("type")
        if state.kind == "model_output" and kind == "text":
            return self._text(index, state, delta)
        if state.kind == "function_call" and kind == "arguments_delta":
            text = delta.get("arguments") or ""
            state.arguments.append(text)
            return [ToolArgsDelta(block=index, text=text)]
        if state.kind == "thought" and kind == "thought_summary":
            return self._summary(index, state, delta.get("content"))
        if state.kind == "thought" and kind == "thought_signature":
            state.signature = delta.get("signature") or ""
            return []
        raise MalformedStream(f"Gemini sent a {kind!r} delta on a {state.kind} step, which this adapter cannot replay.")

    def _stop(self, payload: Dict[str, Any]) -> List[StreamEvent]:
        index, state = self._step(payload)
        if state is None or state.stopped:
            raise MalformedStream(f"Gemini stopped step {index}, which is not open.")
        state.stopped = True
        self._remember_usage(payload.get("usage"))
        if state.kind == "thought":
            return self._stop_thought(index, state)
        if state.kind == "model_output":
            state.raw["content"] = [{"type": "text", "text": "".join(state.texts)}] if state.texts else []
            return []
        return self._stop_call(index, state)

    def _stop_thought(self, index: int, state: StepState) -> List[StreamEvent]:
        if not state.signature:
            raise MalformedStream(f"Gemini ended thought step {index} without a signature.")
        state.raw["signature"] = state.signature
        events: List[StreamEvent] = [OpaqueDelta(block=index, key="signature", value=state.signature)]
        if state.summary:
            state.raw["summary"] = state.summary
            events.append(OpaqueDelta(block=index, key="summary", value=list(state.summary)))
        return events

    def _stop_call(self, index: int, state: StepState) -> List[StreamEvent]:
        streamed = "".join(state.arguments)
        if streamed.strip():
            parsed = parse_args(streamed)
            state.raw["arguments"] = parsed if parsed is not None else {}
            return [ToolCallEnd(block=index)]
        state.raw["arguments"] = state.start_arguments
        return [ToolCallEnd(block=index, args=dict(state.start_arguments))]

    def _finish(self, status: Optional[str], errors: Any) -> List[StreamEvent]:
        if status == "failed":
            first = errors[0] if isinstance(errors, list) and errors else None
            raise stream_failure(first, "The Gemini interaction failed.")
        if any(not state.stopped for state in self._steps.values()):
            raise MalformedStream("The Gemini interaction completed with steps that never finished.")
        self._finished = True
        ordered = [self._steps[index] for index in sorted(self._steps)]
        events: List[StreamEvent] = []
        if self._usage:
            events.append(UsageReported(usage_from(self._usage)))
        events.append(Finish(
            reason=finish_reason(status, any(state.kind == "function_call" for state in ordered)),
            raw_reason=status,
            raw=[state.raw for state in ordered],
        ))
        return events


class GeminiInteractionsAdapter(HttpAdapter):
    name = ADAPTER_NAME

    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest:
        config = request.config
        if not config.api_key:
            raise AuthFailed("The Gemini configuration has no API key.")
        body: Dict[str, Any] = {
            "model": model_id(config.model),
            "input": input_steps(request.messages, Origin.for_config(ADAPTER_NAME, config), profile),
            "store": False,
            "stream": True,
        }
        if request.system:
            body["system_instruction"] = request.system
        if request.tools:
            body["tools"] = [tool_declaration(spec) for spec in request.tools]
        generation = generation_config(request, profile)
        if generation:
            body["generation_config"] = generation
        if request.settings.response_schema is not None:
            body["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "schema": request.settings.response_schema,
            }
        return WireRequest(
            method="POST",
            url=f"{api_root(config.base_url)}/v1beta/interactions",
            headers={**auth_headers(config.api_key), "content-type": "application/json"},
            body=body,
            stream=True,
        )

    def decoder(self, request: RenderRequest, profile: CapabilityProfile) -> StreamDecoder:
        return GeminiDecoder()

    def classify_error(self, failure: WireFailure) -> HarnessError:
        base = super().classify_error(failure)
        if failure.status is None:
            return base
        error = error_object(failure.body)
        code = base.code or (error.get("status") if isinstance(error.get("status"), str) else None)
        retry_after = base.retry_after
        if failure.status == 429:
            cls: Type[HarnessError] = QuotaExhausted if is_quota(error) else RateLimited
            retry_after = failure.retry_after_seconds()
            if retry_after is None:
                retry_after = retry_delay(error)
        elif failure.status == 402:
            cls = QuotaExhausted
        else:
            cls = upgraded_by_message(type(base), base.message)
        if cls is type(base) and code == base.code and retry_after == base.retry_after:
            return base
        return cls(base.message, status=base.status, code=code, retry_after=retry_after, detail=base.detail)

    async def discover(self, config: LLMConfig) -> ProfileLayer:
        payload = await fetch_json(
            "Gemini",
            api_root(config.base_url),
            f"/v1beta/models/{quote(model_id(config.model), safe='')}",
            auth_headers(config.api_key),
        )
        return ProfileLayer(ProfileSource.DISCOVERED, discovered_fields(payload if isinstance(payload, dict) else {}))

    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> List[DiscoveredModel]:
        models: List[DiscoveredModel] = []
        token: Optional[str] = None
        for _ in range(LISTING_MAX_PAGES):
            path = f"/v1beta/models?pageSize={LISTING_PAGE_SIZE}"
            if token:
                path += f"&pageToken={quote(token, safe='')}"
            payload = await fetch_json("Gemini", api_root(base_url), path, auth_headers(api_key))
            entries = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(entries, list):
                raise ModelListingError("Gemini returned no model list.")
            models.extend(listed_model(entry) for entry in entries if is_listed(entry))
            token = payload.get("nextPageToken")
            if not token:
                break
        return sorted_models(models)


def discovered_fields(model: Mapping[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    window = positive_int(model.get("inputTokenLimit"))
    if window:
        fields["context_window"] = window
    limit = positive_int(model.get("outputTokenLimit"))
    if limit:
        fields["max_output_tokens"] = limit
    if model.get("thinking") is False:
        fields["reasoning"] = ReasoningKind.NONE.value
    return fields


def is_listed(entry: Any) -> bool:
    if not isinstance(entry, dict) or not entry.get("name"):
        return False
    methods = entry.get("supportedGenerationMethods")
    return methods is None or bool(GENERATION_METHODS & set(methods))


def listed_model(entry: Mapping[str, Any]) -> DiscoveredModel:
    details = {
        key: entry[key] for key in ("inputTokenLimit", "outputTokenLimit", "thinking") if entry.get(key) is not None
    }
    return DiscoveredModel(
        id=model_id(str(entry["name"])),
        label=entry.get("displayName") or None,
        details=details or None,
    )

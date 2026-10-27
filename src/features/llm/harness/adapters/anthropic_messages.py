from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from src.features.llm.harness.adapter import RenderRequest, ToolChoice, ToolSpec, WireRequest
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
from src.features.llm.harness.profile import (
    CacheMode,
    CapabilityProfile,
    ProfileLayer,
    ProfileSource,
    ReasoningKind,
    StructuredOutputMode,
)
from src.features.llm.harness.sse import SSEFramer, SSERecord
from src.features.llm.harness.transport import HttpAdapter
from src.features.llm.model_listing import DiscoveredModel, ModelListingError, fetch_json, sorted_models

if TYPE_CHECKING:
    from src.features.llm.repository import LLMConfig

ADAPTER_NAME = "anthropic-messages"
API_VERSION = "2026-10-21"
DEFAULT_BASE_URL = "https://api.anthropic.com"
FALLBACK_MAX_TOKENS = 4096
MODEL_PAGE_SIZE = 1000
TOOL_ID_MAX_LENGTH = 64
BODY_DETAIL_CHARS = 4000

THINKING_BLOCKS = frozenset({"thinking", "redacted_thinking"})

FINISH_REASONS = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALLS,
    "pause_turn": FinishReason.PAUSE,
    "refusal": FinishReason.REFUSAL,
    "model_context_window_exceeded": FinishReason.CONTEXT_WINDOW,
}

QUOTA_MARKERS = ("credit balance", "usage limit")
OVERFLOW_MARKERS = ("prompt is too long", "exceed context limit", "context window", "context size")
PAIRING_MARKERS = ("tool_use ids were found without", "unexpected tool_use_id")

TOOL_ID_INVALID = re.compile(r"[^a-zA-Z0-9_-]")


def api_root(base_url: Optional[str]) -> str:
    root = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"anthropic-version": API_VERSION}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def tool_wire_id(call_id: str) -> str:
    return TOOL_ID_INVALID.sub("_", call_id)[:TOOL_ID_MAX_LENGTH]


def text_blocks(text: str) -> List[Dict[str, Any]]:
    return [{"type": "text", "text": text}] if text.strip() else []


def is_url(reference: Optional[str]) -> bool:
    return bool(reference) and reference.startswith(("http://", "https://"))


def image_block(part: ImagePart) -> Dict[str, Any]:
    if part.data is not None:
        source = {"type": "base64", "media_type": part.media_type, "data": part.data}
    elif is_url(part.ref):
        source = {"type": "url", "url": part.ref}
    else:
        raise InvalidRequest(
            "An image has to be resolved to its data before it is sent to Anthropic.", detail={"ref": part.ref}
        )
    return {"type": "image", "source": source}


def document_block(part: FilePart) -> Dict[str, Any]:
    if part.media_type == "application/pdf" and part.data is not None:
        source = {"type": "base64", "media_type": part.media_type, "data": part.data}
    elif part.media_type == "application/pdf" and is_url(part.ref):
        source = {"type": "url", "url": part.ref}
    elif part.media_type == "text/plain" and part.data is not None:
        try:
            text = base64.b64decode(part.data, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise InvalidRequest("A text file attachment is not valid base64 UTF-8.") from exc
        source = {"type": "text", "media_type": "text/plain", "data": text}
    else:
        raise InvalidRequest(f"Anthropic cannot take a {part.media_type} file attachment.", detail={"name": part.name})
    block: Dict[str, Any] = {"type": "document", "source": source}
    if part.name:
        block["title"] = part.name
    return block


def user_blocks(message: Message) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, ImagePart):
            blocks.append(image_block(part))
        elif isinstance(part, FilePart):
            blocks.append(document_block(part))
        elif isinstance(part, CompactionPart):
            blocks.extend(text_blocks(part.summary))
    blocks.extend(text_blocks(user_text(message)))
    return blocks


def tool_result_block(result: ToolResultPart, tool_ids: Dict[str, str]) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_ids.get(result.call_id) or tool_wire_id(result.call_id),
    }
    content: List[Dict[str, Any]] = []
    for item in result.content:
        if isinstance(item, TextPart):
            content.extend(text_blocks(item.text))
        elif isinstance(item, ImagePart):
            content.append(image_block(item))
        elif isinstance(item, JsonPart):
            content.extend(text_blocks(json.dumps(item.value, ensure_ascii=False)))
    if content:
        block["content"] = content
    if result.is_error:
        block["is_error"] = True
    return block


def blocks_from_parts(message: Message, tool_ids: Dict[str, str]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            blocks.extend(text_blocks(part.text))
        elif isinstance(part, ToolCallPart):
            wire_id = tool_wire_id(part.call_id)
            tool_ids[part.call_id] = wire_id
            blocks.append(
                {"type": "tool_use", "id": wire_id, "name": part.name, "input": part.args if part.args is not None else {}}
            )
    return blocks


def violation(message: Message, index: int, reason: str, **detail: Any) -> OpaqueViolation:
    return OpaqueViolation(
        f"The stored Anthropic reply {message.id} (message {index}) cannot be replayed, "
        f"the content check failed: {reason}.",
        detail={"message_id": message.id, "index": index, "check": "content", "reason": reason, **detail},
    )


def expected_parts(message: Message, index: int, content: List[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    expected: List[Tuple[Any, ...]] = []
    for position, block in enumerate(content):
        kind = block.get("type")
        if kind == "thinking":
            thinking, signature = block.get("thinking"), block.get("signature")
            if not isinstance(thinking, str) or not isinstance(signature, str) or not signature:
                raise violation(message, index, "a thinking block lost its signature", block=position)
            expected.append(("reasoning", False, "", thinking, signature))
        elif kind == "redacted_thinking":
            data = block.get("data")
            if not isinstance(data, str) or not data:
                raise violation(message, index, "a redacted thinking block lost its data", block=position)
            expected.append(("reasoning", True, "", "", data))
        elif kind == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise violation(message, index, "a text block lost its text", block=position)
            expected.append(("text", text))
        elif kind == "tool_use":
            ident, name, args = block.get("id"), block.get("name"), block.get("input")
            if not isinstance(ident, str) or not isinstance(name, str) or not isinstance(args, dict):
                raise violation(message, index, "a tool_use block is malformed", block=position)
            expected.append(("tool_use", ident, name, args))
    return expected


def actual_parts(message: Message) -> List[Tuple[Any, ...]]:
    actual: List[Tuple[Any, ...]] = []
    for part in message.parts:
        if isinstance(part, ReasoningPart):
            opaque = part.opaque or {}
            carried = opaque.get("data") if part.redacted else opaque.get("signature")
            actual.append(("reasoning", part.redacted, part.text, part.summary, carried))
        elif isinstance(part, TextPart):
            if part.text.strip():
                actual.append(("text", part.text))
        elif isinstance(part, ToolCallPart):
            actual.append(("tool_use", part.wire_id, part.name, part.args))
    return actual


def same_part(expected: Tuple[Any, ...], actual: Tuple[Any, ...]) -> bool:
    if expected[0] == "tool_use" and actual[0] == "tool_use" and actual[3] is None:
        return expected[:3] == actual[:3]
    return expected == actual


def check_parts(message: Message, index: int, content: List[Dict[str, Any]]) -> None:
    expected, actual = expected_parts(message, index, content), actual_parts(message)
    if len(expected) != len(actual):
        raise violation(message, index, "its blocks no longer match its parts", blocks=len(expected), parts=len(actual))
    for position, (want, got) in enumerate(zip(expected, actual)):
        if not same_part(want, got):
            raise violation(message, index, f"block {position} ({want[0]}) no longer matches its part", block=position)


class WireMessages:
    def __init__(self, target: Origin) -> None:
        self.target = target
        self.messages: List[Dict[str, Any]] = []
        self.tool_ids: Dict[str, str] = {}
        self.degraded: List[Tuple[str, int]] = []
        self._hasher = hashlib.sha256()
        self._hashed = 0

    def add(self, role: str, blocks: List[Dict[str, Any]]) -> None:
        if not blocks:
            return
        if role == "user" and self.messages and self.messages[-1]["role"] == "user":
            self.messages[-1]["content"].extend(blocks)
        else:
            self.messages.append({"role": role, "content": blocks})

    def prefix_digest(self) -> str:
        while self._hashed < len(self.messages):
            self._hasher.update((canonical(self.messages[self._hashed]) + "\n").encode("ascii"))
            self._hashed += 1
        return self._hasher.hexdigest()

    def replayed_blocks(self, message: Message, index: int) -> List[Dict[str, Any]]:
        raw = message.raw
        content = raw.get("content") if isinstance(raw, dict) else None
        if not isinstance(content, list) or not all(isinstance(block, dict) for block in content):
            raise violation(message, index, "the stored payload is malformed")
        if raw.get("sha256") != digest(content):
            raise violation(message, index, "the stored content blocks were modified")
        check_parts(message, index, content)
        if any(block.get("type") in THINKING_BLOCKS for block in content) and raw.get("prefix") != self.prefix_digest():
            self.degraded.append((message.id, index))
            return blocks_from_parts(message, self.tool_ids)
        for part in message.tool_calls():
            self.tool_ids[part.call_id] = part.wire_id
        return copy.deepcopy(content)

    def assistant_blocks(self, message: Message, index: int) -> List[Dict[str, Any]]:
        if message.replays_raw_to(self.target):
            return self.replayed_blocks(message, index)
        return blocks_from_parts(message, self.tool_ids)


def check_tool_pairs(messages: List[Dict[str, Any]]) -> None:
    awaiting: Set[str] = set()
    for message in messages:
        blocks = message["content"]
        if message["role"] == "assistant":
            if awaiting:
                raise OpaqueViolation(
                    "A tool call has no tool_result in the next message.", detail={"tool_use_ids": sorted(awaiting)}
                )
            awaiting = {block["id"] for block in blocks if block["type"] == "tool_use"}
            continue
        kinds = [block["type"] == "tool_result" for block in blocks]
        if kinds != sorted(kinds, reverse=True):
            raise OpaqueViolation("Every tool_result has to come before any other block of its message.")
        ids = [block["tool_use_id"] for block in blocks if block["type"] == "tool_result"]
        if len(set(ids)) != len(ids) or set(ids) != awaiting:
            raise OpaqueViolation(
                "The tool_result blocks do not match the tool calls of the previous message.",
                detail={"tool_use_ids": sorted(awaiting), "tool_result_ids": ids},
            )
        awaiting = set()
    if awaiting:
        raise OpaqueViolation(
            "A tool call has no tool_result in the next message.", detail={"tool_use_ids": sorted(awaiting)}
        )


def wire_messages(request: RenderRequest) -> WireMessages:
    wire = WireMessages(Origin.for_config(ADAPTER_NAME, request.config))
    for index, message in enumerate(request.messages):
        if message.role is Role.SYSTEM:
            raise InvalidRequest("The system prompt belongs in RenderRequest.system, not in the messages.")
        if message.role is Role.USER:
            wire.add("user", user_blocks(message))
        elif message.role is Role.TOOL:
            wire.add("user", [tool_result_block(result, wire.tool_ids) for result in message.tool_results()])
        else:
            wire.add("assistant", wire.assistant_blocks(message, index))
    check_tool_pairs(wire.messages)
    return wire


def thinking_config(request: RenderRequest, profile: CapabilityProfile) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    kind, wanted = profile.reasoning, request.settings.reasoning
    if kind is ReasoningKind.NONE:
        return None, None
    if wanted is False:
        return (None, None) if kind is ReasoningKind.ALWAYS_ON else ({"type": "disabled"}, None)
    if kind is ReasoningKind.BUDGET or (wanted is None and kind is not ReasoningKind.ALWAYS_ON):
        return None, None
    effort = wanted if isinstance(wanted, str) and wanted in profile.reasoning_levels else None
    return {"type": "adaptive"}, effort


def tool_choice_config(choice: Optional[ToolChoice], profile: CapabilityProfile, thinking: bool) -> Optional[Dict[str, str]]:
    if choice is None or choice.mode == "auto":
        return None
    if choice.mode not in (profile.tool_choice_with_reasoning if thinking else profile.tool_choice):
        return None
    if choice.mode == "none":
        return {"type": "none"}
    if choice.mode == "required":
        return {"type": "any"}
    if choice.mode == "named" and choice.name:
        return {"type": "tool", "name": choice.name}
    return None


def tool_config(spec: ToolSpec, profile: CapabilityProfile) -> Dict[str, Any]:
    schema = dict(spec.parameters) if isinstance(spec.parameters, dict) and spec.parameters else {"properties": {}}
    tool: Dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "input_schema": {"type": "object", **schema},
    }
    if spec.strict and profile.strict_tools:
        tool["strict"] = True
    return tool


def max_tokens_for(request: RenderRequest, profile: CapabilityProfile) -> int:
    wanted = request.settings.max_tokens
    if not wanted:
        return profile.max_output_tokens or FALLBACK_MAX_TOKENS
    return min(wanted, profile.max_output_tokens) if profile.max_output_tokens else wanted


def history_tools(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    names: List[str] = []
    for message in messages:
        for block in message["content"]:
            if block["type"] == "tool_use" and block["name"] not in names:
                names.append(block["name"])
    return [{"name": name, "input_schema": {"type": "object"}} for name in names]


def cache_control_config(ttl: Optional[str]) -> Dict[str, str]:
    if ttl in (None, "5m"):
        return {"type": "ephemeral"}
    if ttl == "1h":
        return {"type": "ephemeral", "ttl": "1h"}
    raise InvalidRequest(f"Anthropic caches prompts for 5m or 1h, not {ttl}.")


def build_body(request: RenderRequest, profile: CapabilityProfile, wire: WireMessages) -> Dict[str, Any]:
    thinking, effort = thinking_config(request, profile)
    body: Dict[str, Any] = {"model": request.config.model, "max_tokens": max_tokens_for(request, profile), "stream": True}
    if request.system:
        body["system"] = request.system
    if thinking:
        body["thinking"] = thinking
    if effort:
        body["output_config"] = {"effort": effort}
    if profile.cache is CacheMode.EXPLICIT:
        body["cache_control"] = cache_control_config(request.settings.cache_ttl)
    if request.tools:
        body["tools"] = [tool_config(spec, profile) for spec in request.tools]
        thinking_on = bool(thinking and thinking["type"] == "adaptive") or profile.reasoning is ReasoningKind.ALWAYS_ON
        choice = tool_choice_config(request.settings.tool_choice, profile, thinking_on)
        if choice:
            body["tool_choice"] = choice
    elif history_tools(wire.messages):
        body["tools"] = history_tools(wire.messages)
        body["tool_choice"] = {"type": "none"}
    body["messages"] = wire.messages
    return body


def usage_from(raw: Dict[str, Any]) -> Usage:
    read, written = raw.get("cache_read_input_tokens"), raw.get("cache_creation_input_tokens")
    fresh, output = raw.get("input_tokens"), raw.get("output_tokens")
    prompt = None if fresh is None else fresh + (read or 0) + (written or 0)
    total = None if prompt is None or output is None else prompt + output
    return Usage(
        input_tokens=prompt,
        output_tokens=output,
        total_tokens=total,
        cached_input_tokens=read,
        cache_write_tokens=written,
        raw=dict(raw),
    )


def error_fields(body: bytes) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        return None, None, None
    return payload_error_fields(payload)


def payload_error_fields(payload: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None, None, None
    kind, message = error.get("type"), error.get("message")
    request_id = payload.get("request_id")
    return (
        kind if isinstance(kind, str) else None,
        message if isinstance(message, str) else None,
        request_id if isinstance(request_id, str) else None,
    )


def build_error(
    status: Optional[int],
    kind: Optional[str],
    message: str,
    *,
    retry_after: Optional[float] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> HarnessError:
    lowered = f"{kind or ''} {message}".lower().replace("`", "")
    extras: Dict[str, Any] = {"status": status, "code": kind, "detail": detail}
    code = status or 0
    if kind in ("authentication_error", "permission_error") or code in (401, 403):
        return AuthFailed(message, **extras)
    if kind == "billing_error" or code == 402 or any(marker in lowered for marker in QUOTA_MARKERS):
        return QuotaExhausted(message, **extras)
    if kind == "overloaded_error" or code == 529:
        return Overloaded(message, retry_after=retry_after, **extras)
    if kind == "rate_limit_error" or code == 429:
        return RateLimited(message, retry_after=retry_after, **extras)
    if kind == "invalid_request_error" or code in (400, 413, 422):
        if ("thinking" in lowered and ("block" in lowered or "signature" in lowered)) or any(
            marker in lowered for marker in PAIRING_MARKERS
        ):
            return OpaqueViolation(message, **extras)
        if any(marker in lowered for marker in OVERFLOW_MARKERS):
            return ContextOverflow(message, **extras)
        return InvalidRequest(message, **extras)
    if kind in ("api_error", "timeout_error") or code >= 500:
        return ServerFailed(message, **extras)
    return InvalidRequest(message, **extras)


@dataclass
class OpenBlock:
    raw: Dict[str, Any]
    chunks: List[str] = field(default_factory=list)
    closed: bool = False


class MessagesDecoder:
    def __init__(self, prefix: str) -> None:
        self._framer = SSEFramer(ADAPTER_NAME)
        self._prefix = prefix
        self._blocks: Dict[int, OpenBlock] = {}
        self._usage: Dict[str, Any] = {}
        self._stop_reason: Optional[str] = None
        self._finished = False

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for record in self._framer.feed(chunk):
            if self._finished:
                break
            events.extend(self._handle(self._payload(record)))
        return events

    def close(self) -> List[StreamEvent]:
        self._framer.close()
        if not self._finished:
            raise MalformedStream("The stream ended before message_stop.")
        return []

    @staticmethod
    def _payload(record: SSERecord) -> Dict[str, Any]:
        try:
            payload = json.loads(record.data)
        except ValueError as exc:
            raise MalformedStream(f"A {record.event or 'data'} event is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise MalformedStream(f"A {record.event or 'data'} event is not an object.")
        return payload

    def _handle(self, payload: Dict[str, Any]) -> List[StreamEvent]:
        kind = payload.get("type")
        if kind == "message_start":
            return self._message_start(payload)
        if kind == "content_block_start":
            return self._block_start(payload)
        if kind == "content_block_delta":
            return self._block_delta(payload)
        if kind == "content_block_stop":
            return self._block_stop(payload)
        if kind == "message_delta":
            return self._message_delta(payload)
        if kind == "message_stop":
            return self._message_stop()
        if kind == "error":
            raise self._error(payload)
        return []

    @staticmethod
    def _error(payload: Dict[str, Any]) -> HarnessError:
        kind, message, request_id = payload_error_fields(payload)
        return build_error(
            None, kind, message or "The stream reported an error.", detail={"request_id": request_id} if request_id else None
        )

    def _usage_event(self, usage: Any) -> List[StreamEvent]:
        if not isinstance(usage, dict):
            return []
        self._usage.update({key: value for key, value in usage.items() if value is not None})
        return [UsageReported(usage_from(self._usage))]

    def _message_start(self, payload: Dict[str, Any]) -> List[StreamEvent]:
        message = payload.get("message")
        return self._usage_event(message.get("usage") if isinstance(message, dict) else None)

    @staticmethod
    def _index(payload: Dict[str, Any]) -> int:
        index = payload.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise MalformedStream("A content block event has no index.")
        return index

    def _block_start(self, payload: Dict[str, Any]) -> List[StreamEvent]:
        index = self._index(payload)
        block = payload.get("content_block")
        if not isinstance(block, dict) or index in self._blocks:
            raise MalformedStream(f"Content block {index} cannot be started.")
        kind = block.get("type")
        if not isinstance(kind, str):
            raise MalformedStream(f"Content block {index} has no type.")
        if kind == "text":
            text = block.get("text") or ""
            self._blocks[index] = OpenBlock({"type": "text", "text": ""}, [text] if text else [])
            return [TextDelta(block=index, text=text)] if text else []
        if kind == "thinking":
            text = block.get("thinking") or ""
            self._blocks[index] = OpenBlock(
                {"type": "thinking", "thinking": "", "signature": block.get("signature") or ""}, [text] if text else []
            )
            return [ReasoningDelta(block=index, text=text, summary=True)]
        if kind == "redacted_thinking":
            data = block.get("data") or ""
            self._blocks[index] = OpenBlock({"type": "redacted_thinking", "data": data})
            return [ReasoningDelta(block=index, text="", redacted=True), OpaqueDelta(block=index, key="data", value=data)]
        if kind == "tool_use":
            ident, name = block.get("id") or "", block.get("name") or ""
            self._blocks[index] = OpenBlock({"type": "tool_use", "id": ident, "name": name, "input": {}})
            return [ToolCallStart(block=index, call_id=ident, name=name, wire_id=ident)]
        self._blocks[index] = OpenBlock(copy.deepcopy(block))
        return []

    def _open(self, index: int) -> OpenBlock:
        block = self._blocks.get(index)
        if block is None or block.closed:
            raise MalformedStream(f"Content block {index} is not open.")
        return block

    def _block_delta(self, payload: Dict[str, Any]) -> List[StreamEvent]:
        index = self._index(payload)
        block = self._open(index)
        delta = payload.get("delta")
        kind = delta.get("type") if isinstance(delta, dict) else None
        if kind == "text_delta" and block.raw["type"] == "text":
            text = delta.get("text") or ""
            block.chunks.append(text)
            return [TextDelta(block=index, text=text)]
        if kind == "thinking_delta" and block.raw["type"] == "thinking":
            text = delta.get("thinking") or ""
            block.chunks.append(text)
            return [ReasoningDelta(block=index, text=text, summary=True)]
        if kind == "signature_delta" and block.raw["type"] == "thinking":
            block.raw["signature"] = delta.get("signature") or ""
            return []
        if kind == "input_json_delta" and block.raw["type"] == "tool_use":
            text = delta.get("partial_json") or ""
            block.chunks.append(text)
            return [ToolArgsDelta(block=index, text=text)]
        return []

    def _block_stop(self, payload: Dict[str, Any]) -> List[StreamEvent]:
        index = self._index(payload)
        block = self._open(index)
        block.closed = True
        kind = block.raw["type"]
        if kind == "text":
            block.raw["text"] = "".join(block.chunks)
        elif kind == "thinking":
            block.raw["thinking"] = "".join(block.chunks)
            return [OpaqueDelta(block=index, key="signature", value=block.raw["signature"])]
        elif kind == "tool_use":
            args = self._tool_args("".join(block.chunks))
            block.raw["input"] = args if args is not None else {}
            return [ToolCallEnd(block=index, args=args)]
        return []

    @staticmethod
    def _tool_args(text: str) -> Optional[Dict[str, Any]]:
        if not text.strip():
            return {}
        try:
            decoded = json.loads(text)
        except ValueError:
            return None
        return decoded if isinstance(decoded, dict) else None

    def _message_delta(self, payload: Dict[str, Any]) -> List[StreamEvent]:
        delta = payload.get("delta")
        if isinstance(delta, dict) and delta.get("stop_reason") is not None:
            self._stop_reason = delta["stop_reason"]
        return self._usage_event(payload.get("usage"))

    def _message_stop(self) -> List[StreamEvent]:
        open_blocks = sorted(index for index, block in self._blocks.items() if not block.closed)
        if open_blocks:
            raise MalformedStream(f"Content block {open_blocks[0]} was not closed before message_stop.")
        content = [
            self._blocks[index].raw
            for index in sorted(self._blocks)
            if self._blocks[index].raw["type"] != "text" or self._blocks[index].raw["text"].strip()
        ]
        self._finished = True
        reason = FINISH_REASONS.get(self._stop_reason, FinishReason.UNKNOWN)
        raw = {"content": content, "sha256": digest(content), "prefix": self._prefix}
        return [Finish(reason=reason, raw_reason=self._stop_reason, raw=raw)]


def capability_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    capabilities = entry.get("capabilities") if isinstance(entry.get("capabilities"), dict) else {}
    fields: Dict[str, Any] = {}
    window, limit = entry.get("max_input_tokens"), entry.get("max_tokens")
    if isinstance(window, int) and window > 0:
        fields["context_window"] = window
    if isinstance(limit, int) and limit > 0:
        fields["max_output_tokens"] = limit
    if "image_input" in capabilities:
        fields["vision"] = supported(capabilities["image_input"])
    if "structured_outputs" in capabilities:
        mode = StructuredOutputMode.NATIVE if supported(capabilities["structured_outputs"]) else StructuredOutputMode.PROMPTED
        fields["structured_output"] = mode.value
    if isinstance(capabilities.get("thinking"), dict):
        fields.update(reasoning_fields(capabilities["thinking"], capabilities.get("effort")))
    return fields


def reasoning_fields(thinking: Dict[str, Any], effort: Any) -> Dict[str, Any]:
    types = thinking.get("types") if isinstance(thinking.get("types"), dict) else {}
    if not supported(thinking):
        return {"reasoning": ReasoningKind.NONE.value}
    if supported(types.get("adaptive")) and supported(effort):
        levels = [name for name, node in effort.items() if name != "supported" and supported(node)]
        return {"reasoning": ReasoningKind.EFFORT.value, "reasoning_levels": levels}
    if supported(types.get("adaptive")) or not supported(types.get("enabled")):
        return {"reasoning": ReasoningKind.TOGGLE.value}
    return {"reasoning": ReasoningKind.BUDGET.value}


def supported(node: Any) -> bool:
    if isinstance(node, bool):
        return node
    return isinstance(node, dict) and node.get("supported") is True


async def fetch_models(base_url: Optional[str], api_key: Optional[str]) -> List[Dict[str, Any]]:
    payload = await fetch_json(
        "Anthropic", api_root(base_url), f"/v1/models?limit={MODEL_PAGE_SIZE}", auth_headers(api_key)
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ModelListingError("Anthropic returned a model list in an unexpected shape.")
    return [item for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]


class AnthropicMessagesAdapter(HttpAdapter):
    name = ADAPTER_NAME

    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest:
        wire = wire_messages(request)
        if wire.degraded:
            logging.info(f"[{ADAPTER_NAME}] sent {len(wire.degraded)} stored repl(ies) without their thinking blocks")
        headers = {**auth_headers(request.config.api_key), "content-type": "application/json", "accept": "text/event-stream"}
        return WireRequest(
            method="POST",
            url=f"{api_root(request.config.base_url)}/v1/messages",
            headers=headers,
            body=build_body(request, profile, wire),
            stream=True,
        )

    def decoder(self, request: RenderRequest, profile: CapabilityProfile) -> MessagesDecoder:
        return MessagesDecoder(wire_messages(request).prefix_digest())

    def classify_error(self, failure: WireFailure) -> HarnessError:
        if failure.status is None:
            return super().classify_error(failure)
        kind, message, request_id = error_fields(failure.body)
        body = failure.body.decode("utf-8", errors="replace")[:BODY_DETAIL_CHARS]
        detail: Dict[str, Any] = {"request_id": request_id} if request_id else {}
        if kind is None and body:
            detail["body"] = body
        return build_error(
            failure.status,
            kind,
            message or f"HTTP {failure.status}",
            retry_after=failure.retry_after_seconds(),
            detail=detail,
        )

    async def discover(self, config: LLMConfig) -> ProfileLayer:
        entries = await fetch_models(config.base_url, config.api_key)
        entry = next((item for item in entries if item["id"] == config.model), None)
        if entry is None:
            raise ModelListingError(f"Anthropic does not list the model {config.model}.")
        return ProfileLayer(ProfileSource.DISCOVERED, capability_fields(entry))

    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> List[DiscoveredModel]:
        models = []
        for item in await fetch_models(base_url, api_key):
            details = {
                key: item[source]
                for key, source in (("context_window", "max_input_tokens"), ("max_output_tokens", "max_tokens"))
                if isinstance(item.get(source), int)
            }
            models.append(
                DiscoveredModel(
                    id=item["id"],
                    label=item.get("display_name") or None,
                    modified_at=item.get("created_at") if isinstance(item.get("created_at"), str) else None,
                    details=details or None,
                )
            )
        return sorted_models(models)

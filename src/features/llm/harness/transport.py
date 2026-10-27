import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional

import httpx

from src.features.llm.harness.adapter import RenderRequest, StreamDecoder, WireRequest
from src.features.llm.harness.errors import (
    AuthFailed,
    ConnectionFailed,
    ContextOverflow,
    HarnessError,
    InvalidRequest,
    Overloaded,
    QuotaExhausted,
    RateLimited,
    ServerFailed,
    WireFailure,
)
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer
from src.features.llm.model_listing import DiscoveredModel
from src.features.llm.repository import LLMConfig
from src.platform.http.tls import client_ssl_context

_BODY_DETAIL_CHARS = 4000
_QUOTA_MARKERS = ("insufficient_quota", "exceeded your current quota", "billing", "credit balance")
_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "exceed_context_size",
    "maximum context length",
    "context window",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "reduce the length",
)


def stream_timeout(config: LLMConfig) -> httpx.Timeout:
    return httpx.Timeout(connect=config.timeout, read=None, write=config.timeout, pool=config.timeout)


def error_text(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(text)
    except ValueError:
        return text
    found = _error_message(payload)
    return found if found else text


def _error_message(payload: Any) -> Optional[str]:
    if isinstance(payload, str):
        return payload or None
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            found = _error_message(payload.get(key))
            if found:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _error_message(item)
            if found:
                return found
    return None


def error_code(body: bytes) -> Optional[str]:
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    source = error if isinstance(error, dict) else payload
    if not isinstance(source, dict):
        return None
    for key in ("code", "type"):
        value = source.get(key)
        if isinstance(value, str) and value:
            return value
    return None


class HttpAdapter(ABC):
    name: str = ""

    @abstractmethod
    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest: ...

    @abstractmethod
    def decoder(self, request: RenderRequest, profile: CapabilityProfile) -> StreamDecoder: ...

    @abstractmethod
    async def discover(self, config: Any) -> ProfileLayer: ...

    @abstractmethod
    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> List[DiscoveredModel]: ...

    async def send(self, wire: WireRequest, request: RenderRequest) -> AsyncIterator[bytes]:
        started = False
        try:
            async with httpx.AsyncClient(
                timeout=stream_timeout(request.config), verify=client_ssl_context()
            ) as client:
                async with client.stream(
                    wire.method, wire.url, headers=dict(wire.headers), content=wire.content()
                ) as response:
                    if not response.is_success:
                        body = await response.aread()
                        raise self.classify_error(
                            WireFailure(status=response.status_code, body=body, headers=dict(response.headers))
                        )
                    started = True
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        yield chunk
        except httpx.HTTPError as exc:
            raise self.classify_error(WireFailure(status=None, in_stream=started, cause=exc)) from exc

    def classify_error(self, failure: WireFailure) -> HarnessError:
        status = failure.status
        text = error_text(failure.body) if failure.body else ""
        code = error_code(failure.body) if failure.body else None
        detail = {"body": failure.body.decode("utf-8", errors="replace")[:_BODY_DETAIL_CHARS]} if failure.body else {}
        if status is None:
            cause = failure.cause
            kind = type(cause).__name__ if cause is not None else None
            return ConnectionFailed((str(cause) if cause is not None else "") or kind or "connection failed", code=kind)
        message = text or f"HTTP {status}"
        lowered = f"{code or ''} {text}".lower()
        extras = {"status": status, "code": code, "detail": detail}
        if status in (401, 403):
            return AuthFailed(message, **extras)
        if status == 429:
            if any(marker in lowered for marker in _QUOTA_MARKERS):
                return QuotaExhausted(message, **extras)
            return RateLimited(message, retry_after=failure.retry_after_seconds(), **extras)
        if status in (400, 413, 422) and any(marker in lowered for marker in _OVERFLOW_MARKERS):
            return ContextOverflow(message, **extras)
        if status == 408:
            return ConnectionFailed(message, **extras)
        if status in (503, 529):
            return Overloaded(message, retry_after=failure.retry_after_seconds(), **extras)
        if status >= 500:
            return ServerFailed(message, **extras)
        return InvalidRequest(message, **extras)

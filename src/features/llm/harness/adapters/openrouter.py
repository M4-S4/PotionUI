import hashlib
import json
import re
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type

from src.features.llm.harness import tool_ids
from src.features.llm.harness.adapter import RenderRequest, WireRequest
from src.features.llm.harness.adapters.openai_chat import (
    REASONING_BLOCK,
    OpenAIChatAdapter,
    OpenAIChatDecoder,
)
from src.features.llm.harness.errors import (
    AuthFailed,
    HarnessError,
    InvalidRequest,
    MalformedStream,
    OpaqueViolation,
    QuotaExhausted,
    RateLimited,
    WireFailure,
)
from src.features.llm.harness.events import OpaqueDelta, ReasoningDelta, StreamEvent, UsageReported
from src.features.llm.harness.messages import Message, Origin, Role
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource, ReasoningKind, ToolIdRule
from src.features.llm.harness.rendering import bearer_headers
from src.features.llm.model_listing import DiscoveredModel, ModelListingError, fetch_json, positive_int, sorted_models
from src.features.llm.repository import LLMConfig

ADAPTER_NAME = "openrouter"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
ATTRIBUTION_HEADERS = {"HTTP-Referer": "https://potionui.com", "X-OpenRouter-Title": "PotionUI"}
REASONING_FIELD = "reasoning"
DETAILS_KEY = "reasoning_details"
DIGEST_KEY = "details_digest"
EFFORT_LEVELS = ("minimal", "low", "medium", "high")
TOOL_CHOICE_MODES = ("auto", "none", "named")
TOKENS_PER_PRICE = Decimal(1_000_000)
STREAM_ERROR_STATUS = 502
STREAM_ERROR_MESSAGE = "The provider ended the stream with an error."

ERROR_KINDS: Dict[int, Type[HarnessError]] = {
    401: AuthFailed,
    402: QuotaExhausted,
    403: AuthFailed,
    429: RateLimited,
}


def details_digest(details: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(list(details), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def provider_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def needs_same_provider(details: Sequence[Mapping[str, Any]]) -> bool:
    return any(item.get("type") == "reasoning.encrypted" or item.get("signature") for item in details)


def reasoning_object(setting: Any, configured: Any) -> Optional[Dict[str, Any]]:
    value = setting if setting is not None else configured
    if value is None:
        return None
    if isinstance(value, bool):
        return {"enabled": value}
    text = str(value).strip()
    if text.isdigit():
        return {"max_tokens": int(text)}
    return {"effort": text}


def stored_details(message: Message, target: Origin) -> Optional[List[Dict[str, Any]]]:
    if not message.replays_raw_to(target) or not message.reasoning():
        return None
    raw = message.raw
    details = raw.get(DETAILS_KEY) if isinstance(raw, dict) else None
    if not isinstance(details, list) or not details or not all(isinstance(item, dict) for item in details):
        raise OpaqueViolation("The stored OpenRouter reasoning is not a list of reasoning blocks, so the history cannot be sent.")
    anchors = {(part.opaque or {}).get(DIGEST_KEY) for part in message.reasoning()}
    if details_digest(details) not in anchors:
        raise OpaqueViolation("The stored OpenRouter reasoning blocks no longer match the reasoning they came from.")
    return details


class OpenRouterDecoder(OpenAIChatDecoder):
    def __init__(
        self,
        seed: str,
        rule: ToolIdRule,
        classify: Callable[[WireFailure], HarnessError],
        label: str = ADAPTER_NAME,
    ) -> None:
        super().__init__(seed, rule, classify, label)
        self._details: List[Dict[str, Any]] = []
        self._provider: Optional[str] = None
        self._reasoned = False
        self._json_body: Optional[bytearray] = None
        self._sniffed = False

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        if not self._sniffed:
            lead = chunk.lstrip()
            if not lead:
                return []
            self._sniffed = True
            if lead[:1] == b"{":
                self._json_body = bytearray()
        if self._json_body is not None:
            self._json_body.extend(chunk)
            self._check_json_body(final=False)
            return []
        return super().feed(chunk)

    def close(self) -> List[StreamEvent]:
        if self._json_body is not None:
            self._check_json_body(final=True)
        return super().close()

    def _check_json_body(self, final: bool) -> None:
        try:
            data = json.loads(bytes(self._json_body))
        except ValueError:
            if final:
                raise MalformedStream("OpenRouter answered with a body that is neither an event stream nor JSON.")
            return
        if isinstance(data, dict) and data.get("error"):
            raise self._in_band(data)
        raise MalformedStream("OpenRouter answered a streaming request with a plain JSON body.")

    def _in_band(self, data: Mapping[str, Any]) -> HarnessError:
        error = data.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        if isinstance(code, str) and code.isdigit():
            code = int(code)
        status = code if isinstance(code, int) and not isinstance(code, bool) else 500
        return self._classify(WireFailure(status=status, body=json.dumps(data).encode("utf-8"), in_stream=True))

    def _raise_in_band(self, data: Dict[str, Any]) -> None:
        if data.get("error"):
            raise self._in_band(data)
        choices = data.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        if isinstance(first, dict) and first.get("finish_reason") == "error":
            raise self._in_band({"error": {"code": STREAM_ERROR_STATUS, "message": STREAM_ERROR_MESSAGE}})

    def _chunk(self, data: Dict[str, Any]) -> List[StreamEvent]:
        provider = data.get("provider")
        if isinstance(provider, str) and provider and self._provider is None:
            self._provider = provider
        events = super()._chunk(data)
        if any(isinstance(event, ReasoningDelta) for event in events):
            self._reasoned = True
        return [openrouter_usage(event) if isinstance(event, UsageReported) else event for event in events]

    def _delta(self, delta: Dict[str, Any]) -> List[StreamEvent]:
        details = delta.get(DETAILS_KEY)
        if isinstance(details, list):
            self._collect(details)
        events = super()._delta(delta)
        if isinstance(details, list) and not any(isinstance(event, ReasoningDelta) for event in events):
            return [*detail_text(details), *events]
        return events

    def _collect(self, entries: Sequence[Any]) -> None:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            target = self._same_block(entry)
            if target is None:
                self._details.append(dict(entry))
                continue
            for key, value in entry.items():
                if key in ("text", "summary") and isinstance(value, str):
                    target[key] = (target.get(key) or "") + value
                elif value is not None:
                    target[key] = value

    def _same_block(self, entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        index = entry.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            return None
        for known in reversed(self._details):
            if known.get("index") != index or known.get("type") != entry.get("type"):
                continue
            ids = {known.get("id"), entry.get("id")} - {None, ""}
            return known if len(ids) <= 1 else None
        return None

    def _finish(self, explicit_end: bool) -> List[StreamEvent]:
        events = super()._finish(explicit_end)
        if not self._details:
            return events
        finish = events[-1]
        before = events[:-1]
        if not self._reasoned and not any(isinstance(event, ReasoningDelta) for event in before):
            before.append(ReasoningDelta(block=REASONING_BLOCK, text="", redacted=True))
        before.append(OpaqueDelta(block=REASONING_BLOCK, key=DIGEST_KEY, value=details_digest(self._details)))
        raw = {"provider": self._provider, DETAILS_KEY: self._details}
        return [*before, replace(finish, raw=raw)]


def detail_text(details: Sequence[Any]) -> List[StreamEvent]:
    events: List[StreamEvent] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        summary = item.get("type") == "reasoning.summary"
        text = item.get("summary" if summary else "text")
        if item.get("type") in ("reasoning.summary", "reasoning.text") and isinstance(text, str) and text:
            events.append(ReasoningDelta(block=REASONING_BLOCK, text=text, summary=summary))
    return events


def openrouter_usage(event: UsageReported) -> UsageReported:
    raw = event.usage.raw if isinstance(event.usage.raw, dict) else {}
    cost = raw.get("cost")
    prompt_details = raw.get("prompt_tokens_details")
    written = prompt_details.get("cache_write_tokens") if isinstance(prompt_details, dict) else None
    return UsageReported(replace(
        event.usage,
        cost=float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
        cache_write_tokens=written if isinstance(written, int) and not isinstance(written, bool) else None,
    ))


def error_object(body: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        return {}
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, dict) else {}


def per_million(value: Any) -> Optional[float]:
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    return float(amount * TOKENS_PER_PRICE) if amount.is_finite() and amount >= 0 else None


def discovered_prices(pricing: Any) -> Optional[Dict[str, float]]:
    if not isinstance(pricing, dict):
        return None
    prices = {
        name: per_million(pricing.get(key))
        for name, key in (
            ("input", "prompt"),
            ("output", "completion"),
            ("cached_input", "input_cache_read"),
            ("cache_write", "input_cache_write"),
        )
    }
    reasoning = per_million(pricing.get("internal_reasoning"))
    if reasoning:
        prices["reasoning"] = reasoning
    if prices["input"] is None or prices["output"] is None:
        return None
    return {name: value for name, value in prices.items() if value is not None}


def discovered_fields(entry: Mapping[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    top = entry.get("top_provider") if isinstance(entry.get("top_provider"), dict) else {}
    window = positive_int(entry.get("context_length")) or positive_int(top.get("context_length"))
    if window:
        fields["context_window"] = window
    limit = positive_int(top.get("max_completion_tokens"))
    if limit:
        fields["max_output_tokens"] = limit
    supported = entry.get("supported_parameters")
    if isinstance(supported, list):
        names = {name for name in supported if isinstance(name, str)}
        fields["tools"] = "tools" in names
        if "tools" in names and "tool_choice" in names:
            fields["tool_choice"] = list(TOOL_CHOICE_MODES)
        if "tools" in names and "parallel_tool_calls" in names:
            fields["parallel_tools"] = True
        if "reasoning" in names:
            fields["reasoning"] = ReasoningKind.EFFORT.value
            fields["reasoning_levels"] = list(EFFORT_LEVELS)
        else:
            fields["reasoning"] = ReasoningKind.NONE.value
    architecture = entry.get("architecture")
    modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else None
    if isinstance(modalities, list):
        fields["vision"] = "image" in modalities
    prices = discovered_prices(entry.get("pricing"))
    if prices:
        fields["prices"] = prices
    return fields


def listing_headers(api_key: Optional[str]) -> Dict[str, str]:
    return {**bearer_headers(api_key), **ATTRIBUTION_HEADERS}


class OpenRouterAdapter(OpenAIChatAdapter):
    name = ADAPTER_NAME
    display_name = "OpenRouter"

    def build_body(self, request: RenderRequest, profile: CapabilityProfile) -> Dict[str, Any]:
        field = profile.reasoning_field or REASONING_FIELD
        body = super().build_body(request, replace(profile, reasoning_field=field))
        if "max_completion_tokens" in body:
            body["max_tokens"] = body.pop("max_completion_tokens")
        reasoning = reasoning_object(request.settings.reasoning, body.pop("reasoning_effort", None))
        if reasoning is not None:
            body["reasoning"] = reasoning
        served = self._replay_details(request, body["messages"], field)
        if served:
            body["provider"] = {"order": [served], "allow_fallbacks": False}
        return body

    def _replay_details(self, request: RenderRequest, rendered: List[Dict[str, Any]], field: str) -> Optional[str]:
        target = Origin.for_config(self.name, request.config)
        assistants = [message for message in request.messages if message.role is Role.ASSISTANT]
        entries = [entry for entry in rendered if entry["role"] == "assistant"]
        if len(assistants) != len(entries):
            raise InvalidRequest("The rendered assistant messages no longer line up with the history.")
        pinned: Optional[str] = None
        for message, entry in zip(assistants, entries):
            if "tool_calls" not in entry:
                continue
            details = stored_details(message, target)
            if details is None:
                continue
            entry.pop(field, None)
            entry[DETAILS_KEY] = deepcopy(details)
            if needs_same_provider(details):
                provider = message.raw.get("provider")
                slug = provider_slug(provider) if isinstance(provider, str) else ""
                pinned = slug or pinned
        return pinned

    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest:
        wire = super().render(request, profile)
        return replace(wire, headers={**wire.headers, **ATTRIBUTION_HEADERS})

    def decoder(self, request: RenderRequest, profile: CapabilityProfile) -> OpenRouterDecoder:
        return OpenRouterDecoder(
            seed=tool_ids.seed_for(request.messages),
            rule=profile.tool_id_rule,
            classify=self.classify_error,
            label=self.name,
        )

    def classify_error(self, failure: WireFailure) -> HarnessError:
        base = super().classify_error(failure)
        if failure.in_stream:
            base.detail["in_stream"] = True
        if failure.status is None:
            return base
        metadata = error_object(failure.body).get("metadata")
        detail = dict(base.detail)
        message = base.message
        if isinstance(metadata, dict):
            detail["metadata"] = metadata
            provider = metadata.get("provider_name")
            if isinstance(provider, str) and provider:
                message = f"{message} ({provider})"
        kind = ERROR_KINDS.get(failure.status, type(base))
        retry_after = failure.retry_after_seconds() if kind is RateLimited else base.retry_after
        return kind(message, status=failure.status, code=base.code, retry_after=retry_after, detail=detail)

    async def _listing(self, base_url: str, api_key: Optional[str]) -> List[Dict[str, Any]]:
        payload = await fetch_json(self.display_name, base_url, "/models", listing_headers(api_key))
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ModelListingError(f"{self.display_name} returned no model list.")
        return [entry for entry in entries if isinstance(entry, dict) and entry.get("id")]

    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> List[DiscoveredModel]:
        models = []
        for entry in await self._listing(base_url, api_key):
            window = positive_int(entry.get("context_length"))
            models.append(DiscoveredModel(
                id=str(entry["id"]),
                label=entry.get("name") or None,
                details={"context_length": window} if window else None,
            ))
        return sorted_models(models)

    async def discover(self, config: LLMConfig) -> ProfileLayer:
        for entry in await self._listing(config.base_url, config.api_key):
            if entry["id"] == config.model:
                return ProfileLayer(ProfileSource.DISCOVERED, discovered_fields(entry))
        return ProfileLayer(ProfileSource.DISCOVERED, {})

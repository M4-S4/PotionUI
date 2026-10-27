import asyncio
import logging
import time
from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Union

from src.features.llm import context_budget, trace_collector
from src.features.llm.context_windows import ContextWindows, WindowProbe
from src.features.llm.harness.adapter import Adapter, RenderRequest, Settings, ToolSpec
from src.features.llm.harness.assembler import StepAssembler
from src.features.llm.harness.build import new_id, user_message
from src.features.llm.harness.cost import priced
from src.features.llm.harness.errors import HarnessError
from src.features.llm.harness.events import FinishReason, StreamEvent, Usage
from src.features.llm.harness.messages import Message, Origin, credential_fingerprint
from src.features.llm.harness.ports import ProfileResolver
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.model_listing import DiscoveredModel
from src.features.llm.repository import LLMConfig, LLMRepository
from src.features.llm.types import LLMType, LLMTypes

_NON_PARAM_KEYS = (
    "model", "messages", "tools", "stream", "stream_options",
    "input", "contents", "steps", "system", "system_instruction", "instructions",
)
EMPTY_RESPONSE_ATTEMPTS = 3


@dataclass(frozen=True)
class PreparedCall:
    config: LLMConfig
    entry: LLMType
    profile: CapabilityProfile
    request: RenderRequest
    origin: Origin
    ledger: Dict[str, Any]


@dataclass(frozen=True)
class Completion:
    message: Message
    usage: Optional[Usage]
    finish: Optional[FinishReason]
    finish_raw: Optional[str]

    @property
    def text(self) -> str:
        return self.message.text()


def _traced_message(traced: StepAssembler) -> Optional[Message]:
    try:
        return traced.message()
    except Exception:
        return None


def _usage_payload(usage: Usage) -> Dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cost": usage.cost,
    }


def _error_payload(failure: Optional[BaseException]) -> Optional[Dict[str, Any]]:
    if failure is None:
        return None
    if isinstance(failure, HarnessError):
        return failure.to_json()
    if isinstance(failure, (asyncio.CancelledError, GeneratorExit)):
        return {"kind": "cancelled", "message": "The call was stopped before it finished."}
    return {"kind": "error", "message": str(failure)}


def _error_body(failure: Optional[BaseException]) -> Optional[str]:
    body = failure.detail.get("body") if isinstance(failure, HarnessError) else None
    return body if isinstance(body, str) and body else None


def _reasoning_options(settings: Settings) -> Optional[Dict[str, Any]]:
    return {"think": settings.reasoning} if settings.reasoning is not None else None


class LLMGateway:
    def __init__(
        self,
        llm_repository: LLMRepository,
        types: LLMTypes,
        profiles: ProfileResolver,
        windows: ContextWindows,
    ) -> None:
        self.repository = llm_repository
        self.types = types
        self.profiles = profiles
        self.windows = windows

    def provider_types(self) -> List[Dict[str, Any]]:
        return self.types.describe()

    def supports_model_listing(self, type_id: str) -> bool:
        entry = self.types.get(type_id)
        return entry is not None and callable(getattr(entry.adapter, "list_models", None))

    async def list_models(self, type_id: str, base_url: str, api_key: Optional[str] = None) -> List[DiscoveredModel]:
        if not self.supports_model_listing(type_id):
            raise ValueError(f"LLM type '{type_id}' cannot list models")
        return await self.types.require(type_id).adapter.list_models(base_url, api_key)

    def supports_context_window_discovery(self, type_id: str) -> bool:
        return self.windows.supports(type_id)

    async def refresh_context_window(self, type_id: str, base_url: str, api_key: Optional[str], model: str) -> WindowProbe:
        return await self.windows.refresh(type_id, base_url, api_key, model)

    async def ensure_context_window(self, config: LLMConfig) -> None:
        await self.windows.ensure(config)

    def context_capacity_for(self, config: LLMConfig) -> context_budget.CapacityInfo:
        return self.windows.capacity(config)

    @staticmethod
    def _safe_provider_hook(provider_method: Optional[Any], config: LLMConfig, label: str) -> Optional[Any]:
        if provider_method is None:
            return None
        try:
            return provider_method(config)
        except Exception:
            logging.debug("[LLMGateway] %s failed for '%s'", label, config.id, exc_info=True)
            return None

    def context_profile_for(self, config: LLMConfig, settings: Settings = Settings()) -> context_budget.ContextProfile:
        adapter = self.types.require(config.type).adapter
        capacity = self.windows.capacity(config)
        max_tokens = settings.max_tokens if settings.max_tokens is not None else config.max_tokens
        thinking_enabled = context_budget.resolve_thinking_enabled(config, _reasoning_options(settings))
        reserve = context_budget.resolve_reserve(
            max_tokens=max_tokens, window_tokens=capacity.capacity_tokens, thinking_enabled=thinking_enabled,
        )
        return context_budget.ContextProfile(
            capacity=capacity,
            reserve_tokens=reserve.reserve_tokens,
            counter=self._safe_provider_hook(getattr(adapter, "token_counter", None), config, "token_counter"),
            messages_counter=self._safe_provider_hook(
                getattr(adapter, "messages_token_counter", None), config, "messages_token_counter",
            ),
            image_tokens_override=context_budget.resolve_image_token_override(config),
            max_tokens=max_tokens,
            thinking_enabled=reserve.thinking_enabled,
            thinking_allowance_tokens=reserve.thinking_allowance_tokens,
        )

    def estimate_context_budget(
        self,
        config: LLMConfig,
        system: Optional[str],
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        settings: Settings = Settings(),
    ) -> context_budget.BudgetOutcome:
        return context_budget.budget_with_profile(
            self.context_profile_for(config, settings),
            system_message=system,
            messages=messages,
            tool_schemas=[tool.function_schema() for tool in tools] or None,
        )

    def resolve_system(self, config: LLMConfig, custom_system_message: Optional[str] = None) -> str:
        if custom_system_message:
            return custom_system_message
        if config.disable_system_prompt:
            return ""
        return config.system_message

    def _config(self, target: Union[str, LLMConfig]) -> LLMConfig:
        if isinstance(target, LLMConfig):
            config = target
        else:
            config = self.repository.get_configuration(target)
            if not config:
                raise ValueError(f"LLM configuration '{target}' not found")
        if not config.enabled:
            raise ValueError(f"LLM configuration '{config.id}' is disabled")
        return config

    async def prepare(
        self,
        target: Union[str, LLMConfig],
        messages: Sequence[Message],
        *,
        system: Optional[str] = None,
        settings: Settings = Settings(),
        tools: Sequence[ToolSpec] = (),
    ) -> PreparedCall:
        config = self._config(target)
        entry = self.types.require(config.type)
        if not isinstance(entry.adapter, Adapter):
            raise ValueError(f"LLM type '{config.type}' has no harness adapter")
        await self.windows.ensure(config)
        profile = self.profiles.resolve(config)
        system_message = self.resolve_system(config, system)
        outcome = self.estimate_context_budget(config, system_message, messages, tools, settings)
        ledger = outcome.ledger
        if ledger["max_tokens_sent"] != ledger["max_tokens"]:
            settings = replace(settings, max_tokens=ledger["max_tokens_sent"])
        origin = Origin(
            adapter=entry.adapter.name,
            provider=config.type,
            model=config.model,
            credential=credential_fingerprint(config.base_url, config.api_key),
        )
        request = RenderRequest(
            config=config, system=system_message, messages=tuple(outcome.messages), tools=tuple(tools), settings=settings,
        )
        return PreparedCall(config=config, entry=entry, profile=profile, request=request, origin=origin, ledger=ledger)

    async def run(self, prepared: PreparedCall) -> AsyncIterator[StreamEvent]:
        adapter = prepared.entry.adapter
        wire = adapter.render(prepared.request, prepared.profile)
        decoder = adapter.decoder(prepared.request, prepared.profile)
        traced = StepAssembler(new_id("trace"), prepared.origin)
        capture = trace_collector.RawCapture()
        started = time.monotonic()
        failure: Optional[BaseException] = None
        try:
            async with aclosing(adapter.send(wire, prepared.request)) as chunks:
                async for chunk in chunks:
                    capture.add(chunk)
                    for event in decoder.feed(chunk):
                        traced.feed(event)
                        yield event
            for event in decoder.close():
                traced.feed(event)
                yield event
        except BaseException as error:
            failure = error
            raise
        finally:
            self._record_trace(prepared, wire.body, traced, capture, started, failure)

    async def release(self, target: Union[str, LLMConfig]) -> None:
        config = self._config(target)
        release = getattr(self.types.require(config.type).adapter, "release", None)
        if release is not None:
            await release(config)

    @staticmethod
    def _record_trace(
        prepared: PreparedCall,
        body: Any,
        traced: StepAssembler,
        capture: trace_collector.RawCapture,
        started: float,
        failure: Optional[BaseException],
    ) -> None:
        if not trace_collector.is_active():
            return
        try:
            message = _traced_message(traced)
            usage = priced(traced.usage, prepared.profile) or Usage()
            params = body if isinstance(body, dict) else {}
            trace_collector.record(
                provider=prepared.config.type,
                model=prepared.config.model,
                request_system=prepared.request.system,
                request_messages=params.get("messages") if isinstance(params.get("messages"), list) else [],
                request_params={key: value for key, value in params.items() if key not in _NON_PARAM_KEYS},
                request_tools=[tool.name for tool in prepared.request.tools] or None,
                response_text=message.text() if message is not None else None,
                response_tool_calls=[
                    {"id": call.wire_id or call.call_id, "name": call.name, "arguments": call.args if call.args is not None else call.args_raw}
                    for call in (message.tool_calls() if message is not None else ())
                ] or None,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                duration_ms=int((time.monotonic() - started) * 1000),
                adapter=prepared.origin.adapter,
                request_body=body,
                response_raw=capture.text() or _error_body(failure),
                usage=_usage_payload(usage),
                cost=usage.cost,
                error=_error_payload(failure),
            )
        except Exception:
            logging.warning("[LLMGateway] recording the call trace failed", exc_info=True)

    async def stream(
        self,
        target: Union[str, LLMConfig],
        messages: Sequence[Message],
        *,
        system: Optional[str] = None,
        settings: Settings = Settings(),
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[StreamEvent]:
        prepared = await self.prepare(target, messages, system=system, settings=settings, tools=tools)
        async with aclosing(self.run(prepared)) as events:
            async for event in events:
                yield event

    async def complete(
        self,
        target: Union[str, LLMConfig],
        messages: Union[str, Sequence[Message]],
        *,
        system: Optional[str] = None,
        settings: Settings = Settings(),
        tools: Sequence[ToolSpec] = (),
    ) -> Completion:
        if isinstance(messages, str):
            messages = [user_message(messages)]
        prepared = await self.prepare(target, messages, system=system, settings=settings, tools=tools)
        for attempt in range(1, EMPTY_RESPONSE_ATTEMPTS + 1):
            assembler = StepAssembler(new_id("msg"), prepared.origin)
            async with aclosing(self.run(prepared)) as events:
                async for event in events:
                    assembler.feed(event)
            message = assembler.message()
            if message.text() or message.tool_calls() or attempt == EMPTY_RESPONSE_ATTEMPTS:
                break
            logging.warning(f"[LLM] Empty completion from '{prepared.config.id}' (attempt {attempt}/{EMPTY_RESPONSE_ATTEMPTS}), retrying")
        finish = assembler.finish
        return Completion(
            message=message,
            usage=priced(assembler.usage, prepared.profile),
            finish=finish.reason if finish is not None else None,
            finish_raw=finish.raw_reason if finish is not None else None,
        )

    async def test_configuration(self, config: LLMConfig) -> Dict[str, Any]:
        try:
            completion = await self.complete(
                config,
                [user_message("Hello, this is a test. Please respond with 'Test successful!'")],
                system=config.system_message,
            )
        except Exception:
            logging.error(f"[Chat] Configuration test failed for '{config.id}'", exc_info=True)
            return {"success": False, "error": "Failed to reach the configured LLM provider."}
        usage = completion.usage or Usage()
        return {
            "success": True,
            "response": completion.text,
            "model": config.model,
            "tokens_used": usage.total_tokens,
        }

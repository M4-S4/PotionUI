import logging
import time
from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Sequence, Tuple, Union

from src.features.llm import context_budget, trace_collector
from src.features.llm.clients import LLMClient, LLMResponse
from src.features.llm.context_windows import ContextWindows, WindowProbe
from src.features.llm.harness.adapter import Adapter, RenderRequest, Settings, ToolSpec
from src.features.llm.harness.assembler import StepAssembler
from src.features.llm.harness.build import new_id
from src.features.llm.harness.context import user_text
from src.features.llm.harness.events import FinishReason, StreamEvent, Usage
from src.features.llm.harness.messages import ImagePart, Message, Origin, Role, credential_fingerprint
from src.features.llm.harness.ports import ProfileResolver
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.model_listing import DiscoveredModel
from src.features.llm.repository import LLMConfig, LLMRepository
from src.features.llm.types import LLMType, LLMTypes

_NON_PARAM_KEYS = ("model", "messages", "tools", "stream", "stream_options")


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


def budget_entries(messages: Sequence[Message]) -> Tuple[List[Dict[str, Any]], List[Message]]:
    entries: List[Dict[str, Any]] = []
    owners: List[Message] = []
    for message in messages:
        for entry in _budget_entries(message):
            entries.append(entry)
            owners.append(message)
    return entries, owners


def _budget_entries(message: Message) -> List[Dict[str, Any]]:
    if message.role is Role.TOOL:
        return [
            {"role": "tool", "content": part.text(), "tool_call_id": part.call_id, "name": part.name}
            for part in message.tool_results()
        ]
    if message.role is Role.USER:
        return [{"role": "user", "content": user_text(message)}]
    entry: Dict[str, Any] = {"role": message.role.value, "content": message.text()}
    reasoning = "".join(part.text or part.summary for part in message.reasoning())
    if reasoning:
        entry["reasoning"] = reasoning
    calls = message.tool_calls()
    if calls:
        entry["tool_calls"] = [
            {
                "id": call.wire_id or call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.args if call.args is not None else (call.args_raw or "")},
            }
            for call in calls
        ]
    return [entry]


def has_image(messages: Sequence[Message]) -> bool:
    for message in messages:
        for part in message.parts:
            if isinstance(part, ImagePart):
                return True
        for result in message.tool_results():
            if any(isinstance(item, ImagePart) for item in result.content):
                return True
    return False


def _kept(messages: Sequence[Message], entries: List[Dict[str, Any]], owners: List[Message], kept_entries: List[Dict[str, Any]]) -> Tuple[Message, ...]:
    kept_ids = {id(entry) for entry in kept_entries}
    kept_owners = {id(owner) for entry, owner in zip(entries, owners) if id(entry) in kept_ids}
    return tuple(message for message in messages if id(message) in kept_owners)


def _settings_options(settings: Settings) -> Optional[Dict[str, Any]]:
    options = {
        key: value
        for key, value in (
            ("temperature", settings.temperature),
            ("top_p", settings.top_p),
            ("top_k", settings.top_k),
            ("min_p", settings.min_p),
            ("max_tokens", settings.max_tokens),
            ("think", settings.reasoning),
        )
        if value is not None
    }
    return options or None


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

    def context_profile_for(
        self,
        config: LLMConfig,
        options_override: Optional[Dict[str, Any]] = None,
    ) -> context_budget.ContextProfile:
        adapter = self.types.require(config.type).adapter
        capacity = self.windows.capacity(config)
        max_tokens = (options_override or {}).get("max_tokens", config.max_tokens)
        thinking_enabled = context_budget.resolve_thinking_enabled(config, options_override)
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
        system_message: Optional[str],
        messages: List[Dict[str, Any]],
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[str] = None,
        options_override: Optional[Dict[str, Any]] = None,
    ) -> context_budget.BudgetOutcome:
        inputs = self.context_profile_for(config, options_override)
        return context_budget.budget_with_profile(
            inputs, system_message=system_message, messages=messages,
            tool_schemas=tool_schemas, image_data=image_data,
        )

    def budget_messages(
        self,
        config: LLMConfig,
        system: Optional[str],
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        settings: Settings = Settings(),
    ) -> Tuple[Tuple[Message, ...], Dict[str, Any]]:
        entries, owners = budget_entries(messages)
        outcome = self.estimate_context_budget(
            config,
            system,
            entries,
            [tool.function_schema() for tool in tools] or None,
            "image" if has_image(messages) else None,
            _settings_options(settings),
        )
        return _kept(messages, entries, owners, outcome.messages), outcome.ledger

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
        kept, ledger = self.budget_messages(config, system_message, messages, tools, settings)
        sent = ledger["max_tokens_sent"]
        if sent != ledger["max_tokens"]:
            settings = replace(settings, max_tokens=sent)
        origin = Origin(
            adapter=entry.adapter.name,
            provider=config.type,
            model=config.model,
            credential=credential_fingerprint(config.base_url, config.api_key),
        )
        request = RenderRequest(
            config=config, system=system_message, messages=kept, tools=tuple(tools), settings=settings, stream=True,
        )
        return PreparedCall(config=config, entry=entry, profile=profile, request=request, origin=origin, ledger=ledger)

    async def run(self, prepared: PreparedCall) -> AsyncIterator[StreamEvent]:
        adapter = prepared.entry.adapter
        wire = adapter.render(prepared.request, prepared.profile)
        decoder = adapter.decoder(prepared.request, prepared.profile)
        traced = StepAssembler(new_id("trace"), prepared.origin)
        started = time.monotonic()
        async with aclosing(adapter.send(wire, prepared.request)) as chunks:
            async for chunk in chunks:
                for event in decoder.feed(chunk):
                    traced.feed(event)
                    yield event
        for event in decoder.close():
            traced.feed(event)
            yield event
        self._record_trace(prepared, wire.body, traced, started)

    @staticmethod
    def _record_trace(prepared: PreparedCall, body: Any, traced: StepAssembler, started: float) -> None:
        message = traced.message()
        usage = traced.usage or Usage()
        body = body if isinstance(body, dict) else {}
        trace_collector.record(
            provider=prepared.config.type,
            model=prepared.config.model,
            request_system=prepared.request.system,
            request_messages=body.get("messages") if isinstance(body.get("messages"), list) else [],
            request_params={key: value for key, value in body.items() if key not in _NON_PARAM_KEYS},
            request_tools=[tool.name for tool in prepared.request.tools] or None,
            response_text=message.text(),
            response_tool_calls=[
                {"id": call.wire_id or call.call_id, "name": call.name, "arguments": call.args if call.args is not None else call.args_raw}
                for call in message.tool_calls()
            ] or None,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

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
        messages: Sequence[Message],
        *,
        system: Optional[str] = None,
        settings: Settings = Settings(),
        tools: Sequence[ToolSpec] = (),
    ) -> Completion:
        prepared = await self.prepare(target, messages, system=system, settings=settings, tools=tools)
        assembler = StepAssembler(new_id("msg"), prepared.origin)
        async with aclosing(self.run(prepared)) as events:
            async for event in events:
                assembler.feed(event)
        finish = assembler.finish
        return Completion(
            message=assembler.message(),
            usage=assembler.usage,
            finish=finish.reason if finish is not None else None,
            finish_raw=finish.raw_reason if finish is not None else None,
        )

    async def test_configuration(self, config: LLMConfig) -> Dict[str, Any]:
        try:
            response = await self.generate_response(
                "Hello, this is a test. Please respond with 'Test successful!'", config, config.system_message,
            )
            return {
                "success": True,
                "response": response.content,
                "model": response.model,
                "tokens_used": response.tokens_used,
            }
        except Exception:
            logging.error(f"[Chat] Configuration test failed for '{config.id}'", exc_info=True)
            return {"success": False, "error": "Failed to reach the configured LLM provider."}

    def _client_for(self, config: LLMConfig) -> LLMClient:
        return self.types.require(config.type).adapter

    def _budgeted(
        self,
        config: LLMConfig,
        system_message: Optional[str],
        messages: List[Dict[str, Any]],
        tool_schemas: Optional[List[Dict[str, Any]]],
        image_data: Optional[str],
        options_override: Optional[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        outcome = self.estimate_context_budget(
            config, system_message, messages, tool_schemas, image_data, options_override,
        )
        effective_override = options_override
        sent = outcome.ledger["max_tokens_sent"]
        if sent != outcome.ledger["max_tokens"]:
            effective_override = dict(options_override or {})
            effective_override["max_tokens"] = sent
        return outcome.messages, effective_override

    def _wire_config(self, config: LLMConfig) -> LLMConfig:
        if config.type != "ollama":
            return config
        capacity = self.windows.capacity(config)
        if capacity.source != "discovered":
            return config
        provider_options = dict(config.provider_options or {})
        provider_options["num_ctx"] = capacity.capacity_tokens
        return config.model_copy(update={"provider_options": provider_options})

    async def generate_with_config_id(self, prompt: str, llm_id: str, image_data: Optional[str] = None) -> LLMResponse:
        config = self.repository.get_configuration(llm_id)
        if not config:
            raise ValueError(f"LLM configuration '{llm_id}' not found")
        system_message = "" if config.disable_system_prompt else config.system_message
        return await self.generate_response(prompt=prompt, config=config, system_message=system_message, image_data=image_data)

    async def generate_response(self, prompt: str, config: LLMConfig, system_message: str, image_data: Optional[str] = None) -> LLMResponse:
        if not config.enabled:
            raise ValueError(f"LLM configuration '{config.id}' is disabled")
        await self.windows.ensure(config)
        outcome = self.estimate_context_budget(
            config, system_message, [{"role": "user", "content": prompt}], image_data=image_data,
        )
        sent = outcome.ledger["max_tokens_sent"]
        if sent != config.max_tokens:
            config = config.model_copy(update={"max_tokens": sent})
        config = self._wire_config(config)
        return await self._client_for(config).generate(prompt, config, system_message, image_data)

    async def generate_with_history(
        self,
        messages: list[Dict[str, str]],
        llm_id: str,
        image_data: Optional[str] = None,
        custom_system_message: Optional[str] = None,
        options_override: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        return await self.generate_with_tools(messages, llm_id, None, image_data, custom_system_message, options_override)

    def stream_with_history(
        self,
        messages: list[Dict[str, str]],
        llm_id: str,
        image_data: Optional[str] = None,
        custom_system_message: Optional[str] = None,
        options_override: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[dict, None]:
        return self.stream_with_tools(messages, llm_id, None, image_data, custom_system_message, options_override)

    async def _prepare(
        self,
        messages: list[Dict[str, Any]],
        llm_id: str,
        tools: Optional[List[Dict]],
        image_data: Optional[str],
        custom_system_message: Optional[str],
        options_override: Optional[Dict[str, Any]],
    ) -> tuple[LLMConfig, str, list[Dict[str, Any]], Optional[Dict[str, Any]]]:
        config = self._config(llm_id)
        await self.windows.ensure(config)
        system_message = self.resolve_system(config, custom_system_message)
        messages, options_override = self._budgeted(config, system_message, messages, tools, image_data, options_override)
        return self._wire_config(config), system_message, messages, options_override

    async def generate_with_tools(
        self,
        messages: list[Dict[str, Any]],
        llm_id: str,
        tools: List[Dict] = None,
        image_data: Optional[str] = None,
        custom_system_message: Optional[str] = None,
        options_override: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        config, system_message, messages, options_override = await self._prepare(
            messages, llm_id, tools, image_data, custom_system_message, options_override
        )
        return await self._client_for(config).generate_with_tools(
            messages, config, system_message, tools, image_data, options_override
        )

    async def stream_with_tools(
        self,
        messages: list[Dict[str, Any]],
        llm_id: str,
        tools: Optional[List[Dict]] = None,
        image_data: Optional[str] = None,
        custom_system_message: Optional[str] = None,
        options_override: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[dict, None]:
        config, system_message, messages, options_override = await self._prepare(
            messages, llm_id, tools, image_data, custom_system_message, options_override
        )
        async with aclosing(self._client_for(config).stream_with_tools(
            messages, config, system_message, tools, image_data, options_override
        )) as agen:
            async for event in agen:
                yield event

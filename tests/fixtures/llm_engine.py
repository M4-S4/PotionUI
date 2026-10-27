from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

from src.features.llm.context_windows import ContextWindows
from src.features.llm.gateway import LLMGateway
from src.features.llm.harness.adapters.ollama_chat import OllamaChatAdapter
from src.features.llm.harness.adapters.openai_chat import OpenAIChatAdapter
from src.features.llm.harness.adapters.openai_responses import OpenAIResponsesAdapter
from src.features.llm.harness.baseline import InMemoryTurnStore
from src.features.llm.harness.engine import TurnEngine
from src.features.llm.harness.retry import BackoffRetry
from src.features.llm.harness.tool_runner import ParallelToolRunner
from src.features.llm.harness.profile import Prices
from src.features.llm.harness.profiles import LayeredProfileResolver
from src.features.llm.repository import LLMConfig
from src.features.llm.tools.base import BaseTool, ToolContext
from src.features.llm.tools.executor import ToolExecutor
from src.features.llm.tools.registry import ToolRegistry
from src.features.llm.types import LLMType, LLMTypes


def scripted_config(**overrides: Any) -> LLMConfig:
    values: Dict[str, Any] = dict(
        id="cfg-scripted",
        name="Scripted",
        type="scripted",
        enabled=True,
        base_url="http://scripted.invalid",
        model="scripted-model",
        system_message="You are terse.",
        max_tokens=2000,
    )
    values.update(overrides)
    return LLMConfig(**values)


class ConfigRepository:
    def __init__(self, *configs: LLMConfig) -> None:
        self._configs = {config.id: config for config in configs}

    def get_configuration(self, config_id: str) -> Optional[LLMConfig]:
        return self._configs.get(config_id)


def scripted_gateway(adapter: Any, *configs: LLMConfig) -> LLMGateway:
    types = LLMTypes(core=(LLMType(type="scripted", adapter=adapter),))
    windows = ContextWindows(types)
    return LLMGateway(
        llm_repository=ConfigRepository(*(configs or (scripted_config(),))),
        types=types,
        profiles=LayeredProfileResolver(types.adapter_for),
        windows=windows,
    )


@dataclass
class EngineRig:
    engine: TurnEngine
    gateway: LLMGateway
    executor: ToolExecutor
    store: InMemoryTurnStore
    sleeps: List[float] = field(default_factory=list)


def scripted_engine(
    adapter: Any,
    tools: Sequence[BaseTool] = (),
    *,
    config: Optional[LLMConfig] = None,
    retry: Any = None,
    clock: Any = lambda: 0.0,
) -> EngineRig:
    gateway = scripted_gateway(adapter, config or scripted_config())
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    executor = ToolExecutor(tool_registry=registry)
    store = InMemoryTurnStore()
    sleeps: List[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    engine = TurnEngine(
        gateway=gateway,
        tools=executor,
        runner=ParallelToolRunner(executor),
        store=store,
        retry=retry or BackoffRetry(max_retries=0),
        sleep=sleep,
        clock=clock,
    )
    return EngineRig(engine=engine, gateway=gateway, executor=executor, store=store, sleeps=sleeps)


class PricedResolver:
    def __init__(self, inner: Any, prices: Prices) -> None:
        self._inner = inner
        self._prices = prices

    def resolve(self, config: Any) -> Any:
        return replace(self._inner.resolve(config), prices=self._prices)

    async def refresh(self, config: Any) -> Any:
        return self.resolve(config)


def tool_context(llm_id: str = "cfg-scripted") -> ToolContext:
    return ToolContext(user_id="user-1", llm_id=llm_id)


async def drain(events: Any) -> List[Any]:
    return [event async for event in events]


def core_gateway(llm_repository: Any, provider_registry: Any = None) -> LLMGateway:
    types = LLMTypes(
        core=(
            LLMType(type="ollama", adapter=OllamaChatAdapter()),
            LLMType(type="openai", adapter=OpenAIResponsesAdapter()),
            LLMType(type="openai-compatible", adapter=OpenAIChatAdapter()),
        ),
        provider_registry=provider_registry,
    )
    windows = ContextWindows(types)
    return LLMGateway(llm_repository, types, LayeredProfileResolver(types.adapter_for), windows)


def client_gateway(
    llm_repository: Any,
    provider_registry: Any = None,
    model_lifecycle: Any = None,
    model_roots: Any = None,
) -> LLMGateway:
    from src.features.llm.clients import NativeLLMClient, OllamaClient, OpenAIClient

    clients = {
        "ollama": OllamaClient(),
        "openai": OpenAIClient(),
        "native": NativeLLMClient(model_lifecycle, model_roots),
    }
    types = LLMTypes(
        core=tuple(LLMType(type=type_id, adapter=client) for type_id, client in clients.items()),
        provider_registry=provider_registry,
    )
    windows = ContextWindows(types)
    gateway = LLMGateway(llm_repository, types, LayeredProfileResolver(types.adapter_for), windows)
    gateway._clients = clients
    gateway._ollama, gateway._openai, gateway._native = clients["ollama"], clients["openai"], clients["native"]
    return gateway

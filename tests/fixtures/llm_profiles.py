from typing import Any, Callable, Dict, List, Mapping, Optional

from src.features.llm.harness.profile import ProfileLayer, ProfileSource
from src.features.llm.harness.profile_seed import SeedTable, default_seed
from src.features.llm.repository import LLMConfig


def make_config(**overrides: Any) -> LLMConfig:
    base: Dict[str, Any] = dict(
        id="cfg-1",
        name="Test",
        type="test",
        enabled=True,
        base_url="http://localhost:1234/",
        model="some-model",
        system_message="system",
    )
    base.update(overrides)
    return LLMConfig(**base)


class ProbeAdapter:
    def __init__(self, name: str = "probe", fields: Optional[Mapping[str, Any]] = None, error: Optional[BaseException] = None):
        self.name = name
        self.fields: Mapping[str, Any] = fields or {}
        self.error = error
        self.calls: List[str] = []

    async def discover(self, config: LLMConfig) -> ProfileLayer:
        self.calls.append(config.id)
        if self.error is not None:
            raise self.error
        return ProfileLayer(ProfileSource.DISCOVERED, dict(self.fields))


def adapter_for(adapter: ProbeAdapter) -> Callable[[LLMConfig], ProbeAdapter]:
    return lambda config: adapter


def tiny_seed(*rows: Mapping[str, Any], catalog: Optional[List[Mapping[str, Any]]] = None, **base_overrides: Any) -> SeedTable:
    base = {name: value for name, value in default_seed().base.items()}
    base.update(base_overrides)
    return SeedTable.from_mapping(
        {
            "version": 1,
            "sources": {"doc": "https://example.test/doc"},
            "base": {"fields": base},
            "rows": [dict(row, source=row.get("source", ["doc"])) for row in rows],
            "catalog": list(catalog or []),
        }
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

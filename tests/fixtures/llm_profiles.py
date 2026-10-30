import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import yaml

from src.features.llm.harness.profile import ProfileLayer, ProfileSource
from src.features.llm.harness.profile_catalog import render_seed
from src.features.llm.harness.profile_seed import BASE_PATH, SeedRoot, SeedTable, default_seed, marketplace_root
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK_PROFILES_DIR = REPO_ROOT / "content" / "plugins" / "marketplace" / "deepseek-provider" / "profiles"
MODELS_DEV_PROVIDERS = {
    "anthropic": "anthropic-messages",
    "openai": "openai-responses",
    "google": "gemini-interactions",
    "deepseek": "openai-chat",
}


def shipped_roots() -> List[SeedRoot]:
    return [marketplace_root(), SeedRoot("plugin:deepseek-provider", DEEPSEEK_PROFILES_DIR)]


@lru_cache(maxsize=1)
def shipped_seed() -> SeedTable:
    return SeedTable.scan(shipped_roots())


def shipped_mapping() -> Dict[str, Any]:
    merged: Dict[str, Any] = {"version": 1, "sources": {}, "base": yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))["base"], "rows": [], "catalog": []}
    for root in shipped_roots():
        for path in sorted(root.path.glob("*.yml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            merged["sources"].update(data["sources"])
            merged["rows"].extend(data.get("rows") or [])
            merged["catalog"].extend(data.get("catalog") or [])
    return copy.deepcopy(merged)


def combined_seed_text() -> str:
    shipped = shipped_mapping()
    head = yaml.safe_dump(
        {"version": 1, "sources": shipped["sources"], "models_dev": MODELS_DEV_PROVIDERS, "rows": shipped["rows"]},
        sort_keys=False,
    )
    return render_seed(f"{head}catalog: []\n", shipped["catalog"])


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

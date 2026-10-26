from types import SimpleNamespace

import pytest

from src.features.llm.harness.baseline import InMemoryTurnStore
from src.features.llm.harness.build import user_message
from src.features.llm.harness.messages import Origin
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.harness.turn_state import StepRecord, StepStatus, TurnSnapshot, TurnState, TurnStatus
from src.features.llm.harness.adapter import Settings
from src.features.llm.types import LLMType, LLMTypes


class Probe:
    name = "probe"

    def __init__(self, window=None):
        self.window = window

    async def discover_context_window(self, base_url, api_key, model):
        return self.window


ORIGIN = Origin(adapter="openai-chat", provider="openai", model="m1")


def step(seq, turn_id, message):
    return StepRecord(id=f"st{seq}", session_id="s1", turn_id=turn_id, seq=seq, message=message, status=StepStatus.COMPLETED)


def state(turn_id, status):
    snapshot = TurnSnapshot(
        llm_config_id="c", system="", tools=(), settings=Settings(), profile=CapabilityProfile(adapter="a", model="m"),
    )
    return TurnState(turn_id=turn_id, session_id="s1", status=status, snapshot=snapshot, first_seq=0)


def test_the_in_memory_store_appends_in_order_and_finds_the_open_turn():
    store = InMemoryTurnStore()
    store.append(step(0, "t0", user_message("a")))
    with pytest.raises(ValueError):
        store.append(step(5, "t0", user_message("b")))
    store.save_state(state("t0", TurnStatus.COMPLETED))
    assert store.open_turn("s1") is None
    store.save_state(state("t1", TurnStatus.AWAITING_DECISION))

    assert store.next_seq("s1") == 1
    assert store.open_turn("s1").turn_id == "t1"
    assert store.load_state("t0").status is TurnStatus.COMPLETED
    assert store.steps("other") == []


def test_llm_types_merge_core_and_plugin_entries():
    registry = SimpleNamespace(
        get=lambda type_id: SimpleNamespace(type_name="deepseek", adapter=Probe(), label="DeepSeek", default_base_url="https://x", source="deepseek-provider") if type_id == "deepseek" else None,
        all=lambda: [SimpleNamespace(type_name="deepseek")],
    )
    types = LLMTypes(core=(LLMType(type="scripted", adapter=Probe()),), provider_registry=registry)

    assert [entry.type for entry in types.all()] == ["scripted", "deepseek"]
    assert types.require("deepseek").source == "deepseek-provider"
    assert types.describe()[1] == {
        "type": "deepseek", "label": "DeepSeek", "adapter": "probe", "default_base_url": "https://x",
        "supports_model_listing": False, "source": "deepseek-provider",
    }
    with pytest.raises(ValueError, match="Unsupported LLM type: nope"):
        types.require("nope")


def test_llm_types_resolve_the_adapter_of_a_configuration():
    probe = Probe()
    types = LLMTypes(core=(LLMType(type="scripted", adapter=probe),))

    assert types.adapter_for(SimpleNamespace(type="scripted")) is probe
    with pytest.raises(ValueError, match="Unsupported LLM type: nope"):
        types.adapter_for(SimpleNamespace(type="nope"))

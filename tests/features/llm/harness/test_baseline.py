from types import SimpleNamespace

import pytest

from src.features.llm.context_windows import ContextWindows
from src.features.llm.harness.baseline import InMemoryTurnStore, StaticProfileResolver, turn_only_replay
from src.features.llm.harness.build import user_message
from src.features.llm.harness.messages import Message, Origin, ReasoningPart, Role, TextPart
from src.features.llm.harness.profile import CapabilityProfile, ReasoningKind, ReplayPolicy
from src.features.llm.harness.turn_state import StepRecord, StepStatus, TurnSnapshot, TurnState, TurnStatus
from src.features.llm.harness.adapter import Settings
from src.features.llm.model_listing import DiscoveredWindow
from src.features.llm.types import LLMType, LLMTypes
from tests.fixtures.llm_engine import scripted_config


class Probe:
    name = "probe"

    def __init__(self, window=None):
        self.window = window

    async def discover_context_window(self, base_url, api_key, model):
        return self.window


def resolver(window=None, type_id="scripted"):
    types = LLMTypes(core=(LLMType(type=type_id, adapter=Probe(window)),))
    windows = ContextWindows(types)
    return StaticProfileResolver(types, windows), windows


def test_the_static_profile_reads_the_configured_window_and_flags():
    profiles, _ = resolver()

    profile = profiles.resolve(scripted_config(provider_options={"context_window": 16000, "force_prompt_tools": True}, supports_vision=True))

    assert profile.adapter == "probe" and profile.model == "scripted-model"
    assert profile.context_window == 16000
    assert profile.prompt_tools is True and profile.vision is True
    assert profile.replay is ReplayPolicy.DROP_EARLIER
    assert profile.reasoning is ReasoningKind.NONE


def test_an_unknown_window_leaves_the_profile_window_empty():
    profiles, _ = resolver()

    assert profiles.resolve(scripted_config()).context_window is None


async def test_refresh_uses_the_discovered_window():
    profiles, _ = resolver(DiscoveredWindow(tokens=65536, kind="runtime"))

    profile = await profiles.refresh(scripted_config())

    assert profile.context_window == 65536


def test_native_configs_always_use_prompt_tools_and_ollama_thinking_is_a_toggle():
    native, _ = resolver(type_id="native")
    ollama, _ = resolver(type_id="ollama")

    assert native.resolve(scripted_config(type="native")).prompt_tools is True
    thinking = ollama.resolve(scripted_config(type="ollama", provider_options={"think": True}))
    assert thinking.reasoning is ReasoningKind.TOGGLE and thinking.prompt_tools is False


ORIGIN = Origin(adapter="openai-chat", provider="openai", model="m1")


def step(seq, turn_id, message):
    return StepRecord(id=f"st{seq}", session_id="s1", turn_id=turn_id, seq=seq, message=message, status=StepStatus.COMPLETED)


def reasoned(message_id, origin=ORIGIN):
    return Message(id=message_id, role=Role.ASSISTANT, origin=origin, parts=(ReasoningPart(text="why"), TextPart(text="ok")))


def test_turn_only_replay_keeps_reasoning_of_the_current_turn_and_the_same_model_only():
    steps = [
        step(0, "t0", user_message("a")),
        step(1, "t0", reasoned("old")),
        step(2, "t1", user_message("b")),
        step(3, "t1", reasoned("current")),
        step(4, "t1", reasoned("other-model", Origin(adapter="openai-chat", provider="openai", model="m2"))),
    ]

    messages, notices = turn_only_replay(steps, turn_id="t1", target=ORIGIN, profile=CapabilityProfile(adapter="a", model="m1"))

    assert notices == ()
    assert [message.reasoning() != () for message in messages] == [False, False, False, True, False]
    assert messages[1].text() == "ok"


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

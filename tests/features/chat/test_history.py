from types import SimpleNamespace

from src.features.chat.history import replayed_history, without_tool_exchanges
from src.features.llm.harness.baseline import InMemoryTurnStore, turn_only_replay
from src.features.llm.harness.messages import (
    Message,
    Origin,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.harness.turn_state import StepRecord, StepStatus
from tests.fixtures.llm_turns import state

ORIGIN = Origin(adapter="a", provider="p", model="m")
PROFILE = CapabilityProfile(adapter="a", model="m")


def asking() -> Message:
    return Message(
        id="m2",
        role=Role.ASSISTANT,
        parts=(ReasoningPart(text="hmm"), TextPart(text="Looking."), ToolCallPart(call_id="c1", name="t", args={})),
        origin=ORIGIN,
        raw={"x": 1},
    )


def answering() -> Message:
    return Message(id="m4", role=Role.ASSISTANT, parts=(TextPart(text="Found."),), origin=ORIGIN)


def calling_only() -> Message:
    return Message(id="m5", role=Role.ASSISTANT, parts=(ToolCallPart(call_id="c2", name="t", args={}),), origin=ORIGIN)


def exchange():
    return [
        Message(id="m1", role=Role.USER, parts=(TextPart(text="find"),)),
        asking(),
        Message(id="m3", role=Role.TOOL, parts=(ToolResultPart(call_id="c1", name="t", content=(TextPart(text="r"),)),)),
        calling_only(),
        answering(),
    ]


def seeded_store():
    store = InMemoryTurnStore()
    for seq, message in enumerate(exchange()):
        store.append(StepRecord(
            id=f"s{seq}", session_id="s1", turn_id="t1", seq=seq, message=message, status=StepStatus.COMPLETED,
        ))
    return store


def test_without_tools_the_exchange_keeps_only_the_words():
    kept = without_tool_exchanges(exchange())

    assert [message.id for message in kept] == ["m1", "m2", "m4"]
    assert kept[1].tool_calls() == ()
    assert kept[1].text() == "Looking."
    assert kept[1].raw is None
    assert exchange()[1].tool_calls()


def test_with_tools_the_history_is_the_stored_steps_in_order():
    history = replayed_history(seeded_store(), turn_only_replay, "s1", "t1", ORIGIN, PROFILE, tools_enabled=True)

    assert [message.id for message in history] == ["m1", "m2", "m3", "m5", "m4"]
    assert history[1].reasoning()


def test_a_later_turn_gets_the_earlier_turns_without_reasoning():
    history = replayed_history(seeded_store(), turn_only_replay, "s1", "t2", ORIGIN, PROFILE, tools_enabled=True)

    assert not any(message.reasoning() for message in history)


def test_history_without_tools_drops_the_tool_exchanges():
    history = replayed_history(seeded_store(), turn_only_replay, "s1", "t2", ORIGIN, PROFILE, tools_enabled=False)

    assert [message.id for message in history] == ["m1", "m2", "m4"]


def spying_selector(seen):
    def select(steps, *, turn_id, target, profile, **extra):
        seen.append(extra)
        return tuple(step.message for step in steps), ()

    return select


def test_a_recorded_prefix_change_reaches_the_selector():
    store = seeded_store()
    store.load_state = lambda turn_id: SimpleNamespace(prefix_changed_at=3)
    seen = []

    replayed_history(store, spying_selector(seen), "s1", "t2", ORIGIN, PROFILE, tools_enabled=True)

    assert seen == [{"prefix_changed_at": 3}]


def test_a_session_without_a_recorded_prefix_change_calls_the_selector_plainly():
    store = seeded_store()
    store.save_state(state())
    seen = []

    replayed_history(store, spying_selector(seen), "s1", "t2", ORIGIN, PROFILE, tools_enabled=True)
    replayed_history(InMemoryTurnStore(), spying_selector(seen), "s1", "t2", ORIGIN, PROFILE, tools_enabled=True)

    assert seen == [{}, {}]

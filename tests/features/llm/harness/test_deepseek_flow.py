from typing import List

from src.features.llm.harness.messages import LEGACY_ORIGIN, Origin, Role, TextPart, ToolResultPart, credential_fingerprint, message
from src.features.llm.harness.profile import ReplayPolicy
from src.features.llm.harness.profiles import LayeredProfileResolver
from src.features.llm.harness.replay import missing_reasoning, must_disable_reasoning, policy_replay
from src.features.llm.harness.turn_state import StepRecord, StepStatus
from tests.fixtures.llm_harness import assemble, text_step, tool_step
from tests.fixtures.llm_profiles import ProbeAdapter, adapter_for, make_config, shipped_seed

BASE_URL = "https://api.deepseek.com"
ORIGIN = Origin(
    adapter="openai-chat", provider="deepseek", model="deepseek-v4-pro", credential=credential_fingerprint(BASE_URL, "sk-test")
)


def deepseek_config(**overrides):
    fields = dict(type="deepseek", model="deepseek-v4-pro", base_url=BASE_URL, api_key="sk-test")
    fields.update(overrides)
    return make_config(**fields)


def resolver() -> LayeredProfileResolver:
    return LayeredProfileResolver(adapter_for(ProbeAdapter(name="openai-chat")), shipped_seed())


def step(seq: int, turn: str, msg) -> StepRecord:
    return StepRecord(id=f"s{seq}", session_id="session", turn_id=turn, seq=seq, message=msg, status=StepStatus.COMPLETED)


def tool_result(seq: int, turn: str, call_id: str) -> StepRecord:
    return step(seq, turn, message(f"tool-{seq}", Role.TOOL, ToolResultPart(call_id=call_id, name="lookup", content=(TextPart("ok"),))))


def conversation(origin: Origin) -> List[StepRecord]:
    return [
        step(1, "t1", message("u1", "user", TextPart("find it"))),
        step(2, "t1", assemble(tool_step({"name": "lookup", "args": {"q": "a"}, "id": "c1"}, reasoning="plan the lookup"), "a2", origin)),
        tool_result(3, "t1", "c1"),
        step(4, "t1", assemble(text_step("found it", reasoning="summarise"), "a4", origin)),
        step(5, "t2", message("u5", "user", TextPart("and again"))),
        step(6, "t2", assemble(tool_step({"name": "lookup", "args": {"q": "b"}, "id": "c2"}, reasoning="second lookup"), "a6", origin)),
        tool_result(7, "t2", "c2"),
    ]


def test_the_deepseek_profile_asks_for_thinking_with_reasoning_replayed_for_every_turn():
    profile = resolver().resolve(deepseek_config())

    assert profile.adapter == "openai-chat"
    assert profile.replay is ReplayPolicy.SEND_ALL
    assert profile.replay_required_with_tools is True
    assert profile.reasoning.value == "effort"
    assert profile.reasoning_toggle == "thinking.type"
    assert profile.reasoning_field == "reasoning_content"
    assert profile.interleaved_reasoning is True


def test_every_assistant_message_of_a_tool_conversation_keeps_its_reasoning_so_thinking_can_stay_on():
    profile = resolver().resolve(deepseek_config())

    messages, notices = policy_replay(conversation(ORIGIN), turn_id="t2", target=ORIGIN, profile=profile)

    assert [m.reasoning()[0].text for m in messages if m.role is Role.ASSISTANT] == [
        "plan the lookup", "summarise", "second lookup",
    ]
    assert missing_reasoning(messages) == ()
    assert must_disable_reasoning(profile, messages, tools_present=True) is False
    assert notices == ()


def test_history_from_before_reasoning_was_kept_is_reported_so_thinking_is_turned_off_for_that_call():
    legacy = [
        step(1, "t1", message("u1", "user", TextPart("find it"))),
        step(2, "t1", assemble(tool_step({"name": "lookup", "args": {"q": "a"}, "id": "c1"}), "a2", LEGACY_ORIGIN)),
        tool_result(3, "t1", "c1"),
        step(4, "t1", assemble(text_step("found it"), "a4", LEGACY_ORIGIN)),
        step(5, "t2", message("u5", "user", TextPart("and again"))),
        step(6, "t2", assemble(tool_step({"name": "lookup", "args": {"q": "b"}, "id": "c2"}, reasoning="fresh"), "a6", ORIGIN)),
        tool_result(7, "t2", "c2"),
    ]
    profile = resolver().resolve(deepseek_config())

    messages, _ = policy_replay(legacy, turn_id="t2", target=ORIGIN, profile=profile)

    assert missing_reasoning(messages) == ("a2", "a4")
    assert must_disable_reasoning(profile, messages, tools_present=True) is True
    assert must_disable_reasoning(profile, messages, tools_present=False) is False


def test_an_admin_override_to_drop_earlier_turns_surfaces_the_gap_the_same_way():
    profile = resolver().resolve(deepseek_config(profile_overrides={"replay": "drop_earlier"}))

    messages, _ = policy_replay(conversation(ORIGIN), turn_id="t2", target=ORIGIN, profile=profile)

    assert profile.sources["replay"].value == "override"
    assert missing_reasoning(messages) == ("a2", "a4")


def test_plain_text_reasoning_made_with_another_api_key_still_travels():
    other_key = Origin(adapter="openai-chat", provider="deepseek", model="deepseek-v4-pro", credential="another")
    profile = resolver().resolve(deepseek_config())

    messages, notices = policy_replay(conversation(other_key), turn_id="t2", target=ORIGIN, profile=profile)

    assert missing_reasoning(messages) == ()
    assert notices == ()


def test_forced_tool_choice_is_not_allowed_while_thinking_but_is_otherwise():
    profile = resolver().resolve(deepseek_config())

    assert {"required", "named"} <= profile.tool_choice
    assert profile.tool_choice_with_reasoning == frozenset({"auto", "none"})


def test_sampling_the_api_ignores_while_thinking_is_declared():
    assert resolver().resolve(deepseek_config()).sampling_ignored_with_reasoning == frozenset({"temperature", "top_p"})
    assert resolver().resolve(deepseek_config(model="deepseek-flash")).sampling_ignored_with_reasoning == frozenset(
        {"temperature"}
    )

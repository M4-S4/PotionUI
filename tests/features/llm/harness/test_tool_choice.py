from dataclasses import replace

import pytest

from src.features.llm.harness.adapter import Settings, ToolChoice
from src.features.llm.harness.profile import ReasoningKind
from src.features.llm.harness.tool_choice import effective_tool_choice, reasoning_active
from src.features.llm.harness.turn_state import NoticeKind
from tests.fixtures.llm_harness import scripted_profile

ALL_MODES = frozenset({"auto", "none", "required", "named"})


def profile(**changes):
    return replace(scripted_profile(), **changes)


def asking(mode: str, name=None, **settings) -> Settings:
    return Settings(tool_choice=ToolChoice(mode, name), **settings)


def test_no_request_means_no_override():
    decision = effective_tool_choice(Settings(), profile())
    assert (decision.choice, decision.drop_tools, decision.notice) == (None, False, None)


def test_a_supported_choice_passes_through_untouched():
    wanted = asking("named", "lookup", reasoning=False)
    decision = effective_tool_choice(wanted, profile(tool_choice=ALL_MODES))
    assert decision.choice is wanted.tool_choice
    assert decision.notice is None


def test_forcing_a_tool_the_profile_bans_falls_back_to_auto_with_a_notice():
    decision = effective_tool_choice(asking("required"), profile(tool_choice=frozenset({"auto"}), reasoning=ReasoningKind.NONE))
    assert decision.choice == ToolChoice("auto")
    assert decision.drop_tools is False
    assert decision.notice.kind is NoticeKind.DEGRADED_RETRY
    assert decision.notice.detail["reason"] == "tool_choice_unsupported"
    assert decision.notice.detail["requested"] == "required"


def test_a_named_choice_the_profile_bans_names_the_tool_in_the_notice():
    decision = effective_tool_choice(asking("named", "lookup"), profile(tool_choice=frozenset({"auto"})))
    assert decision.choice == ToolChoice("auto")
    assert "lookup" in decision.notice.message
    assert decision.notice.detail["name"] == "lookup"


def test_reasoning_can_ban_what_the_profile_otherwise_allows():
    capable = profile(
        reasoning=ReasoningKind.TOGGLE,
        tool_choice=ALL_MODES,
        tool_choice_with_reasoning=frozenset({"auto", "none"}),
    )
    thinking = effective_tool_choice(asking("required", reasoning=True), capable)
    quiet = effective_tool_choice(asking("required", reasoning=False), capable)
    assert thinking.choice == ToolChoice("auto")
    assert "while it is reasoning" in thinking.notice.message
    assert thinking.notice.detail["reasoning"] is True
    assert quiet.choice == ToolChoice("required")
    assert quiet.notice is None


def test_an_always_on_reasoner_is_treated_as_reasoning_whatever_the_settings_say():
    always = profile(reasoning=ReasoningKind.ALWAYS_ON, tool_choice=ALL_MODES, tool_choice_with_reasoning=frozenset({"auto"}))
    assert effective_tool_choice(asking("required", reasoning=False), always).choice == ToolChoice("auto")


def test_unspecified_reasoning_on_a_reasoning_model_counts_as_reasoning():
    toggle = profile(reasoning=ReasoningKind.TOGGLE)
    assert reasoning_active(Settings(), toggle) is True
    assert reasoning_active(Settings(reasoning=True), toggle) is True
    assert reasoning_active(Settings(reasoning="high"), toggle) is True


@pytest.mark.parametrize("off", [False, "off", "none", "Disabled", "false"])
def test_reasoning_turned_off_is_not_reasoning(off):
    assert reasoning_active(Settings(reasoning=off), profile(reasoning=ReasoningKind.TOGGLE)) is False


def test_a_model_without_reasoning_is_never_reasoning():
    assert reasoning_active(Settings(reasoning=True), profile(reasoning=ReasoningKind.NONE)) is False


def test_a_wire_with_no_tool_choice_support_omits_auto_instead_of_sending_it():
    decision = effective_tool_choice(asking("required"), profile(tool_choice=frozenset()))
    assert decision.choice is None
    assert decision.notice is not None
    assert effective_tool_choice(asking("auto"), profile(tool_choice=frozenset())).choice is None


def test_turning_tools_off_on_a_wire_that_cannot_says_to_drop_the_tools():
    decision = effective_tool_choice(asking("none"), profile(tool_choice=frozenset({"auto"})))
    assert decision.choice is None
    assert decision.drop_tools is True
    assert decision.notice is None
    assert effective_tool_choice(asking("none"), profile(tool_choice=ALL_MODES)).drop_tools is False


def test_a_named_choice_without_a_name_is_not_forwarded():
    decision = effective_tool_choice(asking("named", None), profile(tool_choice=ALL_MODES))
    assert decision.choice == ToolChoice("auto")
    assert decision.notice is not None


def test_an_unknown_mode_falls_back_to_auto():
    decision = effective_tool_choice(asking("any"), profile(tool_choice=ALL_MODES))
    assert decision.choice == ToolChoice("auto")
    assert decision.notice.detail["requested"] == "any"

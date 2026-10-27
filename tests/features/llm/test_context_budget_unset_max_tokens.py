import math

from src.features.llm import context_budget
from src.features.llm.context_budget import CapacityInfo, ContextProfile
from src.features.llm.harness.build import user_message


def test_an_unset_output_cap_reserves_the_standard_share_of_the_window():
    window = 100_000
    standard = max(context_budget.MIN_RESERVE_TOKENS, math.ceil(context_budget.RESERVE_WINDOW_FRACTION * window))

    unset = context_budget.resolve_reserve(max_tokens=None, window_tokens=window, thinking_enabled=False)
    capped = context_budget.resolve_reserve(max_tokens=10, window_tokens=window, thinking_enabled=False)

    assert unset.base_reserve_tokens == standard
    assert capped.base_reserve_tokens == 10


def test_an_unset_output_cap_stays_unset_whatever_the_window():
    for source in ("override", "discovered", "seed", "unknown"):
        assert context_budget.resolve_sent_max_tokens(
            max_tokens=None, reserve_tokens=500, capacity_tokens=8000, capacity_source=source, estimated_prompt_tokens=7900,
        ) is None


def test_a_set_output_cap_is_still_trimmed_to_what_is_left_of_the_window():
    assert context_budget.resolve_sent_max_tokens(
        max_tokens=4000, reserve_tokens=500, capacity_tokens=8000, capacity_source="override", estimated_prompt_tokens=7000,
    ) == 1000


def test_the_ledger_of_a_call_without_an_output_cap_reports_none():
    profile = ContextProfile(capacity=CapacityInfo(8000, "override"), reserve_tokens=500, max_tokens=None)

    outcome = context_budget.budget_with_profile(
        profile, system_message="s", messages=[user_message("hi")],
    )

    assert outcome.ledger["max_tokens"] is None and outcome.ledger["max_tokens_sent"] is None

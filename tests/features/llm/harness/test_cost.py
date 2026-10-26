import pytest

from src.features.llm.harness.cost import OUTPUT_EXCLUDES_REASONING, priced, step_cost, total_usage
from src.features.llm.harness.events import Usage
from src.features.llm.harness.profile import CapabilityProfile, Prices


def profile(adapter: str, prices: Prices | None) -> CapabilityProfile:
    return CapabilityProfile(adapter=adapter, model="m", prices=prices)


ANTHROPIC = Prices(input=3, output=15, cached_input=0.3, cache_write=3.75)
OPENAI = Prices(input=1.25, output=10, cached_input=0.125)
GEMINI = Prices(input=2, output=12, cached_input=0.2)


def test_anthropic_input_includes_cache_reads_and_writes_each_at_its_own_rate():
    usage = Usage(input_tokens=4600, output_tokens=200, cached_input_tokens=4000, cache_write_tokens=500)

    assert step_cost(usage, ANTHROPIC) == 0.006375


def test_openai_cached_tokens_are_a_subset_of_the_prompt_and_reasoning_is_inside_the_output():
    usage = Usage(input_tokens=5000, output_tokens=300, reasoning_tokens=120, cached_input_tokens=4000)

    assert step_cost(usage, OPENAI) == 0.00475


def test_a_reasoning_price_splits_the_reasoning_tokens_out_of_the_output():
    usage = Usage(input_tokens=5000, output_tokens=300, reasoning_tokens=120, cached_input_tokens=4000)
    prices = Prices(input=1.25, output=10, cached_input=0.125, reasoning=12)

    assert step_cost(usage, prices) == 0.00499


def test_gemini_output_excludes_thought_tokens_so_they_are_added_at_the_output_rate():
    usage = Usage(input_tokens=3000, output_tokens=200, reasoning_tokens=800, cached_input_tokens=2000)

    assert step_cost(usage, GEMINI, output_includes_reasoning=False) == 0.0144
    assert step_cost(usage, GEMINI, output_includes_reasoning=True) == 0.0048


def test_priced_picks_the_gemini_shape_from_the_adapter_name():
    usage = Usage(input_tokens=3000, output_tokens=200, reasoning_tokens=800, cached_input_tokens=2000)

    assert "gemini-interactions" in OUTPUT_EXCLUDES_REASONING
    assert priced(usage, profile("gemini-interactions", GEMINI)).cost == 0.0144
    assert priced(usage, profile("openai-responses", GEMINI)).cost == 0.0048


def test_a_deepseek_hit_miss_split_arrives_as_cached_tokens_and_prices_like_openai():
    usage = Usage(input_tokens=2000, output_tokens=150, reasoning_tokens=100, cached_input_tokens=1500)
    prices = Prices(input=0.28, output=0.42, cached_input=0.028)

    assert step_cost(usage, prices) == 0.000245


def test_a_cost_reported_by_the_provider_wins_over_the_computed_one():
    usage = Usage(input_tokens=1000, output_tokens=100, cost=0.00123)

    assert step_cost(usage, ANTHROPIC) == 0.00123
    assert priced(usage, profile("openrouter", ANTHROPIC)) is usage


def test_local_servers_without_prices_have_no_cost():
    usage = Usage(input_tokens=900, output_tokens=40, total_tokens=940, cached_input_tokens=300)

    assert step_cost(usage, None) is None
    assert priced(usage, profile("ollama-chat", None)) is usage
    assert priced(usage, profile("native", None)).cost is None


def test_priced_fills_the_cost_and_keeps_every_other_field():
    usage = Usage(input_tokens=1000, output_tokens=100, total_tokens=1100, raw={"k": 1})

    result = priced(usage, profile("openai-chat", OPENAI))

    assert result.cost == 0.00225
    assert (result.input_tokens, result.output_tokens, result.total_tokens, result.raw) == (1000, 100, 1100, {"k": 1})


def test_priced_passes_a_missing_usage_through():
    assert priced(None, profile("openai-chat", OPENAI)) is None


def test_a_cached_price_that_is_missing_falls_back_to_the_input_price():
    usage = Usage(input_tokens=1000, output_tokens=0, cached_input_tokens=600, cache_write_tokens=100)
    prices = Prices(input=2, output=8)

    assert step_cost(usage, prices) == 0.002


def test_a_price_is_only_needed_for_the_tokens_that_exist():
    assert step_cost(Usage(input_tokens=1000, output_tokens=0), Prices(input=2)) == 0.002
    assert step_cost(Usage(input_tokens=0, output_tokens=100), Prices(output=8)) == 0.0008


@pytest.mark.parametrize(
    "usage",
    [
        Usage(input_tokens=1000, output_tokens=100),
        Usage(input_tokens=1000, output_tokens=100, reasoning_tokens=10, cached_input_tokens=5),
    ],
)
def test_a_token_kind_with_no_price_makes_the_cost_unknown(usage):
    assert step_cost(usage, Prices(input=1)) is None


@pytest.mark.parametrize(
    "usage",
    [Usage(input_tokens=1000), Usage(output_tokens=100), Usage(), Usage(cached_input_tokens=5, cache_write_tokens=5)],
)
def test_a_step_that_reports_only_part_of_its_tokens_has_no_cost(usage):
    assert step_cost(usage, ANTHROPIC) is None


def test_inconsistent_counts_are_clamped_instead_of_going_negative():
    over_cached = Usage(input_tokens=100, output_tokens=0, cached_input_tokens=500)
    over_reasoned = Usage(input_tokens=0, output_tokens=50, reasoning_tokens=400)

    assert step_cost(over_cached, Prices(input=10, output=1, cached_input=1)) == 0.0001
    assert step_cost(over_reasoned, Prices(input=1, output=10, reasoning=20)) == 0.001


def test_a_free_model_costs_zero_not_unknown():
    assert step_cost(Usage(input_tokens=500, output_tokens=50), Prices(input=0, output=0)) == 0.0


def test_total_usage_adds_every_counter_and_the_cost_across_steps():
    first = Usage(input_tokens=1000, output_tokens=100, total_tokens=1100, cached_input_tokens=800, cost=0.001)
    second = Usage(input_tokens=1200, output_tokens=50, total_tokens=1250, reasoning_tokens=20, cache_write_tokens=10, cost=0.0007)

    assert total_usage([first, second]) == Usage(
        input_tokens=2200,
        output_tokens=150,
        total_tokens=2350,
        reasoning_tokens=20,
        cached_input_tokens=800,
        cache_write_tokens=10,
        cost=0.0017,
    )


def test_total_usage_skips_steps_without_usage_and_has_none_when_nothing_was_reported():
    only = Usage(input_tokens=10, output_tokens=5)

    assert total_usage([None, only, None]) == only
    assert total_usage([None, None]) is None
    assert total_usage([]) is None


def test_total_usage_keeps_a_counter_unknown_when_no_step_reported_it():
    total = total_usage([Usage(input_tokens=10), Usage(input_tokens=5, output_tokens=2)])

    assert total.input_tokens == 15 and total.output_tokens == 2
    assert total.cost is None and total.reasoning_tokens is None

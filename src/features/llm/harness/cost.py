from dataclasses import replace
from typing import Iterable, Optional, Tuple

from src.features.llm.harness.events import Usage
from src.features.llm.harness.profile import CapabilityProfile, Prices

TOKENS_PER_PRICE = 1_000_000
COST_DIGITS = 12

_SUMMED = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "cost",
)


def step_cost(usage: Usage, prices: Optional[Prices]) -> Optional[float]:
    if usage.cost is not None:
        return usage.cost
    if prices is None or usage.input_tokens is None or usage.output_tokens is None:
        return None
    cached = min(usage.cached_input_tokens or 0, usage.input_tokens)
    written = min(usage.cache_write_tokens or 0, usage.input_tokens - cached)
    fresh = usage.input_tokens - cached - written
    reasoning = min(usage.reasoning_tokens or 0, usage.output_tokens)
    answer = usage.output_tokens - reasoning
    lines: Tuple[Tuple[int, Optional[float]], ...] = (
        (fresh, prices.input),
        (cached, prices.cached_input if prices.cached_input is not None else prices.input),
        (written, prices.cache_write if prices.cache_write is not None else prices.input),
        (answer, prices.output),
        (reasoning, prices.reasoning if prices.reasoning is not None else prices.output),
    )
    total = 0.0
    for tokens, price in lines:
        if tokens <= 0:
            continue
        if price is None:
            return None
        total += tokens * price
    return round(total / TOKENS_PER_PRICE, COST_DIGITS)


def priced(usage: Optional[Usage], profile: CapabilityProfile) -> Optional[Usage]:
    if usage is None or usage.cost is not None:
        return usage
    cost = step_cost(usage, profile.prices)
    return usage if cost is None else replace(usage, cost=cost)


def total_usage(usages: Iterable[Optional[Usage]]) -> Optional[Usage]:
    present = [usage for usage in usages if usage is not None]
    if not present:
        return None
    totals = {name: _sum(getattr(usage, name) for usage in present) for name in _SUMMED}
    if totals["cost"] is not None:
        totals["cost"] = round(totals["cost"], COST_DIGITS)
    return Usage(**totals)


def _sum(values: Iterable[Optional[float]]) -> Optional[float]:
    known = [value for value in values if value is not None]
    return sum(known) if known else None

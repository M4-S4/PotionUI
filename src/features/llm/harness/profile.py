from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from typing import Any, FrozenSet, Mapping, Optional, Sequence, Tuple

from src.features.llm.harness.codec import field_from_json


class ReplayPolicy(StrEnum):
    SEND_ALL = "send_all"
    DROP_EARLIER = "drop_earlier"
    PRESERVE = "preserve"
    NEVER = "never"


class ReasoningKind(StrEnum):
    NONE = "none"
    TOGGLE = "toggle"
    EFFORT = "effort"
    BUDGET = "budget"
    ALWAYS_ON = "always_on"


class StructuredOutputMode(StrEnum):
    NATIVE = "native"
    TOOL = "tool"
    PROMPTED = "prompted"


class CacheMode(StrEnum):
    NONE = "none"
    AUTOMATIC = "automatic"
    EXPLICIT = "explicit"
    LOCAL_PREFIX = "local_prefix"


class ToolIdRule(StrEnum):
    PASSTHROUGH = "passthrough"
    MINT = "mint"
    NINE_DIGITS = "nine_digits"


class ProfileSource(StrEnum):
    SEED = "seed"
    DISCOVERED = "discovered"
    OVERRIDE = "override"


@dataclass(frozen=True)
class Prices:
    input: Optional[float] = None
    output: Optional[float] = None
    cached_input: Optional[float] = None
    cache_write: Optional[float] = None
    reasoning: Optional[float] = None


@dataclass(frozen=True)
class Verification:
    suite: str
    passed_at: str
    runtime: Optional[str] = None
    gpu_tier: Optional[str] = None


@dataclass(frozen=True)
class ProfileLayer:
    source: ProfileSource
    fields: Mapping[str, Any]


@dataclass(frozen=True)
class CapabilityProfile:
    adapter: str
    model: str
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    tools: bool = True
    prompt_tools: bool = False
    parallel_tools: bool = False
    tool_choice: FrozenSet[str] = frozenset({"auto"})
    tool_choice_with_reasoning: FrozenSet[str] = frozenset({"auto"})
    strict_tools: bool = False
    tool_id_rule: ToolIdRule = ToolIdRule.PASSTHROUGH
    structured_output: StructuredOutputMode = StructuredOutputMode.PROMPTED
    reasoning: ReasoningKind = ReasoningKind.NONE
    reasoning_levels: Tuple[str, ...] = ()
    reasoning_toggle: Optional[str] = None
    reasoning_field: Optional[str] = None
    server: Optional[str] = None
    replay: ReplayPolicy = ReplayPolicy.SEND_ALL
    replay_required_with_tools: bool = False
    reasoning_prefix_bound: bool = False
    interleaved_reasoning: bool = False
    vision: bool = False
    max_images: Optional[int] = 1
    tool_result_media: bool = False
    sampling_rejected: FrozenSet[str] = frozenset()
    sampling_ignored_with_reasoning: FrozenSet[str] = frozenset()
    recommended_sampling: Mapping[str, float] = field(default_factory=dict)
    cache: CacheMode = CacheMode.NONE
    cache_min_tokens: Optional[int] = None
    prices: Optional[Prices] = None
    verified: Optional[Verification] = None
    sources: Mapping[str, ProfileSource] = field(default_factory=dict)

    def layered(self, layers: Sequence[ProfileLayer]) -> "CapabilityProfile":
        updates: dict = {}
        sources = dict(self.sources)
        for layer in layers:
            for name, value in layer.fields.items():
                if name not in LAYERED_FIELDS:
                    raise ValueError(f"unknown profile field {name!r} in the {layer.source.value} layer")
                updates[name] = field_from_json(CapabilityProfile, name, value)
                sources[name] = ProfileSource(layer.source)
        return replace(self, **updates, sources=sources)


LAYERED_FIELDS = frozenset(item.name for item in fields(CapabilityProfile)) - {"adapter", "model", "sources"}

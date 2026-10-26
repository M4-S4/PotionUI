import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.features.llm.harness import codec
from src.features.llm.harness.profile import (
    CacheMode,
    CapabilityProfile,
    ReasoningKind,
    ReplayPolicy,
    StructuredOutputMode,
    ToolIdRule,
)

TOOL_CHOICE_MODES = ("auto", "none", "required", "named")
SAMPLING_PARAMETERS = ("temperature", "top_p", "top_k", "min_p")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    label: str
    group: str
    options: Tuple[str, ...] = ()
    minimum: Optional[int] = None
    nullable: bool = False
    readonly: bool = False

    def to_json(self) -> Dict[str, Any]:
        encoded: Dict[str, Any] = {"name": self.name, "kind": self.kind, "label": self.label, "group": self.group}
        if self.options:
            encoded["options"] = list(self.options)
        if self.minimum is not None:
            encoded["min"] = self.minimum
        if self.nullable:
            encoded["nullable"] = True
        if self.readonly:
            encoded["readonly"] = True
        return encoded


def _values(enum_type: Any) -> Tuple[str, ...]:
    return tuple(member.value for member in enum_type)


PROFILE_FIELDS: Tuple[FieldSpec, ...] = (
    FieldSpec("context_window", "int", "Context window", "context", minimum=1024, nullable=True),
    FieldSpec("max_output_tokens", "int", "Max output tokens", "context", minimum=1, nullable=True),
    FieldSpec("tools", "bool", "Tool calling", "tools"),
    FieldSpec("prompt_tools", "bool", "Tools rendered into the prompt", "tools"),
    FieldSpec("parallel_tools", "bool", "Parallel tool calls", "tools"),
    FieldSpec("tool_choice", "set", "Tool choice modes", "tools", options=TOOL_CHOICE_MODES),
    FieldSpec("tool_choice_with_reasoning", "set", "Tool choice modes while reasoning", "tools", options=TOOL_CHOICE_MODES),
    FieldSpec("strict_tools", "bool", "Strict tool schemas", "tools"),
    FieldSpec("tool_id_rule", "enum", "Tool call ids", "tools", options=_values(ToolIdRule)),
    FieldSpec("structured_output", "enum", "Structured output", "tools", options=_values(StructuredOutputMode)),
    FieldSpec("reasoning", "enum", "Reasoning control", "reasoning", options=_values(ReasoningKind)),
    FieldSpec("reasoning_levels", "list", "Reasoning levels", "reasoning"),
    FieldSpec("reasoning_toggle", "text", "Reasoning switch", "reasoning", nullable=True),
    FieldSpec("reasoning_field", "text", "Reasoning field", "reasoning", nullable=True),
    FieldSpec("server", "text", "Server software", "context", nullable=True),
    FieldSpec("replay", "enum", "Reasoning replay", "reasoning", options=_values(ReplayPolicy)),
    FieldSpec("replay_required_with_tools", "bool", "Reasoning must be replayed when tools are used", "reasoning"),
    FieldSpec("reasoning_prefix_bound", "bool", "Reasoning breaks when earlier messages change", "reasoning"),
    FieldSpec("interleaved_reasoning", "bool", "Reasoning between tool calls", "reasoning"),
    FieldSpec("vision", "bool", "Image input", "media"),
    FieldSpec("max_images", "int", "Images per request", "media", minimum=0, nullable=True),
    FieldSpec("tool_result_media", "bool", "Media in tool results", "media"),
    FieldSpec("sampling_rejected", "set", "Rejected sampling parameters", "sampling", options=SAMPLING_PARAMETERS),
    FieldSpec(
        "sampling_ignored_with_reasoning", "set", "Ignored sampling parameters while reasoning", "sampling",
        options=SAMPLING_PARAMETERS,
    ),
    FieldSpec("recommended_sampling", "numbers", "Recommended sampling", "sampling", options=SAMPLING_PARAMETERS),
    FieldSpec("cache", "enum", "Prompt caching", "cache", options=_values(CacheMode)),
    FieldSpec("cache_min_tokens", "int", "Minimum cacheable prefix", "cache", minimum=0, nullable=True),
    FieldSpec("prices", "prices", "Prices per million tokens", "cost", nullable=True),
    FieldSpec("verified", "verification", "Verified", "verification", nullable=True, readonly=True),
)

_SPECS: Mapping[str, FieldSpec] = {spec.name: spec for spec in PROFILE_FIELDS}


def schema_json() -> Dict[str, Any]:
    return {"fields": [spec.to_json() for spec in PROFILE_FIELDS]}


def spec_for(name: str) -> Optional[FieldSpec]:
    return _SPECS.get(name)


def normalize_value(name: str, value: Any) -> Any:
    spec = _SPECS.get(name)
    if spec is None or spec.readonly:
        raise ValueError(f"{name!r} is not an editable profile field")
    try:
        typed = codec.field_from_json(CapabilityProfile, name, value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: {exc}") from exc
    if spec.kind == "set":
        unknown = sorted(set(typed) - set(spec.options))
        if unknown:
            raise ValueError(f"{name}: unknown values {unknown}, expected some of {list(spec.options)}")
    if spec.kind == "list" and not all(item.strip() for item in typed):
        raise ValueError(f"{name}: levels cannot be blank")
    if spec.kind == "text" and not typed.strip():
        raise ValueError(f"{name}: cannot be blank")
    if spec.kind == "numbers":
        _check_sampling_values(name, typed, spec.options)
    if spec.kind == "prices" and any(price is not None and price < 0 for price in asdict(typed).values()):
        raise ValueError(f"{name}: prices cannot be negative")
    if spec.minimum is not None and isinstance(typed, int) and typed < spec.minimum:
        raise ValueError(f"{name}: must be at least {spec.minimum}")
    return codec.to_json(typed)


def _check_sampling_values(name: str, values: Mapping[str, float], allowed: Tuple[str, ...]) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"{name}: unknown parameters {unknown}, expected some of {list(allowed)}")
    for key, value in values.items():
        upper = 1.0 if key in ("top_p", "min_p") else math.inf
        if not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f"{name}: {key} must be between 0 and {upper if upper != math.inf else 'infinity'}")


def validate_overrides(patch: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(patch, Mapping):
        raise ValueError("overrides must be an object")
    normalized: Dict[str, Any] = {}
    for name, value in patch.items():
        if value is None:
            if name not in _SPECS or _SPECS[name].readonly:
                raise ValueError(f"{name!r} is not an editable profile field")
            normalized[name] = None
        else:
            normalized[name] = normalize_value(name, value)
    return normalized


def merge_overrides(existing: Optional[Mapping[str, Any]], patch: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(existing or {})
    for name, value in validate_overrides(patch).items():
        if value is None:
            merged.pop(name, None)
        else:
            merged[name] = value
    return merged


def clean_overrides(stored: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    cleaned: Dict[str, Any] = {}
    problems: List[str] = []
    for name, value in (stored or {}).items():
        if value is None:
            continue
        try:
            cleaned[name] = normalize_value(name, value)
        except ValueError as exc:
            problems.append(str(exc))
    return cleaned, problems

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from src.features.llm.harness.adapter import RenderRequest, ToolSpec, WireRequest
from src.features.llm.harness.codec import dumps

SYSTEM_KEYS = ("system", "system_instruction", "instructions")
MESSAGE_KEYS = ("messages", "input", "contents", "steps")
STRUCTURAL_KEYS = frozenset({"tools", *SYSTEM_KEYS, *MESSAGE_KEYS})
VOLATILE_PARAMS = frozenset({"max_tokens", "max_completion_tokens", "max_output_tokens"})
TERMINATOR = 1


@dataclass(frozen=True)
class Segment:
    section: str
    data: bytes


@dataclass(frozen=True)
class Divergence:
    offset: int
    section: str


def append_only(previous: Sequence[ToolSpec], current: Sequence[ToolSpec]) -> Tuple[ToolSpec, ...]:
    known = {spec.name for spec in previous}
    return (*previous, *(spec for spec in current if spec.name not in known))


def request_segments(request: RenderRequest) -> Tuple[Segment, ...]:
    params = {"reasoning": request.settings.reasoning, "tool_choice": request.settings.tool_choice}
    segments = [Segment("params", dumps(params).encode("ascii"))]
    segments.extend(Segment(f"tools[{index}]", dumps(spec).encode("ascii")) for index, spec in enumerate(request.tools))
    segments.append(Segment("system", dumps(request.system).encode("ascii")))
    for index, message in enumerate(request.messages):
        entry = {"role": message.role, "parts": message.parts, "raw": message.raw}
        segments.append(Segment(f"messages[{index}]", dumps(entry).encode("ascii")))
    return tuple(segments)


def wire_segments(wire: WireRequest) -> Tuple[Segment, ...]:
    body: Mapping[str, Any] = wire.body if isinstance(wire.body, Mapping) else {}
    params = {key: value for key, value in body.items() if key not in STRUCTURAL_KEYS and key not in VOLATILE_PARAMS}
    segments = [Segment("params", _canonical(params))]
    segments.extend(_items("tools", body.get("tools")))
    for key in SYSTEM_KEYS:
        if key in body:
            segments.append(Segment(key, _canonical(body[key])))
    for key in MESSAGE_KEYS:
        if key in body:
            segments.extend(_items(key, body[key]))
    return tuple(segments)


def first_divergence(previous: Sequence[Segment], current: Sequence[Segment]) -> Optional[Divergence]:
    offset = 0
    for position, old in enumerate(previous):
        new = current[position].data if position < len(current) else b""
        if position < len(current) and new == old.data:
            offset += len(old.data) + TERMINATOR
            continue
        return Divergence(offset=offset + _shared_prefix(old.data, new), section=old.section)
    return None


def _items(section: str, value: Any) -> Tuple[Segment, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(Segment(f"{section}[{index}]", _canonical(item)) for index, item in enumerate(value))
    return (Segment(section, _canonical(value)),)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("ascii")


def _shared_prefix(left: bytes, right: bytes) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))

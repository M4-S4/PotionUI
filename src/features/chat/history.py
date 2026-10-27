from dataclasses import replace
from typing import List, Sequence

from src.features.llm.harness.messages import Message, Origin, Role, ToolCallPart
from src.features.llm.harness.ports import ReplaySelector, TurnStore
from src.features.llm.harness.profile import CapabilityProfile


def without_tool_exchanges(messages: Sequence[Message]) -> List[Message]:
    kept: List[Message] = []
    for message in messages:
        if message.role is Role.TOOL:
            continue
        if message.role is Role.ASSISTANT and message.tool_calls():
            parts = tuple(part for part in message.parts if not isinstance(part, ToolCallPart))
            if not parts:
                continue
            message = replace(message, parts=parts, raw=None)
        kept.append(message)
    return kept


def replayed_history(
    store: TurnStore,
    selector: ReplaySelector,
    session_id: str,
    turn_id: str,
    target: Origin,
    profile: CapabilityProfile,
    *,
    tools_enabled: bool,
) -> List[Message]:
    steps = store.steps(session_id)
    latest = store.load_state(steps[-1].turn_id) if steps else None
    changed_at = getattr(latest, "prefix_changed_at", None)
    extra = {} if changed_at is None else {"prefix_changed_at": changed_at}
    messages, _notices = selector(steps, turn_id=turn_id, target=target, profile=profile, **extra)
    return list(messages) if tools_enabled else without_tool_exchanges(messages)

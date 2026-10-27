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
    latest = store.latest_turn(session_id)
    messages, _notices = selector(
        store.steps(session_id), turn_id=turn_id, target=target, profile=profile,
        prefix_changed_at=latest.prefix_changed_at if latest is not None else None,
    )
    return list(messages) if tools_enabled else without_tool_exchanges(messages)

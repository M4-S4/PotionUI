from enum import StrEnum
from typing import Any, List, Mapping, Optional, Sequence

from src.features.llm.harness.turn_state import Decision, DecisionKind, TurnState, TurnStatus

MAX_REASON_CHARS = 2000


class DecisionRefusal(StrEnum):
    NOT_AWAITING = "not_awaiting"
    INDEX_OUT_OF_RANGE = "index_out_of_range"
    NOT_PENDING = "not_pending"
    NOT_AN_APPROVAL = "not_an_approval"


class DecisionRefused(Exception):
    def __init__(self, refusal: DecisionRefusal, message: str) -> None:
        super().__init__(message)
        self.refusal = refusal


def execution_call_ids(metadata: Optional[Mapping[str, Any]]) -> List[Optional[str]]:
    executions = (metadata or {}).get("tool_executions")
    if not isinstance(executions, list):
        return []
    return [item.get("tool_call_id") if isinstance(item, Mapping) else None for item in executions]


def _trimmed(reason: Optional[str]) -> Optional[str]:
    return (reason or "")[:MAX_REASON_CHARS] or None


def decision_for(
    state: Optional[TurnState],
    call_ids: Sequence[Optional[str]],
    tool_index: int,
    approved: bool,
    reason: Optional[str] = None,
) -> Decision:
    if state is None or state.status is not TurnStatus.AWAITING_DECISION:
        raise DecisionRefused(DecisionRefusal.NOT_AWAITING, "This turn is not waiting for a decision")
    if tool_index < 0 or tool_index >= len(call_ids):
        raise DecisionRefused(
            DecisionRefusal.INDEX_OUT_OF_RANGE,
            f"Tool index {tool_index} out of range (message has {len(call_ids)} tool executions)",
        )
    call_id = call_ids[tool_index]
    pending = next((call for call in state.pending if call.call_id == call_id), None) if call_id else None
    if pending is None:
        raise DecisionRefused(DecisionRefusal.NOT_PENDING, f"Tool execution {tool_index} is not waiting for a decision")
    if pending.decision is not DecisionKind.APPROVAL:
        raise DecisionRefused(DecisionRefusal.NOT_AN_APPROVAL, f"Tool execution {tool_index} is not an approval")
    return Decision(call_id=pending.call_id, approved=approved, reason=_trimmed(reason))


def rejections_for_reply(state: Optional[TurnState], content: str) -> List[Decision]:
    if state is None or state.status is not TurnStatus.AWAITING_DECISION:
        return []
    return [
        Decision(call_id=call.call_id, approved=False, reason=_trimmed(content))
        for call in state.pending
        if call.decision is DecisionKind.APPROVAL
    ]

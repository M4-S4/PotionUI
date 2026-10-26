from typing import Optional, Sequence

from src.features.llm.harness.messages import Message, Role, TextPart, ToolResultPart
from src.features.llm.harness.turn_state import StepRecord, StepStatus, TurnState

CANCELLED_REASON = "Cancelled before this tool finished. It may or may not have run."
FAILED_REASON = "The turn failed before this tool finished. It may or may not have run."
INTERRUPTED_REASON = "The server stopped before this tool finished. It may or may not have run."


def interrupted_tool_step(
    steps: Sequence[StepRecord],
    state: TurnState,
    *,
    step_id: str,
    seq: int,
    status: StepStatus,
    reason: str,
) -> Optional[StepRecord]:
    turn_steps = [step for step in steps if step.turn_id == state.turn_id]
    if not turn_steps:
        return None
    last = max(turn_steps, key=lambda step: step.seq)
    if last.message.role is not Role.ASSISTANT:
        return None
    calls = last.message.tool_calls()
    if not calls:
        return None
    gathered = {result.call_id: result for result in state.gathered}
    results = tuple(
        gathered.get(call.call_id)
        or ToolResultPart(call_id=call.call_id, name=call.name, content=(TextPart(text=reason),), is_error=True)
        for call in calls
    )
    return StepRecord(
        id=step_id,
        session_id=state.session_id,
        turn_id=state.turn_id,
        seq=seq,
        message=Message(id=step_id, role=Role.TOOL, parts=results),
        status=status,
    )

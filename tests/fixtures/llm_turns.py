from src.features.llm.harness.adapter import Settings, ToolSpec
from src.features.llm.harness.events import FinishReason
from src.features.llm.harness.messages import TextPart, ToolCallPart, ToolResultPart, message
from src.features.llm.harness.turn_state import StepRecord, StepStatus, TurnSnapshot, TurnState, TurnStatus
from tests.fixtures.llm_harness import SCRIPTED_ORIGIN, scripted_profile


def snapshot() -> TurnSnapshot:
    return TurnSnapshot(
        llm_config_id="cfg",
        system="You are terse.",
        tools=(ToolSpec(name="search", description="find", parameters={"type": "object", "properties": {}}),),
        settings=Settings(temperature=0.2),
        profile=scripted_profile(),
        mode="assistant",
    )


def state(turn_id="t1", session_id="s1", status=TurnStatus.RUNNING, first_seq=0, **fields) -> TurnState:
    return TurnState(turn_id=turn_id, session_id=session_id, status=status, snapshot=snapshot(), first_seq=first_seq, **fields)


def user_step(seq, turn_id="t1", session_id="s1", text="find cats") -> StepRecord:
    return StepRecord(
        id=f"{session_id}.u{seq}", session_id=session_id, turn_id=turn_id, seq=seq,
        message=message(f"mu{seq}", "user", TextPart(text=text)), status=StepStatus.COMPLETED,
        created_at=f"2026-10-26T10:00:{seq:02d}+00:00",
    )


def ask_step(seq, calls, turn_id="t1", session_id="s1", text="", status=StepStatus.COMPLETED, finish=FinishReason.TOOL_CALLS, **fields) -> StepRecord:
    parts = ([TextPart(text=text)] if text else []) + [
        ToolCallPart(call_id=call_id, name=name, args=args, args_raw=None, wire_id=call_id) for call_id, name, args in calls
    ]
    return StepRecord(
        id=f"{session_id}.a{seq}", session_id=session_id, turn_id=turn_id, seq=seq,
        message=message(f"ma{seq}", "assistant", *parts, origin=SCRIPTED_ORIGIN), status=status,
        finish=finish, created_at=f"2026-10-26T10:00:{seq:02d}+00:00", **fields,
    )


def result_step(seq, results, turn_id="t1", session_id="s1", status=StepStatus.COMPLETED) -> StepRecord:
    parts = [
        ToolResultPart(call_id=call_id, name=name, content=(TextPart(text=text),), is_error=is_error)
        for call_id, name, text, is_error in results
    ]
    return StepRecord(
        id=f"{session_id}.r{seq}", session_id=session_id, turn_id=turn_id, seq=seq,
        message=message(f"mr{seq}", "tool", *parts), status=status, created_at=f"2026-10-26T10:00:{seq:02d}+00:00",
    )


def final_step(seq, text, turn_id="t1", session_id="s1", status=StepStatus.COMPLETED, finish=FinishReason.STOP, **fields) -> StepRecord:
    return StepRecord(
        id=f"{session_id}.f{seq}", session_id=session_id, turn_id=turn_id, seq=seq,
        message=message(f"mf{seq}", "assistant", TextPart(text=text), origin=SCRIPTED_ORIGIN), status=status,
        finish=finish, created_at=f"2026-10-26T10:00:{seq:02d}+00:00", **fields,
    )

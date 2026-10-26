import json
import logging
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from src.features.llm.harness import codec
from src.features.llm.harness.errors import HarnessError
from src.features.llm.harness.events import FinishReason, Usage
from src.features.llm.harness.messages import Message, Origin, Part, Role
from src.features.llm.harness.recovery import INTERRUPTED_REASON, interrupted_tool_step
from src.features.llm.harness.turn_repository import TurnRepository
from src.features.llm.harness.turn_state import StepRecord, StepStatus, TurnState, TurnStatus
from src.platform.util.ids import generate_ulid

logger = logging.getLogger(__name__)

OPEN_STATUSES = (TurnStatus.RUNNING.value, TurnStatus.AWAITING_DECISION.value)

INTERRUPTED_ERROR = HarnessError("The server restarted before this turn finished.", code="interrupted")


def step_from_row(row) -> StepRecord:
    message = Message(
        id=row["message_id"],
        role=Role(row["role"]),
        parts=codec.loads(Tuple[Part, ...], row["parts"]),
        origin=codec.loads(Origin, row["origin"]) if row["origin"] is not None else None,
        raw=json.loads(row["raw"]) if row["raw"] is not None else None,
    )
    return StepRecord(
        id=row["id"],
        session_id=row["session_id"],
        turn_id=row["turn_id"],
        seq=row["seq"],
        message=message,
        status=StepStatus(row["status"]),
        usage=codec.loads(Usage, row["usage"]) if row["usage"] is not None else None,
        finish=FinishReason(row["finish"]) if row["finish"] is not None else None,
        finish_raw=row["finish_raw"],
        created_at=row["created_at"],
    )


def _step_values(step: StepRecord) -> Tuple:
    message = step.message
    return (
        step.id,
        step.session_id,
        step.turn_id,
        step.seq,
        message.id,
        message.role.value,
        codec.dumps(message.parts),
        codec.dumps(message.raw) if message.raw is not None else None,
        codec.dumps(message.origin) if message.origin is not None else None,
        codec.dumps(step.usage) if step.usage is not None else None,
        step.finish.value if step.finish is not None else None,
        step.finish_raw,
        step.status.value,
        step.created_at,
    )


class SqliteTurnStore:
    def __init__(self, repository: Optional[TurnRepository] = None) -> None:
        self._repository = repository or TurnRepository()

    def steps(self, session_id: str) -> List[StepRecord]:
        return [step_from_row(row) for row in self._repository.step_rows(session_id)]

    def turn_steps(self, turn_id: str) -> List[StepRecord]:
        return [step_from_row(row) for row in self._repository.turn_step_rows(turn_id)]

    def steps_by_turn(self, turn_ids: Sequence[str]) -> Dict[str, List[StepRecord]]:
        return {
            turn_id: [step_from_row(row) for row in rows]
            for turn_id, rows in self._repository.step_rows_by_turn(turn_ids).items()
        }

    def states(self, turn_ids: Sequence[str]) -> Dict[str, TurnState]:
        return {
            turn_id: codec.loads(TurnState, text) for turn_id, text in self._repository.state_jsons(turn_ids).items()
        }

    def next_seq(self, session_id: str) -> int:
        return self._repository.next_seq(session_id)

    def append(self, step: StepRecord) -> None:
        self._repository.insert_step(_step_values(step))

    def save_state(self, state: TurnState) -> None:
        self._repository.upsert_state(
            state.turn_id,
            state.session_id,
            state.status.value,
            codec.dumps(state),
            state.snapshot.llm_config_id or None,
        )

    def load_state(self, turn_id: str) -> Optional[TurnState]:
        text = self._repository.state_json(turn_id)
        return codec.loads(TurnState, text) if text is not None else None

    def open_turn(self, session_id: str) -> Optional[TurnState]:
        text = self._repository.open_state_json(session_id, OPEN_STATUSES)
        return codec.loads(TurnState, text) if text is not None else None

    def fail_interrupted(self) -> int:
        running = self._repository.state_jsons_with_status(TurnStatus.RUNNING.value)
        for turn_id, text in running:
            try:
                state = codec.loads(TurnState, text)
            except (ValueError, TypeError):
                logger.exception("Turn %s has unreadable state; marking it failed", turn_id)
                self._repository.set_status(turn_id, TurnStatus.FAILED.value)
                continue
            self._close_interrupted(state)
        return len(running)

    def _close_interrupted(self, state: TurnState) -> None:
        tool_step = interrupted_tool_step(
            self.turn_steps(state.turn_id),
            state,
            step_id=generate_ulid(),
            seq=self.next_seq(state.session_id),
            status=StepStatus.FAILED,
            reason=INTERRUPTED_REASON,
        )
        if tool_step is not None:
            self.append(tool_step)
        self.save_state(replace(state, status=TurnStatus.FAILED, error=INTERRUPTED_ERROR.to_json()))

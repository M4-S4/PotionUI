from typing import Dict, List, Optional, Sequence

from src.features.llm.harness.turn_state import StepRecord, TurnState, TurnStatus


class InMemoryTurnStore:
    def __init__(self) -> None:
        self._steps: Dict[str, List[StepRecord]] = {}
        self._states: Dict[str, TurnState] = {}

    def steps(self, session_id: str) -> List[StepRecord]:
        return list(self._steps.get(session_id, []))

    def turn_steps(self, turn_id: str) -> List[StepRecord]:
        return [step for steps in self._steps.values() for step in steps if step.turn_id == turn_id]

    def steps_by_turn(self, turn_ids: Sequence[str]) -> Dict[str, List[StepRecord]]:
        wanted = set(turn_ids)
        grouped: Dict[str, List[StepRecord]] = {}
        for steps in self._steps.values():
            for step in steps:
                if step.turn_id in wanted:
                    grouped.setdefault(step.turn_id, []).append(step)
        return grouped

    def states(self, turn_ids: Sequence[str]) -> Dict[str, TurnState]:
        return {turn_id: self._states[turn_id] for turn_id in turn_ids if turn_id in self._states}

    def next_seq(self, session_id: str) -> int:
        return len(self._steps.get(session_id, []))

    def append(self, step: StepRecord) -> None:
        recorded = self._steps.setdefault(step.session_id, [])
        if step.seq != len(recorded):
            raise ValueError(f"step seq {step.seq} is not the next seq {len(recorded)} of session {step.session_id}")
        recorded.append(step)

    def save_state(self, state: TurnState) -> None:
        self._states[state.turn_id] = state

    def load_state(self, turn_id: str) -> Optional[TurnState]:
        return self._states.get(turn_id)

    def open_turn(self, session_id: str) -> Optional[TurnState]:
        waiting = [
            state for state in self._states.values()
            if state.session_id == session_id and state.status is TurnStatus.AWAITING_DECISION
        ]
        return waiting[-1] if waiting else None

    def latest_turn(self, session_id: str) -> Optional[TurnState]:
        turns = [state for state in self._states.values() if state.session_id == session_id]
        return turns[-1] if turns else None

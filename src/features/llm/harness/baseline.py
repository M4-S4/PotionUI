from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.features.llm import context_budget
from src.features.llm.harness.messages import Message, Origin, ReasoningPart
from src.features.llm.harness.profile import CapabilityProfile, ReasoningKind, ReplayPolicy
from src.features.llm.harness.turn_state import Notice, StepRecord, TurnState, TurnStatus

PROMPT_TOOL_TYPES = frozenset({"native"})


class StaticProfileResolver:
    def __init__(self, types: Any, windows: Any) -> None:
        self._types = types
        self._windows = windows

    def resolve(self, config: Any) -> CapabilityProfile:
        entry = self._types.require(config.type)
        options = config.provider_options or {}
        capacity = self._windows.capacity(config)
        thinking = context_budget.resolve_thinking_enabled(config)
        return CapabilityProfile(
            adapter=getattr(entry.adapter, "name", config.type),
            model=config.model,
            context_window=capacity.capacity_tokens if capacity.source != "unknown" else None,
            prompt_tools=config.type in PROMPT_TOOL_TYPES or bool(options.get("force_prompt_tools", False)),
            reasoning=ReasoningKind.TOGGLE if thinking else ReasoningKind.NONE,
            replay=ReplayPolicy.DROP_EARLIER,
            vision=bool(config.supports_vision),
        )

    async def refresh(self, config: Any) -> CapabilityProfile:
        await self._windows.ensure(config)
        return self.resolve(config)


def _same_model(left: Optional[Origin], right: Origin) -> bool:
    return left is not None and (left.adapter, left.provider, left.model) == (right.adapter, right.provider, right.model)


def _without_reasoning(message: Message) -> Message:
    if not message.reasoning():
        return message
    return Message(
        id=message.id,
        role=message.role,
        parts=tuple(part for part in message.parts if not isinstance(part, ReasoningPart)),
        origin=message.origin,
        raw=None,
    )


def turn_only_replay(
    steps: Sequence[StepRecord],
    *,
    turn_id: str,
    target: Origin,
    profile: CapabilityProfile,
) -> Tuple[Tuple[Message, ...], Tuple[Notice, ...]]:
    messages = tuple(
        step.message if step.turn_id == turn_id and _same_model(step.message.origin, target) else _without_reasoning(step.message)
        for step in steps
    )
    return messages, ()


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

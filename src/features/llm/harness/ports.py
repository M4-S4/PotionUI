from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, List, Optional, Protocol, Sequence, Tuple, Union

from src.features.llm.harness.errors import HarnessError
from src.features.llm.harness.messages import Message, Origin, ToolCallPart
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.harness.turn_state import Decision, Notice, PendingCall, StepRecord, ToolFinished, ToolStarted, TurnState

if TYPE_CHECKING:
    from src.features.llm.repository import LLMConfig


class TurnStore(Protocol):
    def steps(self, session_id: str) -> List[StepRecord]: ...

    def next_seq(self, session_id: str) -> int: ...

    def append(self, step: StepRecord) -> None: ...

    def save_state(self, state: TurnState) -> None: ...

    def load_state(self, turn_id: str) -> Optional[TurnState]: ...

    def open_turn(self, session_id: str) -> Optional[TurnState]: ...

    def latest_turn(self, session_id: str) -> Optional[TurnState]: ...


class ProfileResolver(Protocol):
    def resolve(self, config: LLMConfig) -> CapabilityProfile: ...

    async def refresh(self, config: LLMConfig) -> CapabilityProfile: ...


class ReplaySelector(Protocol):
    def __call__(
        self,
        steps: Sequence[StepRecord],
        *,
        turn_id: str,
        target: Origin,
        profile: CapabilityProfile,
    ) -> Tuple[Tuple[Message, ...], Tuple[Notice, ...]]: ...


class ToolRunner(Protocol):
    def run(
        self,
        calls: Sequence[ToolCallPart],
        state: TurnState,
        context: Any,
        decisions: Sequence[Decision] = (),
    ) -> AsyncIterator[Union[ToolStarted, ToolFinished, PendingCall]]: ...


class RetryPolicy(Protocol):
    def delay(self, error: HarnessError, attempt: int) -> Optional[float]: ...

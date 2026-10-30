from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, FrozenSet, Mapping, Optional, Tuple, Union

from src.features.llm.harness.adapter import Settings, ToolSpec
from src.features.llm.harness.events import FinishReason, StreamEvent, Usage
from src.features.llm.harness.messages import Message, ToolCallPart, ToolResultPart
from src.features.llm.harness.profile import CapabilityProfile


class TurnStatus(StrEnum):
    RUNNING = "running"
    AWAITING_DECISION = "awaiting_decision"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class StepStatus(StrEnum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    FAILED = "failed"


class DecisionKind(StrEnum):
    APPROVAL = "approval"
    EXTERNAL = "external"


class NoticeKind(StrEnum):
    REASONING_DROPPED = "reasoning_dropped"
    DEGRADED_RETRY = "degraded_retry"
    RETRYING = "retrying"
    TRUNCATED = "truncated"
    APPROVAL_EXPIRED = "approval_expired"
    TOOL_CHOICE_DOWNGRADED = "tool_choice_downgraded"
    STRUCTURED_FALLBACK = "structured_fallback"
    STRUCTURED_REPAIRED = "structured_repaired"


@dataclass(frozen=True)
class Notice:
    kind: NoticeKind
    message: str
    step_seq: Optional[int] = None
    detail: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class PendingCall:
    TYPE: ClassVar[str] = "pending_call"
    call_id: str
    name: str
    decision: DecisionKind
    preview: Any = None


@dataclass(frozen=True)
class Decision:
    call_id: str
    approved: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class TurnSnapshot:
    llm_config_id: str
    system: str
    tools: Tuple[ToolSpec, ...]
    settings: Settings
    profile: CapabilityProfile
    mode: Optional[str] = None
    deferred_tools: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnState:
    turn_id: str
    session_id: str
    status: TurnStatus
    snapshot: TurnSnapshot
    first_seq: int
    rounds_used: int = 0
    max_rounds: int = 20
    pending: Tuple[PendingCall, ...] = ()
    gathered: Tuple[ToolResultPart, ...] = ()
    failed_calls: Mapping[str, str] = field(default_factory=dict)
    tool_failures: Mapping[str, int] = field(default_factory=dict)
    applied_calls: FrozenSet[str] = frozenset()
    loaded_tool_groups: Tuple[str, ...] = ()
    rescue_retries: int = 0
    notices: Tuple[Notice, ...] = ()
    error: Optional[Mapping[str, Any]] = None
    prefix_changed_at: Optional[int] = None


@dataclass(frozen=True)
class StepRecord:
    id: str
    session_id: str
    turn_id: str
    seq: int
    message: Message
    status: StepStatus
    usage: Optional[Usage] = None
    finish: Optional[FinishReason] = None
    finish_raw: Optional[str] = None
    created_at: Optional[str] = None


@dataclass(frozen=True)
class StepCommitted:
    TYPE: ClassVar[str] = "step_committed"
    step: StepRecord


@dataclass(frozen=True)
class StepDiscarded:
    TYPE: ClassVar[str] = "step_discarded"
    seq: int


@dataclass(frozen=True)
class ToolStarted:
    TYPE: ClassVar[str] = "tool_started"
    call: ToolCallPart


@dataclass(frozen=True)
class ToolFinished:
    TYPE: ClassVar[str] = "tool_finished"
    result: ToolResultPart
    record: Any = None


@dataclass(frozen=True)
class DecisionRequired:
    TYPE: ClassVar[str] = "decision_required"
    state: TurnState


@dataclass(frozen=True)
class NoticeRaised:
    TYPE: ClassVar[str] = "notice"
    notice: Notice


@dataclass(frozen=True)
class TurnEnded:
    TYPE: ClassVar[str] = "turn_ended"
    state: TurnState


EngineEvent = Union[StreamEvent, StepCommitted, StepDiscarded, ToolStarted, ToolFinished, DecisionRequired, NoticeRaised, TurnEnded]

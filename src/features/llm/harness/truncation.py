from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from src.features.llm.harness.adapter import Settings
from src.features.llm.harness.errors import HarnessError
from src.features.llm.harness.events import FinishReason
from src.features.llm.harness.messages import Message
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.harness.turn_state import Notice, NoticeKind

MAX_TRUNCATION_RETRIES = 1
TRUNCATION_GROWTH = 4


class OutputTruncated(HarnessError):
    kind = "output_truncated"


class TruncationAction(StrEnum):
    NONE = "none"
    RETRY = "retry"
    FAIL = "fail"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class TruncationDecision:
    action: TruncationAction
    max_tokens: Optional[int] = None
    notice: Optional[Notice] = None
    error: Optional[OutputTruncated] = None


def raised_limit(settings: Settings, profile: CapabilityProfile) -> Optional[int]:
    ceiling = profile.max_output_tokens
    current = settings.max_tokens
    if current is None:
        return ceiling
    target = current * TRUNCATION_GROWTH
    if ceiling is not None:
        target = min(target, ceiling)
    return target if target > current else None


def decide_truncation(
    message: Message,
    reason: Optional[FinishReason],
    settings: Settings,
    profile: CapabilityProfile,
    retries_used: int = 0,
) -> TruncationDecision:
    if reason is not FinishReason.LENGTH:
        return TruncationDecision(TruncationAction.NONE)
    if not message.tool_calls():
        return TruncationDecision(
            TruncationAction.INCOMPLETE,
            notice=Notice(NoticeKind.TRUNCATED, "The reply was cut off by the output limit.", detail={"calls": 0}),
        )
    limit = raised_limit(settings, profile) if retries_used < MAX_TRUNCATION_RETRIES else None
    if limit is None:
        text = (
            "The reply was cut off in the middle of a tool call again after the output limit was raised, so the tool was not run."
            if retries_used
            else "The reply was cut off in the middle of a tool call and the output limit cannot be raised, so the tool was not run."
        )
        return TruncationDecision(
            TruncationAction.FAIL,
            notice=Notice(NoticeKind.TRUNCATED, text, detail={"retries": retries_used}),
            error=OutputTruncated(text, detail={"max_tokens": settings.max_tokens, "retries": retries_used}),
        )
    return TruncationDecision(
        TruncationAction.RETRY,
        max_tokens=limit,
        notice=Notice(
            NoticeKind.TRUNCATED,
            f"The reply was cut off in the middle of a tool call, so it is being retried with a larger output limit ({limit} tokens).",
            detail={"max_tokens": limit, "previous": settings.max_tokens},
        ),
    )

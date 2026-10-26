from typing import Dict, List, Optional, Sequence, Tuple

from src.features.llm.harness.messages import (
    CompactionPart,
    Message,
    Origin,
    Part,
    ReasoningPart,
    Role,
    ToolCallPart,
)
from src.features.llm.harness.profile import CapabilityProfile, ReplayPolicy
from src.features.llm.harness.turn_state import Notice, NoticeKind, StepRecord

PLACEHOLDER_SIGNATURE = "placeholder_signature"

CREDENTIAL_CHANGED = "credential"
MODEL_CHANGED = "model"


def policy_replay(
    steps: Sequence[StepRecord],
    *,
    turn_id: str,
    target: Origin,
    profile: CapabilityProfile,
) -> Tuple[Tuple[Message, ...], Tuple[Notice, ...]]:
    messages: List[Message] = []
    dropped: Dict[Tuple[str, str], List[int]] = {}
    for step in steps:
        message = step.message
        if message.role is not Role.ASSISTANT:
            messages.append(message)
            continue
        selected, foreign_drop = _select(message, step.turn_id == turn_id, target, profile.replay)
        if foreign_drop:
            reason = CREDENTIAL_CHANGED if _same_model(message.origin, target) else MODEL_CHANGED
            dropped.setdefault((reason, message.origin.model), []).append(step.seq)
        if selected is not None:
            messages.append(selected)
    return tuple(messages), tuple(_notices(dropped))


def missing_reasoning(messages: Sequence[Message]) -> Tuple[str, ...]:
    return tuple(
        message.id
        for message in messages
        if message.role is Role.ASSISTANT
        and not any(part.text or part.summary or part.opaque for part in message.reasoning())
    )


def _select(
    message: Message, in_progress: bool, target: Origin, policy: ReplayPolicy
) -> Tuple[Optional[Message], bool]:
    foreign = not _same_origin(message.origin, target)
    keep_reasoning = _keeps_reasoning(policy, in_progress)
    parts: List[Part] = []
    changed = False
    foreign_drop = False
    for part in message.parts:
        if isinstance(part, ReasoningPart):
            if not keep_reasoning:
                changed = True
            elif foreign and part.opaque:
                changed = foreign_drop = True
            else:
                parts.append(part)
        elif isinstance(part, ToolCallPart) and foreign:
            changed = True
            parts.append(_without_provider_data(part))
        elif isinstance(part, CompactionPart) and foreign and part.opaque:
            changed = True
            parts.append(CompactionPart(summary=part.summary))
        else:
            parts.append(part)
    if not changed:
        return message, False
    if not parts and message.parts:
        return None, foreign_drop
    return Message(id=message.id, role=message.role, parts=tuple(parts), origin=message.origin, raw=None), foreign_drop


def _without_provider_data(call: ToolCallPart) -> ToolCallPart:
    return ToolCallPart(
        call_id=call.call_id,
        name=call.name,
        args=call.args,
        args_raw=call.args_raw,
        wire_id=call.wire_id,
        opaque={PLACEHOLDER_SIGNATURE: True},
    )


def _keeps_reasoning(policy: ReplayPolicy, in_progress: bool) -> bool:
    if policy is ReplayPolicy.NEVER:
        return False
    return in_progress or policy in (ReplayPolicy.SEND_ALL, ReplayPolicy.PRESERVE)


def _same_model(origin: Origin, target: Origin) -> bool:
    return (origin.adapter, origin.provider, origin.model) == (target.adapter, target.provider, target.model)


def _same_origin(origin: Origin, target: Origin) -> bool:
    return _same_model(origin, target) and origin.credential == target.credential


def _notices(dropped: Dict[Tuple[str, str], List[int]]) -> List[Notice]:
    notices = []
    for (reason, model), seqs in dropped.items():
        notices.append(
            Notice(
                kind=NoticeKind.REASONING_DROPPED,
                message=_dropped_text(reason, model),
                step_seq=max(seqs),
                detail={"reason": reason, "model": model, "steps": list(seqs)},
            )
        )
    return notices


def _dropped_text(reason: str, model: str) -> str:
    if reason == CREDENTIAL_CHANGED:
        return "Reasoning made with a different API key was left out."
    return f"Reasoning made by {model} was left out because the model changed."

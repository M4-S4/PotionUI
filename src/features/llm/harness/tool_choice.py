from dataclasses import dataclass
from typing import Optional

from src.features.llm.harness.adapter import Settings, ToolChoice
from src.features.llm.harness.profile import CapabilityProfile, ReasoningKind
from src.features.llm.harness.turn_state import Notice, NoticeKind

AUTO = "auto"
NONE = "none"
REQUIRED = "required"
NAMED = "named"

REASONING_OFF = frozenset({"off", "none", "false", "disabled", "0"})


@dataclass(frozen=True)
class ToolChoiceDecision:
    choice: Optional[ToolChoice]
    drop_tools: bool = False
    notice: Optional[Notice] = None


def reasoning_active(settings: Settings, profile: CapabilityProfile) -> bool:
    if profile.reasoning is ReasoningKind.NONE:
        return False
    if profile.reasoning is ReasoningKind.ALWAYS_ON:
        return True
    value = settings.reasoning
    if value is False:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in REASONING_OFF
    return True


def effective_tool_choice(settings: Settings, profile: CapabilityProfile) -> ToolChoiceDecision:
    choice = settings.tool_choice
    if choice is None:
        return ToolChoiceDecision(None)
    thinking = reasoning_active(settings, profile)
    allowed = profile.tool_choice_with_reasoning if thinking else profile.tool_choice
    usable = choice.mode != NAMED or bool(choice.name)
    if usable and choice.mode in allowed:
        return ToolChoiceDecision(choice)
    if choice.mode == AUTO:
        return ToolChoiceDecision(None)
    if choice.mode == NONE:
        return ToolChoiceDecision(None, drop_tools=True)
    return ToolChoiceDecision(
        ToolChoice(AUTO) if AUTO in allowed else None,
        notice=Notice(
            kind=NoticeKind.DEGRADED_RETRY,
            message=_message(choice, thinking, profile),
            detail={
                "reason": "tool_choice_unsupported",
                "requested": choice.mode,
                "name": choice.name,
                "reasoning": thinking,
            },
        ),
    )


def _message(choice: ToolChoice, thinking: bool, profile: CapabilityProfile) -> str:
    target = f"the {choice.name} tool" if choice.mode == NAMED and choice.name else "a tool"
    if thinking and choice.mode in profile.tool_choice:
        return f"This model cannot be forced to call {target} while it is reasoning, so it chooses for itself."
    return f"This model cannot be forced to call {target}, so it chooses for itself."

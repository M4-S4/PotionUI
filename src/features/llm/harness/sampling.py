from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from src.features.llm.harness.adapter import Settings
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.harness.profile_schema import SAMPLING_PARAMETERS

if TYPE_CHECKING:
    from src.features.llm.repository import LLMConfig

EFFORT_OFF = "none"


def reasoning_from_options(options: Optional[Mapping[str, Any]]) -> Union[bool, str, None]:
    options = options or {}
    effort = options.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip():
        return False if effort == EFFORT_OFF else effort
    think = options.get("think")
    if isinstance(think, bool) or (isinstance(think, str) and think.strip()):
        return think
    thinking = options.get("thinking")
    return thinking if isinstance(thinking, bool) else None


def _first(*candidates: Any) -> Any:
    return next((value for value in candidates if value is not None), None)


def resolve_settings(config: LLMConfig, profile: CapabilityProfile, request: Settings) -> Settings:
    options = config.provider_options or {}
    configured = {"temperature": config.temperature}
    values = {
        name: _first(
            getattr(request, name),
            options.get(name),
            configured.get(name),
            profile.recommended_sampling.get(name),
        )
        for name in SAMPLING_PARAMETERS
    }
    if values["top_k"] is not None:
        values["top_k"] = int(values["top_k"])
    for name in profile.sampling_rejected:
        if name in values:
            values[name] = None
    return replace(
        request,
        max_tokens=_first(request.max_tokens, options.get("max_tokens"), config.max_tokens),
        reasoning=_first(request.reasoning, reasoning_from_options(options)),
        **values,
    )

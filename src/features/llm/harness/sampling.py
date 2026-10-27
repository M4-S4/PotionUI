from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from src.features.llm.harness.adapter import Settings
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.harness.profile_schema import SAMPLING_PARAMETERS

if TYPE_CHECKING:
    from src.features.llm.repository import LLMConfig


def _first(*candidates: Any) -> Any:
    return next((value for value in candidates if value is not None), None)


def resolve_settings(config: LLMConfig, profile: CapabilityProfile, request: Settings) -> Settings:
    configured = Settings.from_options(config.provider_options)
    values = {
        name: _first(
            getattr(request, name),
            getattr(configured, name),
            config.temperature if name == "temperature" else None,
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
        max_tokens=_first(request.max_tokens, configured.max_tokens, config.max_tokens),
        reasoning=_first(request.reasoning, configured.reasoning),
        **values,
    )

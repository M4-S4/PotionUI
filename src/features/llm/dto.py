"""LLM Data Transfer Objects for API requests and responses."""

import math
from typing import Dict, Optional, Any
from pydantic import BaseModel, field_validator, model_validator

from src.features.llm.harness.profile_schema import validate_overrides

# provider_options keys shared by the native and Ollama sampling controls
# (same name, same meaning on both providers) plus native's own
# repetition_penalty (Ollama's equivalent, "repeat_penalty", is a distinct
# key left unvalidated — legacy free-form).
_PROBABILITY_OPTION_KEYS = ("top_p", "min_p")

_THINK_LEVELS = ("low", "medium", "high")

_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


# ============================================================================
# LLM Configuration DTOs
# ============================================================================

class LLMConfigRequest(BaseModel):
    """Request model for creating/updating LLM configuration."""
    id: Optional[str] = None
    name: str
    type: str  # "ollama" or "openai"
    enabled: bool
    base_url: str
    api_key: Optional[str] = None
    model: str
    system_message: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: int = 30
    supports_vision: bool = False
    disable_system_prompt: bool = False
    memory_reflection: bool = True
    show_thinking: bool = False
    provider_options: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_reasoning_options(self) -> "LLMConfigRequest":
        options = self.provider_options or {}
        thinking = options.get("thinking")
        if thinking is not None and not isinstance(thinking, bool):
            raise ValueError(f"provider_options.thinking must be true, false, or omitted (null), got {thinking!r}")
        think = options.get("think")
        if think is not None and not isinstance(think, bool) and think not in _THINK_LEVELS:
            raise ValueError(f"provider_options.think must be true, false or one of {_THINK_LEVELS!r}, got {think!r}")
        return self

    @model_validator(mode="after")
    def _validate_sampling_options(self) -> "LLMConfigRequest":
        """``top_k``/``top_p``/``min_p``/``repetition_penalty`` are shared
        sampling knobs (native reads them directly; Ollama copies ``top_k``/
        ``top_p``/``min_p`` verbatim into its request options — see
        ``OLLAMA_OPTION_KEYS``). Malformed values fail loudly here rather than
        being silently truncated or substituted downstream. Omitted/null stays
        valid — that is how a config leaves a knob to the provider/model."""
        if not self.provider_options:
            return self
        opts = self.provider_options

        if "top_k" in opts and opts["top_k"] is not None:
            value = opts["top_k"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"provider_options.top_k must be a non-negative integer or omitted (null), got {value!r}"
                )

        for key in _PROBABILITY_OPTION_KEYS:
            if key in opts and opts[key] is not None:
                value = opts[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not (0.0 <= value <= 1.0)
                ):
                    raise ValueError(
                        f"provider_options.{key} must be a probability between 0 and 1, or omitted (null), got {value!r}"
                    )

        if "repetition_penalty" in opts and opts["repetition_penalty"] is not None:
            value = opts["repetition_penalty"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"provider_options.repetition_penalty must be a positive number, or omitted (null), got {value!r}"
                )

        return self

    @model_validator(mode="after")
    def _validate_request_options(self) -> "LLMConfigRequest":
        if not self.provider_options:
            return self
        opts = self.provider_options

        if "seed" in opts and opts["seed"] is not None:
            value = opts["seed"]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "provider_options.seed must be an integer or omitted (null), "
                    f"got {value!r}"
                )

        if "reasoning_effort" in opts and opts["reasoning_effort"] is not None:
            value = opts["reasoning_effort"]
            if value not in _REASONING_EFFORTS:
                raise ValueError(
                    "provider_options.reasoning_effort must be one of "
                    f"{_REASONING_EFFORTS!r}, or omitted (null), got {value!r}"
                )

        if "parallel_tool_calls" in opts and opts["parallel_tool_calls"] is not None:
            value = opts["parallel_tool_calls"]
            if not isinstance(value, bool):
                raise ValueError(
                    "provider_options.parallel_tool_calls must be true, false, or omitted (null), "
                    f"got {value!r}"
                )

        if "stop" in opts and opts["stop"] is not None:
            value = opts["stop"]
            is_valid = isinstance(value, str) or (
                isinstance(value, list)
                and 1 <= len(value) <= 4
                and all(isinstance(item, str) for item in value)
            )
            if not is_valid:
                raise ValueError(
                    "provider_options.stop must be a string, a list of up to 4 strings, or "
                    f"omitted (null), got {value!r}"
                )

        return self


# ============================================================================
# Tool Governance DTOs
# ============================================================================

class ModelDiscoveryRequest(BaseModel):
    type: str
    base_url: str = ""
    api_key: Optional[str] = None
    config_id: Optional[str] = None


class ProfileOverridesRequest(BaseModel):
    overrides: Dict[str, Any]

    @field_validator("overrides")
    @classmethod
    def _validate_overrides(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return validate_overrides(value)


class ToolGovernanceUpdateRequest(BaseModel):
    """Admin update to one tool's governance row. Both fields optional so a
    client can flip just `enabled` or just `locked` in one call."""
    enabled: Optional[bool] = None
    locked: Optional[bool] = None


class UserToolPreferenceRequest(BaseModel):
    """A user's opt-out toggle for one tool. `llm_config_id` names the caller's
    active LLM config - the toggle is always checked against that config's
    governance row (403/409 if it disabled or locked the tool); the opt-out
    itself is always stored globally, not per-config."""
    disabled: bool
    llm_config_id: str


class LLMConfigResponse(BaseModel):
    """Response model for LLM configuration.

    The API key is never returned. ``api_key_set`` reports whether a key is
    stored so the UI can show a "configured" state without exposing the
    secret; to update the key, clients send a new value on the request DTO.
    """
    id: str
    name: str
    type: str
    enabled: bool
    base_url: str
    api_key_set: bool = False
    model: str
    system_message: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: int
    supports_vision: bool = False
    disable_system_prompt: bool = False
    memory_reflection: bool = True
    show_thinking: bool = False
    provider_options: Optional[Dict[str, Any]] = None
    is_default: bool = False


# ============================================================================
# LLM Generation DTOs
# ============================================================================

class LLMGenerateRequest(BaseModel):
    """Request model for LLM text generation."""
    prompt: str
    config_id: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded image or file path for vision models


class LLMGenerateResponse(BaseModel):
    """Response model for LLM text generation."""
    content: str
    model: str
    provider_id: str
    tokens_used: Optional[int] = None


# ============================================================================
# User LLM Assignment DTOs
# ============================================================================

class UserLLMAssignmentRequest(BaseModel):
    """Request model for assigning LLM to user."""
    user_id: str
    llm_config_id: str

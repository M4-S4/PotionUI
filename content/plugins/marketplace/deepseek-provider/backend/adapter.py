from typing import Any, Callable, Dict

from src.plugin_api import plugin_settings
from src.plugin_api.llm import CapabilityProfile, OpenAIChatAdapter, RenderRequest

PLUGIN_ID = "deepseek-provider"
PLUGIN_VERSION = "0.1.0"
DEFAULT_THINKING = "auto"
DEFAULT_REASONING_EFFORT = "high"


def uses_tools(body: Dict[str, Any]) -> bool:
    return bool(body.get("tools")) or any(
        message.get("role") == "tool" or message.get("tool_calls") for message in body["messages"]
    )


def lacks_reasoning(body: Dict[str, Any]) -> bool:
    return any(
        message.get("role") == "assistant" and not message.get("reasoning_content")
        for message in body["messages"]
    )


class DeepSeekAdapter(OpenAIChatAdapter):
    display_name = "DeepSeek"

    def __init__(self, settings: Callable[[], Dict[str, Any]] = lambda: plugin_settings(PLUGIN_ID)) -> None:
        self._settings = settings

    def _thinking(self, body: Dict[str, Any]) -> Dict[str, Any]:
        settings = self._settings()
        if settings.get("thinking", DEFAULT_THINKING) == "off" or (
            uses_tools(body) and lacks_reasoning(body)
        ):
            return {"thinking": {"type": "disabled"}}
        return {
            "thinking": {"type": "enabled"},
            "reasoning_effort": settings.get("reasoning_effort", DEFAULT_REASONING_EFFORT),
        }

    def build_body(self, request: RenderRequest, profile: CapabilityProfile) -> Dict[str, Any]:
        body = super().build_body(request, profile)
        if "max_completion_tokens" in body:
            body["max_tokens"] = body.pop("max_completion_tokens")
        body.pop("reasoning_effort", None)
        body.update(self._thinking(body))
        return body

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tests.features.llm.harness.history import IMAGE_B64, reasoned_history, tool_history, tool_schemas

HISTORY = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
    {"role": "user", "content": "how are you"},
]

PROVIDER_OPTIONS = {
    "top_p": 0.9,
    "presence_penalty": 0.25,
    "frequency_penalty": 0.5,
    "unsupported_key": "ignored",
}

OVERRIDE = {"temperature": 0.1, "max_tokens": 64, "top_p": 0.5, "top_k": 40, "think": False}

REASONING = "The form has to be read before anything is changed."

EXPOSED_KEYS = ["top_p", "presence_penalty", "frequency_penalty", "seed", "stop", "reasoning_effort", "parallel_tool_calls"]


@dataclass
class Case:
    id: str
    history: List[Dict[str, Any]]
    config: Dict[str, Any] = field(default_factory=dict)
    override: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    image: Optional[str] = None
    system: str = "SYS"
    profile: Dict[str, Any] = field(default_factory=dict)


def string_arguments_history() -> List[Dict[str, Any]]:
    history = tool_history()
    history[1]["tool_calls"][0]["function"]["arguments"] = '{"scope": "all"}'
    return history


CASES: List[Case] = [
    Case("prompt_only", [{"role": "user", "content": "say hi"}]),
    Case("prompt_with_image", [{"role": "user", "content": "say hi"}], image=IMAGE_B64),
    Case("api_key_becomes_a_bearer_header", [{"role": "user", "content": "say hi"}], config={"api_key": "sk-golden"}),
    Case("plain_history", HISTORY),
    Case(
        "provider_options_and_override",
        HISTORY,
        config={"provider_options": PROVIDER_OPTIONS},
        override=OVERRIDE,
    ),
    Case("override_top_p_without_a_saved_one", HISTORY, override={"top_p": 0.3}),
    Case("history_image_on_the_last_user_message", HISTORY, image=IMAGE_B64),
    Case("tool_history_with_tools", tool_history(), tools=tool_schemas()),
    Case("tool_arguments_as_a_json_string", string_arguments_history(), tools=tool_schemas()),
    Case("tool_history_without_tools", tool_history()),
    Case("tools_with_an_image", tool_history(), tools=tool_schemas(), image=IMAGE_B64),
    Case(
        "seed_stop_effort_and_parallel_calls",
        HISTORY,
        config={
            "provider_options": {
                "seed": 42,
                "stop": ["\n\n", "END"],
                "reasoning_effort": "high",
                "parallel_tool_calls": False,
            }
        },
    ),
    Case("explicit_null_options_are_omitted", HISTORY, config={"provider_options": {key: None for key in EXPOSED_KEYS}}),
    Case(
        "foreign_option_keys_never_reach_the_wire",
        HISTORY,
        config={
            "provider_options": {
                "top_k": 40,
                "min_p": 0.05,
                "repetition_penalty": 1.1,
                "repeat_penalty": 1.1,
                "num_ctx": 8192,
                "think": False,
            }
        },
    ),
    Case(
        "tools_options_image_and_override_together",
        tool_history(),
        config={"api_key": "sk-golden", "provider_options": PROVIDER_OPTIONS},
        override=OVERRIDE,
        tools=tool_schemas(),
        image=IMAGE_B64,
    ),
    Case("tool_turn_carries_its_reasoning", reasoned_history(REASONING), tools=tool_schemas()),
    Case("empty_reasoning_sends_nothing", reasoned_history(""), tools=tool_schemas()),
    Case("missing_reasoning_sends_nothing", reasoned_history(None), tools=tool_schemas()),
]

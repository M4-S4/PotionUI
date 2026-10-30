from typing import List

from tests.features.llm.harness.history import IMAGE_B64, reasoned_history, tool_history, tool_schemas
from tests.features.llm.harness.openai_chat_cases import HISTORY, REASONING, Case

PROVIDER_OPTIONS = {
    "num_ctx": 8192,
    "top_k": 20,
    "top_p": 0.8,
    "stop": ["</done>"],
    "keep_alive": "5m",
    "think": False,
    "not_an_option": "ignored",
}

OVERRIDE = {"temperature": 0.1, "max_tokens": 64, "top_p": 0.5, "top_k": 40, "think": False}

WINDOW = {"context_request": 8192}
PROMPT_TOOLS = {"prompt_tools": True}


def string_arguments_history() -> list:
    history = tool_history()
    history[1]["tool_calls"][0]["function"]["arguments"] = '{"scope": "all"}'
    return history


CASES: List[Case] = [
    Case("prompt_only", [{"role": "user", "content": "say hi"}]),
    Case("prompt_with_an_image", [{"role": "user", "content": "say hi"}], image=IMAGE_B64),
    Case("a_named_thinking_level", [{"role": "user", "content": "say hi"}], config={"provider_options": {"think": "high"}}),
    Case("provider_options_reach_options_and_root", [{"role": "user", "content": "say hi"}],
         config={"provider_options": PROVIDER_OPTIONS}, profile=WINDOW),
    Case("plain_history", HISTORY),
    Case("override_wins_over_provider_options", HISTORY, config={"provider_options": PROVIDER_OPTIONS},
         override=OVERRIDE, profile=WINDOW),
    Case("saved_min_p", HISTORY, config={"provider_options": {"min_p": 0.05}}),
    Case("per_call_min_p_wins", HISTORY, config={"provider_options": {"min_p": 0.05}}, override={"min_p": 0.2}),
    Case("history_image_on_the_last_user_message", HISTORY, image=IMAGE_B64),
    Case("empty_system_sends_no_system_turn", HISTORY, system=""),
    Case("native_tools", tool_history(), tools=tool_schemas()),
    Case("native_tool_arguments_as_an_object", string_arguments_history(), tools=tool_schemas()),
    Case("tool_history_without_tools", tool_history()),
    Case("prompt_tools_rewrite_the_conversation", tool_history(), tools=tool_schemas(), profile=PROMPT_TOOLS),
    Case("saved_null_think_is_false", tool_history(), config={"provider_options": {"think": None}}),
    Case("saved_null_think_with_tools", tool_history(), config={"provider_options": {"think": None}}, tools=tool_schemas()),
    Case("null_override_keeps_the_saved_think", HISTORY, config={"provider_options": {"think": "high"}},
         override={"think": None}),
    Case("explicit_true_with_tools", tool_history(), config={"provider_options": {"think": True}}, tools=tool_schemas()),
    Case("explicit_false_with_tools", tool_history(), config={"provider_options": {"think": False}}, tools=tool_schemas()),
    Case("named_level_with_tools", tool_history(), config={"provider_options": {"think": "high"}}, tools=tool_schemas()),
    Case("override_think_wins_with_tools", tool_history(), config={"provider_options": {"think": True}},
         override={"think": "medium"}, tools=tool_schemas()),
    Case("unrelated_override_keeps_saved_think", tool_history(), config={"provider_options": {"think": True}},
         override={"top_p": 0.5}, tools=tool_schemas()),
    Case("everything_together", tool_history(), config={"provider_options": PROVIDER_OPTIONS}, override=OVERRIDE,
         tools=tool_schemas(), image=IMAGE_B64, profile=WINDOW),
    Case("native_tool_turn_carries_its_thinking", reasoned_history(REASONING), tools=tool_schemas()),
    Case("empty_reasoning_sends_no_thinking", reasoned_history(""), tools=tool_schemas()),
    Case("missing_reasoning_sends_no_thinking", reasoned_history(None), tools=tool_schemas()),
    Case("prompt_tools_carry_the_thinking_on_the_text_turn", reasoned_history(REASONING), tools=tool_schemas(),
         profile=PROMPT_TOOLS),
    Case("prompt_tools_without_reasoning", reasoned_history(""), tools=tool_schemas(), profile=PROMPT_TOOLS),
]

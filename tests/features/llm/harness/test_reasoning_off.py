import asyncio
import json

from src.features.llm.harness.adapter import RenderRequest, Settings
from src.features.llm.harness.adapters.anthropic_messages import AnthropicMessagesAdapter
from src.features.llm.harness.adapters.gemini_interactions import GeminiInteractionsAdapter
from src.features.llm.harness.adapters.ollama_chat import OllamaChatAdapter
from src.features.llm.harness.adapters.openai_chat import OpenAIChatAdapter
from src.features.llm.harness.adapters.openai_responses import OpenAIResponsesAdapter
from src.features.llm.harness.messages import TextPart, message
from src.features.llm.harness.profiles import LayeredProfileResolver
from tests.fixtures.llm_profiles import ProbeAdapter, adapter_for, make_config, shipped_seed


def rendered(adapter, adapter_name, config_type, model, reasoning, **discovered):
    config = make_config(type=config_type, model=model, base_url="https://api.test", api_key="key")
    probe = ProbeAdapter(name=adapter_name, fields=discovered)
    resolver = LayeredProfileResolver(adapter_for(probe), shipped_seed())
    asyncio.run(resolver.ensure(config))
    profile = resolver.resolve(config)
    request = RenderRequest(config=config, system="s", messages=(message("u1", "user", TextPart("hi")),), settings=Settings(reasoning=reasoning))
    return adapter.render(request, profile).body


def test_openai_chat_for_deepseek_switches_thinking_off_in_its_dialect():
    off = rendered(OpenAIChatAdapter(), "openai-chat", "deepseek", "deepseek-v4-pro", False)
    default = rendered(OpenAIChatAdapter(), "openai-chat", "deepseek", "deepseek-v4-pro", None)

    assert off["thinking"] == {"type": "disabled"}
    assert "thinking" not in default


def test_ollama_sends_think_false_and_omits_nothing_else():
    off = rendered(OllamaChatAdapter(), "ollama-chat", "ollama", "qwen3.6:27b", False)
    on = rendered(OllamaChatAdapter(), "ollama-chat", "ollama", "qwen3.6:27b", True)

    assert off["think"] is False
    assert on["think"] is True


def test_openai_responses_sends_effort_none_when_the_model_has_that_level_and_otherwise_omits_reasoning():
    off = rendered(OpenAIResponsesAdapter(), "openai-responses", "openai", "gpt-5.5", False)
    assert off["reasoning"] == {"effort": "none"}

    legacy = rendered(OpenAIResponsesAdapter(), "openai-responses", "openai", "o3", False)
    assert "effort" not in legacy.get("reasoning", {})


def test_anthropic_disables_thinking_on_a_model_that_allows_it():
    allowed = rendered(AnthropicMessagesAdapter(), "anthropic-messages", "anthropic", "claude-opus-5", False)
    default = rendered(AnthropicMessagesAdapter(), "anthropic-messages", "anthropic", "claude-opus-5", None)

    assert allowed["thinking"] == {"type": "disabled"}
    assert "thinking" not in default


def test_gemini_selects_its_lowest_level_once_the_levels_are_known_and_otherwise_omits_reasoning():
    levels = dict(reasoning_levels=["minimal", "low", "medium", "high"])
    known = rendered(GeminiInteractionsAdapter(), "gemini-interactions", "gemini", "gemini-3-flash", False, **levels)
    unknown = rendered(GeminiInteractionsAdapter(), "gemini-interactions", "gemini", "gemini-3-flash", False)

    assert '"thinking_level":"minimal"' in json.dumps(known, separators=(",", ":"))
    assert "thinking_level" not in json.dumps(unknown)

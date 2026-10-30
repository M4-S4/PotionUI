from contextlib import aclosing
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

from src.features.llm.harness.adapters.native import requested_thinking
from src.features.llm.harness.adapter import Settings
from src.features.llm.native.runtime import NativeRuntime, RunStats


def sampling_for(config: Any, options_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    overrides = options_override or {}
    options = config.provider_options or {}
    sampling = {
        "temperature": overrides.get("temperature", config.temperature),
        "max_new_tokens": overrides.get("max_tokens", config.max_tokens),
        "top_p": overrides.get("top_p", options.get("top_p")),
        "top_k": overrides.get("top_k", options.get("top_k")),
        "min_p": overrides.get("min_p", options.get("min_p")),
        "repetition_penalty": overrides.get("repetition_penalty", options.get("repetition_penalty")),
    }
    return {key: value for key, value in sampling.items() if value is not None}


def chat_for(messages: List[Dict[str, Any]], system_message: Optional[str], image_data: Optional[str]) -> List[Dict[str, Any]]:
    chat: List[Dict[str, Any]] = []
    if system_message:
        chat.append({"role": "system", "content": system_message})
    chat.extend({"role": message["role"], "content": message["content"]} for message in messages)
    if image_data:
        for index in range(len(chat) - 1, -1, -1):
            if chat[index]["role"] == "user":
                chat[index]["images"] = [image_data]
                break
    return chat


class NativeDriver(NativeRuntime):
    async def run(
        self,
        messages: List[Dict[str, Any]],
        config: Any,
        system_message: Optional[str],
        image_data: Optional[str] = None,
        options_override: Optional[Dict[str, Any]] = None,
        stats: Optional[RunStats] = None,
    ) -> AsyncIterator[str]:
        stats = stats if stats is not None else RunStats()
        async with aclosing(self.stream(
            config,
            chat_for(messages, system_message, image_data),
            sampling_for(config, options_override),
            requested_thinking(Settings(), config),
            stats,
        )) as texts:
            async for text in texts:
                yield text

    async def stream_with_history(self, messages, config, system_message, image_data=None, options_override=None):
        stats = RunStats()
        async with aclosing(self.run(messages, config, system_message, image_data, options_override, stats)) as texts:
            async for text in texts:
                yield {"type": "token", "content": text}
        yield {
            "type": "usage",
            "thinking_mode": stats.thinking_mode,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "tokens_used": stats.prompt_tokens + stats.completion_tokens,
            "finish": stats.finish,
        }

    async def generate_with_history(self, messages, config, system_message, image_data=None, options_override=None):
        stats = RunStats()
        pieces: List[str] = []
        async with aclosing(self.run(messages, config, system_message, image_data, options_override, stats)) as texts:
            async for text in texts:
                pieces.append(text)
        return SimpleNamespace(
            content="".join(pieces),
            model=config.model,
            provider_id="native",
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            tokens_used=stats.prompt_tokens + stats.completion_tokens,
            finish=stats.finish,
            thinking_mode=stats.thinking_mode,
        )

    def messages_token_counter(self, config):
        count = self.prompt_token_counter(config, requested_thinking(Settings(), config))
        if count is None:
            return None
        return lambda system_message, messages, tools=None: count(chat_for(messages, system_message, None))

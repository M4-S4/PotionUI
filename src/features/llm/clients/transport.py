import time
from typing import Any, Dict, List, Optional

import httpx

from src.features.llm import trace_collector
from src.features.llm.repository import LLMConfig


def stream_timeout(config: LLMConfig) -> httpx.Timeout:
    return httpx.Timeout(
        connect=config.timeout, read=None, write=config.timeout, pool=config.timeout
    )


def record_trace(
    provider: str,
    config: LLMConfig,
    started: float,
    *,
    request_system: Optional[str],
    request_messages: List[Dict[str, Any]],
    request_params: Dict[str, Any],
    request_tools: Optional[Any],
    response_text: str,
    response_tool_calls: Optional[Any],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> None:
    trace_collector.record(
        provider=provider,
        model=config.model,
        request_system=request_system,
        request_messages=request_messages,
        request_params=request_params,
        request_tools=request_tools,
        response_text=response_text,
        response_tool_calls=response_tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_ms=int((time.monotonic() - started) * 1000),
    )

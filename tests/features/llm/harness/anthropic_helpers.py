import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.features.llm.harness import codec
from src.features.llm.harness.adapter import RenderRequest, Settings, ToolSpec
from src.features.llm.harness.adapters.anthropic_messages import AnthropicMessagesAdapter, target_origin
from src.features.llm.harness.assembler import StepAssembler
from src.features.llm.harness.messages import (
    ContextPart,
    ImagePart,
    Message,
    Origin,
    TextPart,
    ToolResultPart,
    message,
)
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.repository import LLMConfig
from tests.features.llm.harness import cassettes
from tests.fixtures.llm_harness import decode_chunks

ADAPTER_DIR = "anthropic-messages"
MODEL = "claude-sonnet-5-5"
BASE_URL = "https://api.anthropic.com"
API_KEY = "cassette-key"


def scenario_path(name: str) -> Path:
    return cassettes.CASSETTE_ROOT / ADAPTER_DIR / name


def step_dirs(name: str) -> List[Path]:
    return cassettes.step_dirs(scenario_path(name))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def chunks_of(directory: Path) -> List[bytes]:
    return cassettes.load_step(directory).chunks


PROFILE_FIELDS: Dict[str, Any] = {
    "context_window": 1000000,
    "max_output_tokens": 64000,
    "parallel_tools": True,
    "tool_choice": ["auto", "none", "required", "named"],
    "tool_choice_with_reasoning": ["auto", "none"],
    "reasoning": "effort",
    "reasoning_levels": ["low", "medium", "high", "max"],
    "replay": "send_all",
    "interleaved_reasoning": True,
    "vision": True,
    "max_images": 100,
    "tool_result_media": True,
    "cache": "explicit",
    "cache_min_tokens": 512,
}

WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
CLOCK = ToolSpec(
    name="get_time",
    description="Current local time for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


def make_profile(**overrides: Any) -> CapabilityProfile:
    fields = {**PROFILE_FIELDS, **overrides}
    return CapabilityProfile(adapter="anthropic-messages", model=MODEL).layered(
        [ProfileLayer(ProfileSource.OVERRIDE, fields)]
    )


def bare_profile(**fields: Any) -> CapabilityProfile:
    profile = CapabilityProfile(adapter="anthropic-messages", model=MODEL)
    return profile.layered([ProfileLayer(ProfileSource.OVERRIDE, fields)]) if fields else profile


def make_config(**fields: Any) -> LLMConfig:
    values = {
        "id": "cassette",
        "name": "cassette",
        "type": "anthropic",
        "enabled": True,
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "model": MODEL,
        "system_message": "",
        **fields,
    }
    return LLMConfig(**values)


def make_request(
    messages: Sequence[Message],
    *,
    system: str = "",
    tools: Sequence[ToolSpec] = (),
    settings: Settings = Settings(),
    **config: Any,
) -> RenderRequest:
    return RenderRequest(
        config=make_config(**config), system=system, messages=tuple(messages), tools=tuple(tools), settings=settings
    )


def own_origin(**config: Any) -> Origin:
    return target_origin(make_config(**config))


def render_body(request: RenderRequest, profile: Optional[CapabilityProfile] = None) -> Dict[str, Any]:
    return AnthropicMessagesAdapter().render(request, profile or make_profile()).body


def sse_bytes(events: Sequence[Tuple[str, Mapping[str, Any]]]) -> bytes:
    return "".join(
        f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n" for name, data in events
    ).encode("utf-8")


def stream(*blocks: List[Any], stop: str = "end_turn", usage: Optional[Dict[str, Any]] = None, extra: Sequence[Any] = ()) -> bytes:
    start = {"type": "message_start", "message": {"id": "msg_1", "usage": usage or {"input_tokens": 9, "output_tokens": 1}}}
    events: List[Any] = [("message_start", start)]
    for events_of_block in blocks:
        events.extend(events_of_block)
    events.extend(extra)
    events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 5}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return sse_bytes(events)


def block_start(index: int, block: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    return ("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})


def block_delta(index: int, delta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    return ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})


def block_stop(index: int) -> Tuple[str, Dict[str, Any]]:
    return ("content_block_stop", {"type": "content_block_stop", "index": index})


def text_block(index: int, *parts: str) -> List[Any]:
    return [
        block_start(index, {"type": "text", "text": ""}),
        *(block_delta(index, {"type": "text_delta", "text": part}) for part in parts),
        block_stop(index),
    ]


def tool_block(index: int, ident: str, name: str, *parts: str) -> List[Any]:
    return [
        block_start(index, {"type": "tool_use", "id": ident, "name": name, "input": {}}),
        *(block_delta(index, {"type": "input_json_delta", "partial_json": part}) for part in parts),
        block_stop(index),
    ]


def thinking_block(index: int, text: str, signature: str) -> List[Any]:
    return [
        block_start(index, {"type": "thinking", "thinking": ""}),
        block_delta(index, {"type": "thinking_delta", "thinking": text}),
        block_delta(index, {"type": "signature_delta", "signature": signature}),
        block_stop(index),
    ]


def redacted_block(index: int, data: str) -> List[Any]:
    return [block_start(index, {"type": "redacted_thinking", "data": data}), block_stop(index)]


def synthetic_history() -> Tuple[List[Message], RenderRequest]:
    adapter, profile = AnthropicMessagesAdapter(), make_profile()
    settings = Settings(max_tokens=8192, reasoning="medium")
    history: List[Message] = [message("u1", "user", TextPart("What is the weather in Paris and Tokyo, and what time is it there?"))]

    def request_for(messages: Sequence[Message]) -> RenderRequest:
        return make_request(messages, system="You are a travel assistant.", tools=[WEATHER, CLOCK], settings=settings)

    replies = [
        stream(
            thinking_block(0, "Both lookups are independent.", "sig-1"),
            redacted_block(1, "red-1"),
            text_block(2, "Checking both cities."),
            tool_block(3, "toolu_paris", "get_weather", "", '{"city": ', '"Paris"}'),
            tool_block(4, "toolu_tokyo", "get_weather", "", '{"city": "Tokyo"}'),
            stop="tool_use",
        ),
        stream(thinking_block(0, "Now the time.", "sig-2"), tool_block(1, "toolu_time", "get_time", '{"city": "Tokyo"}'), stop="tool_use"),
        stream(
            [block_start(0, {"type": "thinking", "thinking": ""}), block_delta(0, {"type": "signature_delta", "signature": "sig-3"}), block_stop(0)],
            text_block(1, "Paris is overcast, Tokyo is clear and it is 21:42 there."),
        ),
    ]
    follow_ups: List[List[Message]] = [
        [
            message(
                "t1",
                "tool",
                ToolResultPart("toolu_paris", "get_weather", (TextPart('{"temp_c": 14}'),)),
                ToolResultPart("toolu_tokyo", "get_weather", (TextPart('{"temp_c": 22}'), ImagePart(media_type="image/png", data="AAAA"))),
            )
        ],
        [
            message("t2", "tool", ToolResultPart("toolu_time", "get_time", (TextPart('{"local_time": "21:42"}'),))),
            message("u2", "user", ContextPart(kind="turn_note", text="Two of four tool rounds are used.")),
        ],
        [],
    ]
    for number, (reply, follow_up) in enumerate(zip(replies, follow_ups), start=1):
        request = request_for(history)
        events = decode_chunks(adapter.decoder(request, profile), [reply])
        assembler = StepAssembler(f"a{number}", target_origin(request.config))
        assembler.feed_all(events)
        history = history + [assembler.message()] + follow_up
    return history, request_for(history)

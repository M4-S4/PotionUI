import json
from typing import Any, Dict, List, Optional, Sequence

from src.features.llm.harness.adapter import RenderRequest, Settings
from src.features.llm.harness.adapters.openai_responses import OpenAIResponsesAdapter
from src.features.llm.harness.messages import Message, Origin, Role, TextPart
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.repository import LLMConfig
from tests.fixtures.llm_harness import assemble, decode_chunks

ADAPTER = OpenAIResponsesAdapter()
MODEL = "gpt-5.6"
REASONING_PROFILE: Dict[str, Any] = {
    "reasoning": "effort",
    "reasoning_levels": ["none", "minimal", "low", "medium", "high"],
    "parallel_tools": True,
    "strict_tools": True,
    "tool_choice": ["auto", "none", "required", "named"],
    "tool_choice_with_reasoning": ["auto"],
    "sampling_rejected": ["top_p"],
    "sampling_ignored_with_reasoning": ["temperature"],
    "tool_result_media": True,
}


def make_config(**fields: Any) -> LLMConfig:
    values: Dict[str, Any] = {
        "id": "cfg_1",
        "name": "OpenAI",
        "type": "openai",
        "enabled": True,
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key-a",
        "model": MODEL,
        "system_message": "",
        "show_thinking": False,
    }
    values.update(fields)
    return LLMConfig(**values)


def make_profile(**fields: Any) -> CapabilityProfile:
    return CapabilityProfile(adapter="openai-responses", model=MODEL).layered(
        [ProfileLayer(ProfileSource.OVERRIDE, fields)] if fields else []
    )


def make_request(
    messages: Sequence[Message] = (),
    *,
    config: Optional[LLMConfig] = None,
    tools: Sequence[Any] = (),
    settings: Optional[Settings] = None,
    system: str = "Be brief.",
    stream: bool = True,
) -> RenderRequest:
    return RenderRequest(
        config=config or make_config(),
        system=system,
        messages=tuple(messages),
        tools=tuple(tools),
        settings=settings or Settings(),
        stream=stream,
    )


def user_message(text: str = "Hello", message_id: str = "user_1") -> Message:
    return Message(id=message_id, role=Role.USER, parts=(TextPart(text),))


def sse(*events: Dict[str, Any], newline: str = "\n") -> bytes:
    frames = []
    for number, event in enumerate(events):
        payload = {"sequence_number": number, **event}
        frames.append(f"event: {event['type']}{newline}data: {json.dumps(payload, ensure_ascii=False)}{newline}{newline}")
    return "".join(frames).encode("utf-8")


def reasoning_item(item_id: str = "rs_1", encrypted: Optional[str] = "enc-1", summaries: Sequence[str] = ()) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "id": item_id,
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text} for text in summaries],
    }
    if encrypted is not None:
        item["encrypted_content"] = encrypted
    return item


def message_item(text: str = "Done.", phase: Optional[str] = "final_answer", item_id: str = "msg_1") -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "content": [{"type": "output_text", "annotations": [], "text": text}],
        "role": "assistant",
    }
    if phase is not None:
        item["phase"] = phase
    return item


def call_item(
    call_id: str = "call_1", name: str = "get_weather", arguments: str = '{"city":"Paris"}', item_id: str = "fc_1",
) -> Dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call",
        "status": "completed",
        "arguments": arguments,
        "call_id": call_id,
        "name": name,
    }


def completed(output: List[Dict[str, Any]], usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {"id": "resp_1", "status": "completed", "output": output, "usage": usage},
    }


def item_events(index: int, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"type": "response.output_item.added", "output_index": index, "item": item},
        {"type": "response.output_item.done", "output_index": index, "item": item},
    ]


def stream_of(output: List[Dict[str, Any]], usage: Optional[Dict[str, Any]] = None) -> bytes:
    events: List[Dict[str, Any]] = [{"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}]
    for index, item in enumerate(output):
        events.extend(item_events(index, item))
    events.append(completed(output, usage))
    return sse(*events)


def decode_all(request: RenderRequest, data: bytes, profile: Optional[CapabilityProfile] = None) -> list:
    decoder = ADAPTER.decoder(request, profile or make_profile())
    return decode_chunks(decoder, [data])


def assistant_message(
    output: List[Dict[str, Any]], config: Optional[LLMConfig] = None, message_id: str = "asst_1",
) -> Message:
    config = config or make_config()
    request = make_request([user_message()], config=config)
    events = decode_all(request, stream_of(output))
    return assemble(events, message_id, Origin.for_config(ADAPTER.name, config))

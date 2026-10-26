import json
from typing import Dict, Optional

from src.features.llm.harness.errors import InvalidRequest
from src.features.llm.harness.messages import ImagePart, JsonPart, Message, ReasoningPart, TextPart, ToolResultPart


def bearer_headers(api_key: Optional[str]) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def image_data(part: ImagePart) -> str:
    if part.data is None:
        raise InvalidRequest(f"an image reference was not resolved before rendering: {part.ref}")
    return part.data


def image_data_uri(part: ImagePart) -> str:
    return f"data:{part.media_type};base64,{image_data(part)}"


def result_text(result: ToolResultPart) -> str:
    pieces = []
    for item in result.content:
        if isinstance(item, TextPart):
            pieces.append(item.text)
        elif isinstance(item, JsonPart):
            pieces.append(json.dumps(item.value))
    return "".join(pieces)


def reasoning_text(message: Message) -> str:
    return "".join(part.text for part in message.parts if isinstance(part, ReasoningPart) and not part.redacted)

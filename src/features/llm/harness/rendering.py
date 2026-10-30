from typing import Dict, Optional, Sequence

from src.features.llm.harness.errors import InvalidRequest
from src.features.llm.harness.messages import ImagePart, Message, ReasoningPart, Role


def bearer_headers(api_key: Optional[str]) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def image_data(part: ImagePart) -> str:
    if part.data is None:
        raise InvalidRequest(f"an image reference was not resolved before rendering: {part.ref}")
    return part.data


def image_data_uri(part: ImagePart) -> str:
    return f"data:{part.media_type};base64,{image_data(part)}"


def selected_image(messages: Sequence[Message]) -> Optional[ImagePart]:
    if messages and messages[-1].role is Role.TOOL:
        for result in reversed(messages[-1].tool_results()):
            for item in reversed(result.content):
                if isinstance(item, ImagePart):
                    return item
    for message in reversed(messages):
        if message.role is Role.USER:
            for part in reversed(message.parts):
                if isinstance(part, ImagePart):
                    return part
    return None


def last_user_index(rendered: Sequence[Dict[str, object]]) -> Optional[int]:
    for index in range(len(rendered) - 1, -1, -1):
        if rendered[index]["role"] == "user":
            return index
    return None


def reasoning_text(message: Message) -> str:
    return "".join(part.text for part in message.parts if isinstance(part, ReasoningPart) and not part.redacted)

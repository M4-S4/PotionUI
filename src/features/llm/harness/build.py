import uuid
from typing import Iterable, Optional, Sequence

from src.features.llm.harness.messages import ContextPart, ImagePart, Message, Role, TextPart

JPEG = "image/jpeg"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def user_message(
    text: str,
    *,
    images: Iterable[str] = (),
    contexts: Sequence[ContextPart] = (),
    message_id: Optional[str] = None,
) -> Message:
    parts = [TextPart(text=text), *(ImagePart(media_type=JPEG, data=image) for image in images if image), *contexts]
    return Message(id=message_id or new_id("msg"), role=Role.USER, parts=tuple(parts))


def context_message(kind: str, text: str) -> Message:
    return Message(
        id=new_id("msg"),
        role=Role.USER,
        parts=(ContextPart(kind=kind, text=f"<context>\n{text}\n</context>"),),
    )


def with_context(message: Message, kind: str, text: str) -> Message:
    return Message(
        id=message.id,
        role=message.role,
        parts=(*message.parts, ContextPart(kind=kind, text=text)),
        origin=message.origin,
        raw=message.raw,
    )


def attach_context(messages: list, kind: str, text: str) -> None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role is Role.USER:
            messages[index] = with_context(messages[index], kind, text)
            return
    messages.append(Message(id=new_id("msg"), role=Role.USER, parts=(ContextPart(kind=kind, text=text),)))

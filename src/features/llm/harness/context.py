from src.features.llm.harness.messages import ContextPart, Message, TextPart


def wrap_context(text: str) -> str:
    return f"\n\n<context>\n{text}\n</context>"


def user_text(message: Message) -> str:
    has_text = any(isinstance(part, TextPart) for part in message.parts)
    rendered = ""
    first_context = True
    for part in message.parts:
        if isinstance(part, TextPart):
            rendered += part.text
        elif isinstance(part, ContextPart):
            rendered += part.text if first_context and not has_text else wrap_context(part.text)
            first_context = False
    return rendered

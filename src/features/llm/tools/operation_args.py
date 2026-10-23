from collections import Counter
from typing import Any, Iterable, List, Optional, Tuple

from src.features.llm.tools.tool_call_rescue import decode_payload, demangle_quote_tokens


def _coerce(operations: Any, call_example: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    if isinstance(operations, dict):
        return [operations], None
    if isinstance(operations, list):
        return operations, None
    if isinstance(operations, str):
        decoded, problem = decode_payload(demangle_quote_tokens(operations))
        if isinstance(decoded, list):
            return decoded, None
        if isinstance(decoded, dict):
            return [decoded], None
        return None, (
            "'operations' must be an ARRAY of operation objects, not text. "
            + (f"The text you sent could not be read: {problem}. " if problem else "")
            + f"Send the whole call as JSON with double quotes: {call_example}"
        )
    return None, (
        f"'operations' must be an array of operation objects, got {type(operations).__name__}. "
        f"The whole call looks like this: {call_example}"
    )


def operations_from(raw: Any, call_example: str, empty_error: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    if not raw:
        return None, empty_error
    operations, problem = _coerce(raw, call_example)
    if problem:
        return None, problem
    return (operations, None) if operations else (None, empty_error)


def collapse_repeats(messages: Iterable[str]) -> List[str]:
    counts = Counter(messages)
    return [message if count == 1 else f"{message} (×{count})" for message, count in counts.items()]

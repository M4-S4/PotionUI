import hashlib
from typing import Dict, Iterable

from src.features.llm.harness.messages import Message
from src.features.llm.harness.profile import ToolIdRule


def mint_call_id(seed: str, index: int) -> str:
    return f"call_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:8]}_{index}"


def nine_digit_id(call_id: str) -> str:
    return str(int(hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:12], 16) % 10**9).zfill(9)


def seed_for(messages: Iterable[Message]) -> str:
    last = ""
    for message in messages:
        last = message.id
    return last


def wire_ids(messages: Iterable[Message], rule: ToolIdRule) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for message in messages:
        for call in message.tool_calls():
            if rule is ToolIdRule.NINE_DIGITS:
                mapping[call.call_id] = nine_digit_id(call.call_id)
            else:
                mapping[call.call_id] = call.wire_id or call.call_id
    return mapping

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.features.llm.harness.adapter import ToolSpec
from src.features.llm.harness.assembler import parse_args
from src.features.llm.harness.messages import ToolCallPart

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"

_MANGLED_QUOTE = re.compile(r"<\|(['\"])\|>")


def demangle_quote_tokens(text: str) -> str:
    return _MANGLED_QUOTE.sub(r"\1", text)


@lru_cache(maxsize=16)
def _tools_text(key: Tuple[Tuple[str, str, str], ...]) -> str:
    lines = [
        "\n\n## Available Tools\n",
        "You have the following tools available. To call a tool, output a "
        "`<tool_call>` XML block with a JSON body containing `name` and `arguments`.\n",
        "Example:",
        '<tool_call>{"name": "tool_name", "arguments": {"arg1": "value1"}}</tool_call>\n',
        "You may call multiple tools in a single response. "
        "After each tool call you will receive the result in the next message. "
        "When you have enough information, respond normally without any tool_call blocks.\n",
        "### Tools\n",
    ]
    for name, description, parameters in key:
        lines.append(f"**{name}**: {description}")
        lines.append(f"  Parameters: {parameters}")
        lines.append("")
    return "\n".join(lines)


def render_tools_text(tools: Sequence[ToolSpec]) -> str:
    return _tools_text(tuple(
        (tool.name, tool.description, json.dumps(tool.parameters, sort_keys=True, separators=(",", ":")))
        for tool in tools
    ))


def placeholder_value(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return "..."
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    for union_key in ("anyOf", "oneOf"):
        branches = schema.get(union_key)
        if isinstance(branches, list) and branches:
            return placeholder_value(branches[0])
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), None)
    if schema_type in ("number", "integer"):
        value = schema.get("minimum", 0)
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            value = maximum
        return value
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        item_schema = schema.get("items") or {}
        return [placeholder_value(item_schema) for _ in range(schema.get("minItems") or 0)]
    if schema_type == "object":
        properties = schema.get("properties") or {}
        return {key: placeholder_value(properties[key]) for key in (schema.get("required") or []) if key in properties}
    placeholder = "..."
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and min_length > len(placeholder):
        placeholder = placeholder.ljust(min_length, ".")
    return placeholder


def example_tool_call(tools: Sequence[ToolSpec]) -> str:
    named = [(tool.name, tool.parameters or {}) for tool in tools if tool.name]
    for name, params in named:
        if not (params.get("required") or []):
            return f"{OPEN_TAG}{json.dumps({'name': name, 'arguments': {}})}{CLOSE_TAG}"
    for name, params in named:
        required = params.get("required") or []
        properties = params.get("properties") or {}
        if required and all(key in properties for key in required):
            arguments = {key: placeholder_value(properties[key]) for key in required}
            return f"{OPEN_TAG}{json.dumps({'name': name, 'arguments': arguments})}{CLOSE_TAG}"
    return '<tool_call>{"name": "<tool name>", "arguments": {...}}</tool_call>'


def render_inline_tools_text(tools: Sequence[ToolSpec]) -> str:
    if not tools:
        return ""
    lines = [
        "\n\nYou have access to the following tools. To call one, respond with "
        "EXACTLY one <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call> "
        f"block and nothing else — for example {example_tool_call(tools)}. "
        "Do not wrap a call in a <tool_action> tag or a code fence; only a "
        "<tool_call> block runs.",
    ]
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.description} (arguments schema: {json.dumps(tool.parameters)})")
    return "\n".join(lines)


def call_block(call: ToolCallPart) -> str:
    body = json.dumps({"name": call.name, "arguments": call.args if call.args is not None else {}})
    return f"{OPEN_TAG}{body}{CLOSE_TAG}"


def assistant_text(text: str, calls: Sequence[ToolCallPart]) -> str:
    parts = [text] if text else []
    parts.extend(call_block(call) for call in calls)
    return "\n".join(parts)


def result_text(name: str, content: str) -> str:
    return f"[Tool Result: {name or 'tool'}]\n{content}"


@dataclass(frozen=True)
class TextItem:
    text: str


@dataclass(frozen=True)
class CallItem:
    name: str
    args: Optional[Dict[str, Any]]
    raw: str


Item = Union[TextItem, CallItem]


def parse_call_block(block: str) -> Optional[CallItem]:
    raw = block[len(OPEN_TAG):-len(CLOSE_TAG)].strip()
    try:
        data = json.loads(raw)
    except ValueError:
        try:
            data = json.loads(demangle_quote_tokens(raw))
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    function = data.get("function")
    source = function if isinstance(function, dict) else data
    name = source.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = source.get("arguments") or source.get("parameters") or {}
    args = arguments if isinstance(arguments, dict) else parse_args(arguments) if isinstance(arguments, str) else None
    return CallItem(name=name, args=args, raw=arguments if isinstance(arguments, str) else json.dumps(arguments))


class PromptToolStage:
    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, text: str) -> List[Item]:
        self._buffer += text
        items: List[Item] = []
        while True:
            if not self._inside:
                start = self._buffer.find(OPEN_TAG)
                if start == -1:
                    keep = _partial_suffix(self._buffer)
                    cut = len(self._buffer) - keep
                    if cut:
                        items.append(TextItem(self._buffer[:cut]))
                        self._buffer = self._buffer[cut:]
                    break
                if start:
                    items.append(TextItem(self._buffer[:start]))
                self._buffer = self._buffer[start:]
                self._inside = True
            end = self._buffer.find(CLOSE_TAG)
            if end == -1:
                break
            end += len(CLOSE_TAG)
            block, self._buffer = self._buffer[:end], self._buffer[end:]
            self._inside = False
            call = parse_call_block(block)
            items.append(call if call is not None else TextItem(block))
        return items

    def finish(self) -> List[Item]:
        rest, self._buffer = self._buffer, ""
        self._inside = False
        return [TextItem(rest)] if rest else []


def _partial_suffix(buffer: str) -> int:
    for size in range(min(len(buffer), len(OPEN_TAG) - 1), 0, -1):
        if OPEN_TAG.startswith(buffer[-size:]):
            return size
    return 0

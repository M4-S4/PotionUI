import json
import re
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.features.llm.harness.adapter import Settings, ToolChoice, ToolSpec
from src.features.llm.harness.errors import HarnessError
from src.features.llm.harness.messages import Message, Role, TextPart, ToolResultPart
from src.features.llm.harness.profile import CapabilityProfile, StructuredOutputMode
from src.features.llm.harness.tool_choice import NAMED, effective_tool_choice
from src.features.llm.harness.turn_state import Notice, NoticeKind

STRUCTURED_TOOL_NAME = "submit_result"
STRUCTURED_TOOL_DESCRIPTION = "Submit the final result. Call this tool exactly once with the complete result."
WRAP_KEY = "value"
MAX_REF_HOPS = 32
DEFINITION_KEYS = ("$defs", "definitions")

Complete = Callable[[str, Tuple[Message, ...], Settings, Tuple[ToolSpec, ...]], Awaitable[Message]]

FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
OPENER = re.compile(r"[\[{]")

TYPE_CHECKS: Dict[str, Callable[[Any], bool]] = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
    "null": lambda value: value is None,
}


class StructuredOutputFailed(HarnessError):
    kind = "structured_output_invalid"


@dataclass(frozen=True)
class StructuredPlan:
    mode: StructuredOutputMode
    system: str
    tools: Tuple[ToolSpec, ...]
    settings: Settings
    schema: Mapping[str, Any]
    wrapped: bool
    notices: Tuple[Notice, ...] = ()


@dataclass(frozen=True)
class Parsed:
    value: Any = None
    errors: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class StructuredResult:
    value: Any
    message: Message
    mode: StructuredOutputMode
    repaired: bool
    notices: Tuple[Notice, ...]


def wrap_schema(schema: Mapping[str, Any]) -> Dict[str, Any]:
    wrapper: Dict[str, Any] = {
        "type": "object",
        "properties": {WRAP_KEY: {key: item for key, item in schema.items() if key not in DEFINITION_KEYS}},
        "required": [WRAP_KEY],
        "additionalProperties": False,
    }
    wrapper.update({key: schema[key] for key in DEFINITION_KEYS if key in schema})
    return wrapper


def prompted_instruction(schema: Mapping[str, Any]) -> str:
    rendered = json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return (
        "Reply with one JSON value that matches the JSON Schema below and nothing else: "
        f"no code fence and no commentary.\nSchema: {rendered}"
    )


def plan_structured(system: str, settings: Settings, profile: CapabilityProfile) -> StructuredPlan:
    schema = settings.response_schema
    if not isinstance(schema, Mapping):
        raise ValueError("response_schema must be a JSON Schema object")
    mode = profile.structured_output
    notices: Tuple[Notice, ...] = ()
    forced = ToolChoice(NAMED, STRUCTURED_TOOL_NAME)
    if mode is StructuredOutputMode.TOOL:
        granted = effective_tool_choice(replace(settings, tool_choice=forced, response_schema=None), profile).choice
        if not profile.tools or granted != forced:
            mode = StructuredOutputMode.PROMPTED
            notices = (Notice(
                kind=NoticeKind.DEGRADED_RETRY,
                message="This model cannot return structured output through a tool call, so the format is requested in the prompt instead.",
                detail={"reason": "structured_output_fallback", "requested": "tool", "used": "prompted"},
            ),)
    if mode is StructuredOutputMode.PROMPTED:
        suffix = prompted_instruction(schema)
        return StructuredPlan(
            mode=mode,
            system=f"{system}\n\n{suffix}" if system else suffix,
            tools=(),
            settings=replace(settings, response_schema=None),
            schema=schema,
            wrapped=False,
            notices=notices,
        )
    wrapped = schema.get("type") != "object"
    wire = wrap_schema(schema) if wrapped else dict(schema)
    if mode is StructuredOutputMode.TOOL:
        tool = ToolSpec(STRUCTURED_TOOL_NAME, STRUCTURED_TOOL_DESCRIPTION, wire, strict=profile.strict_tools)
        return StructuredPlan(
            mode=mode,
            system=system,
            tools=(tool,),
            settings=replace(settings, response_schema=None, tool_choice=forced),
            schema=schema,
            wrapped=wrapped,
            notices=notices,
        )
    return StructuredPlan(
        mode=mode,
        system=system,
        tools=(),
        settings=replace(settings, response_schema=wire),
        schema=schema,
        wrapped=wrapped,
        notices=notices,
    )


def extract_json(text: str) -> Tuple[Any, Optional[str]]:
    stripped = text.strip()
    if not stripped:
        return None, "the reply was empty"
    candidates = [stripped]
    fenced = FENCE.search(stripped)
    if fenced:
        candidates.append(fenced.group(1))
    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except ValueError:
            pass
        opener = OPENER.search(candidate)
        if opener:
            try:
                value, _ = json.JSONDecoder().raw_decode(candidate[opener.start():])
                return value, None
            except ValueError:
                pass
    return None, "the reply was not valid JSON"


def parse_structured(message: Message, plan: StructuredPlan) -> Parsed:
    if plan.mode is StructuredOutputMode.TOOL:
        call = next((item for item in message.tool_calls() if item.name == STRUCTURED_TOOL_NAME), None)
        if call is None:
            return Parsed(errors=(f"the {STRUCTURED_TOOL_NAME} tool was not called",))
        if call.args is None:
            return Parsed(errors=("the tool arguments were not valid JSON",))
        value: Any = call.args
    else:
        value, problem = extract_json(message.text())
        if problem:
            return Parsed(errors=(problem,))
    if plan.wrapped:
        if not isinstance(value, dict) or WRAP_KEY not in value:
            return Parsed(errors=(f"the result was not wrapped in an object with a {WRAP_KEY!r} key",))
        value = value[WRAP_KEY]
    errors = validate(value, plan.schema)
    return Parsed(None if errors else value, tuple(errors))


def repair_messages(reply: Message, errors: Sequence[str], plan: StructuredPlan) -> Tuple[Message, ...]:
    problems = "; ".join(errors)
    history: Tuple[Message, ...] = (reply,) if reply.parts else ()
    calls = reply.tool_calls()
    if calls:
        feedback = f"The call was not accepted: {problems}. Call {STRUCTURED_TOOL_NAME} again with a corrected result."
        results = tuple(
            ToolResultPart(call_id=call.call_id, name=call.name, content=(TextPart(feedback),), is_error=True)
            for call in calls
        )
        return history + (Message(id=f"{reply.id}:repair", role=Role.TOOL, parts=results),)
    feedback = f"That reply was not accepted: {problems}. Reply again with only the corrected JSON."
    return history + (Message(id=f"{reply.id}:repair", role=Role.USER, parts=(TextPart(feedback),)),)


async def run_structured(
    complete: Complete,
    *,
    system: str,
    messages: Sequence[Message],
    settings: Settings,
    profile: CapabilityProfile,
) -> StructuredResult:
    plan = plan_structured(system, settings, profile)
    history = tuple(messages)
    reply = await complete(plan.system, history, plan.settings, plan.tools)
    parsed = parse_structured(reply, plan)
    if parsed.ok:
        return StructuredResult(parsed.value, reply, plan.mode, False, plan.notices)
    history += repair_messages(reply, parsed.errors, plan)
    repair = Notice(
        kind=NoticeKind.DEGRADED_RETRY,
        message="The reply did not match the requested format, so the model was asked again.",
        detail={"reason": "structured_output_repair", "errors": list(parsed.errors)},
    )
    retry = await complete(plan.system, history, plan.settings, plan.tools)
    parsed = parse_structured(retry, plan)
    if parsed.ok:
        return StructuredResult(parsed.value, retry, plan.mode, True, plan.notices + (repair,))
    raise StructuredOutputFailed(
        f"The model did not return valid structured output: {'; '.join(parsed.errors)}",
        detail={"mode": plan.mode.value, "errors": list(parsed.errors)},
    )


def validate(value: Any, schema: Any, path: str = "$", root: Any = None) -> List[str]:
    root = schema if root is None else root
    schema = _resolve(schema, root)
    if not isinstance(schema, Mapping):
        return []
    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else list(types)
        if not any(TYPE_CHECKS.get(name, lambda _: True)(value) for name in names):
            return [f"{path}: expected {' or '.join(names)}, got {_kind(value)}"]
    errors: List[str] = []
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: must equal {json.dumps(schema['const'])}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {json.dumps(schema['enum'])}")
    for part in schema.get("allOf", ()):
        errors += validate(value, part, path, root)
    if "anyOf" in schema and not any(not validate(value, part, path, root) for part in schema["anyOf"]):
        errors.append(f"{path}: matches none of the allowed shapes")
    if "oneOf" in schema and sum(not validate(value, part, path, root) for part in schema["oneOf"]) != 1:
        errors.append(f"{path}: must match exactly one of the allowed shapes")
    if isinstance(value, str):
        errors += _check_string(value, schema, path)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        errors += _check_number(value, schema, path)
    elif isinstance(value, list):
        errors += _check_array(value, schema, path, root)
    elif isinstance(value, dict):
        errors += _check_object(value, schema, path, root)
    return errors


def _kind(value: Any) -> str:
    for name in ("boolean", "integer", "number", "string", "array", "object", "null"):
        if TYPE_CHECKS[name](value):
            return name
    return type(value).__name__


def _resolve(schema: Any, root: Any) -> Any:
    for _ in range(MAX_REF_HOPS):
        if not isinstance(schema, Mapping) or "$ref" not in schema:
            return schema
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#"):
            raise ValueError(f"unsupported schema reference {ref!r}")
        node = root
        for token in (item for item in ref[1:].split("/") if item):
            node = node[token.replace("~1", "/").replace("~0", "~")]
        schema = node
    raise ValueError("schema references loop")


def _check_string(value: str, schema: Mapping[str, Any], path: str) -> List[str]:
    errors: List[str] = []
    if "minLength" in schema and len(value) < schema["minLength"]:
        errors.append(f"{path}: must have at least {schema['minLength']} characters")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(f"{path}: must have at most {schema['maxLength']} characters")
    if "pattern" in schema and not re.search(schema["pattern"], value):
        errors.append(f"{path}: must match the pattern {schema['pattern']}")
    return errors


def _check_number(value: float, schema: Mapping[str, Any], path: str) -> List[str]:
    errors: List[str] = []
    if "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{path}: must be at least {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        errors.append(f"{path}: must be at most {schema['maximum']}")
    return errors


def _check_array(value: list, schema: Mapping[str, Any], path: str, root: Any) -> List[str]:
    errors: List[str] = []
    if "minItems" in schema and len(value) < schema["minItems"]:
        errors.append(f"{path}: must have at least {schema['minItems']} items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        errors.append(f"{path}: must have at most {schema['maxItems']} items")
    if isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{index}]", root)
    return errors


def _check_object(value: dict, schema: Mapping[str, Any], path: str, root: Any) -> List[str]:
    errors: List[str] = []
    properties = schema.get("properties") or {}
    for name in schema.get("required", ()):
        if name not in value:
            errors.append(f"{path}: missing required property {name!r}")
    for name, item in value.items():
        if name in properties:
            errors += validate(item, properties[name], f"{path}.{name}", root)
        elif schema.get("additionalProperties") is False:
            errors.append(f"{path}: unexpected property {name!r}")
        elif isinstance(schema.get("additionalProperties"), Mapping):
            errors += validate(item, schema["additionalProperties"], f"{path}.{name}", root)
    return errors

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.features.llm.harness import codec
from src.features.llm.harness.adapter import RenderRequest, Settings, ToolSpec
from src.features.llm.harness.adapters.anthropic_messages import AnthropicMessagesAdapter, target_origin
from src.features.llm.harness.messages import Message, Origin
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.repository import LLMConfig

CASSETTES = Path(__file__).resolve().parents[3] / "fixtures" / "llm_cassettes" / "anthropic-messages"
MODEL = "claude-sonnet-5-5"
BASE_URL = "https://api.anthropic.com"
API_KEY = "cassette-key"


def scenario_path(name: str) -> Path:
    return CASSETTES / name


def step_dirs(name: str) -> List[Path]:
    return sorted(path for path in scenario_path(name).iterdir() if path.is_dir())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def chunks_of(directory: Path) -> List[bytes]:
    lines = (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return [base64.b64decode(json.loads(line)["b64"]) for line in lines if line.strip()]


def cassette_profile_fields() -> Dict[str, Any]:
    return read_json(step_dirs("thinking_parallel_tools")[0] / "meta.json")["profile_overrides"]


def make_profile(**overrides: Any) -> CapabilityProfile:
    fields = {**cassette_profile_fields(), **overrides}
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


def stored_history(name: str) -> Tuple[List[Message], RenderRequest]:
    history: List[Message] = []
    carried: Dict[str, Any] = {}
    for directory in step_dirs(name):
        step_input = read_json(directory / "input.json")
        carried = {**carried, **{key: value for key, value in step_input.items() if key != "messages"}}
        history += codec.from_json(List[Message], step_input["messages"])
        step = directory / "step.json"
        if step.exists():
            history.append(codec.from_json(Message, read_json(step)["message"]))
    request = RenderRequest(
        config=LLMConfig(**carried["config"]),
        system=carried["system"],
        messages=tuple(history),
        tools=tuple(codec.from_json(List[ToolSpec], carried["tools"])),
        settings=codec.from_json(Settings, carried["settings"]),
    )
    return history, request


def sse_bytes(events: Sequence[Tuple[str, Mapping[str, Any]]]) -> bytes:
    return "".join(
        f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n" for name, data in events
    ).encode("utf-8")

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.features.llm.harness import codec
from src.features.llm.harness.adapter import Adapter, RenderRequest, Settings, ToolSpec, WireRequest
from src.features.llm.harness.assembler import StepAssembler
from src.features.llm.harness.errors import HarnessError
from src.features.llm.harness.events import Finish, StreamEvent, Usage
from src.features.llm.harness.messages import Message, Origin, credential_fingerprint
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.repository import LLMConfig
from tests.fixtures.llm_harness import decode_chunks

CASSETTE_ROOT = Path(__file__).resolve().parents[3] / "fixtures" / "llm_cassettes"

SCRUBBED = "[scrubbed]"
CASSETTE_API_KEY = "cassette-key"
SCRUBBED_HEADERS = frozenset({"authorization", "x-api-key", "x-goog-api-key", "openai-organization"})
SCRUBBED_QUERY = frozenset({"key", "api_key", "access_token"})

STEP_FILES = ("meta.json", "input.json", "request.json", "chunks.jsonl", "events.json", "step.json")
HAND_BUILT = "hand-built"
RECORDED = "recorded"

Events = Union[List[StreamEvent], HarnessError]


@dataclass
class CassetteStep:
    directory: Path
    meta: Dict[str, Any]
    input: Dict[str, Any]
    request: Dict[str, Any]
    chunks: List[bytes]
    events: Events
    step: Optional[Dict[str, Any]]


def scrub_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    return {name: SCRUBBED if name.lower() in SCRUBBED_HEADERS else value for name, value in headers.items()}


def scrub_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.query:
        return url
    query = [(key, SCRUBBED if key.lower() in SCRUBBED_QUERY else value) for key, value in parse_qsl(parts.query, keep_blank_values=True)]
    return urlunsplit(parts._replace(query=urlencode(query, safe="[]")))


def scrubbed_request(wire: WireRequest) -> Dict[str, Any]:
    return {
        "method": wire.method,
        "url": scrub_url(wire.url),
        "headers": scrub_headers(wire.headers),
        "body": wire.body,
    }


def request_bytes(body: Any) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def scenario_dirs(adapter: str, root: Path = CASSETTE_ROOT) -> List[Path]:
    base = root / adapter
    if not base.is_dir():
        return []
    return sorted(path for path in base.iterdir() if path.is_dir())


def step_dirs(scenario: Path) -> List[Path]:
    return sorted(path for path in scenario.iterdir() if path.is_dir())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_step(directory: Path) -> CassetteStep:
    missing = [name for name in STEP_FILES if not (directory / name).exists() and name != "step.json"]
    if missing:
        raise AssertionError(f"{directory}: missing {missing}")
    chunks = [
        base64.b64decode(json.loads(line)["b64"])
        for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_events = _read_json(directory / "events.json")
    if isinstance(raw_events, dict) and "error" in raw_events:
        events: Events = HarnessError.from_json(raw_events["error"])
    else:
        events = codec.from_json(List[StreamEvent], raw_events)
    step_path = directory / "step.json"
    return CassetteStep(
        directory=directory,
        meta=_read_json(directory / "meta.json"),
        input=_read_json(directory / "input.json"),
        request=_read_json(directory / "request.json"),
        chunks=chunks,
        events=events,
        step=_read_json(step_path) if step_path.exists() else None,
    )


def load_scenario(scenario: Path) -> List[CassetteStep]:
    steps = [load_step(directory) for directory in step_dirs(scenario)]
    if not steps:
        raise AssertionError(f"{scenario}: no step directories")
    return steps


def profile_for(meta: Mapping[str, Any]) -> CapabilityProfile:
    profile = CapabilityProfile(adapter=meta["adapter"], model=meta["model"])
    overrides = meta.get("profile_overrides") or {}
    return profile.layered([ProfileLayer(ProfileSource.OVERRIDE, overrides)]) if overrides else profile


def origin_for(meta: Mapping[str, Any], config: LLMConfig) -> Origin:
    return Origin(
        adapter=meta["adapter"],
        provider=config.type,
        model=config.model,
        credential=credential_fingerprint(config.base_url, config.api_key),
    )


def config_from(meta: Mapping[str, Any], data: Mapping[str, Any]) -> LLMConfig:
    defaults = {
        "id": "cassette",
        "name": "cassette",
        "type": meta["adapter"],
        "enabled": True,
        "model": meta["model"],
        "system_message": "",
        "api_key": CASSETTE_API_KEY,
    }
    return LLMConfig(**{**defaults, **data})


def config_to_dict(config: LLMConfig) -> Dict[str, Any]:
    data = config.model_dump()
    if data.get("api_key"):
        data["api_key"] = CASSETTE_API_KEY
    return data


def replayed_config(config: LLMConfig) -> LLMConfig:
    return LLMConfig(**config_to_dict(config))


def request_from(
    meta: Mapping[str, Any], carried: Mapping[str, Any], history: Sequence[Message],
) -> RenderRequest:
    return RenderRequest(
        config=config_from(meta, carried["config"]),
        system=carried.get("system", ""),
        messages=tuple(history),
        tools=tuple(codec.from_json(List[ToolSpec], carried.get("tools", []))),
        settings=codec.from_json(Settings, carried.get("settings", {})),
        stream=True,
    )


def input_from(request: RenderRequest, new_messages: Sequence[Message], *, full: bool) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if full:
        data["config"] = config_to_dict(request.config)
        data["system"] = request.system
        data["tools"] = codec.to_json(list(request.tools))
        data["settings"] = codec.to_json(request.settings)
    data["messages"] = codec.to_json(list(new_messages))
    return data


def step_payload(assembler: StepAssembler) -> Dict[str, Any]:
    return {
        "message": codec.to_json(assembler.message()),
        "usage": codec.to_json(assembler.usage),
        "finish": codec.to_json(assembler.finish),
    }


def write_step(
    directory: Path,
    *,
    meta: Mapping[str, Any],
    input: Mapping[str, Any],
    wire: WireRequest,
    chunks: Sequence[bytes],
    events: Events,
    step: Optional[Mapping[str, Any]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "meta.json", meta)
    _write_json(directory / "input.json", input)
    _write_json(directory / "request.json", scrubbed_request(wire))
    (directory / "chunks.jsonl").write_text(
        "".join(json.dumps({"b64": base64.b64encode(chunk).decode("ascii")}) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    if isinstance(events, HarnessError):
        _write_json(directory / "events.json", {"error": events.to_json()})
    else:
        _write_json(directory / "events.json", codec.to_json(events))
    step_path = directory / "step.json"
    if step is not None:
        _write_json(step_path, step)
    elif step_path.exists():
        step_path.unlink()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _dump(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return text if len(text) <= 6000 else text[:6000] + "\n..."


def _same(step: CassetteStep, label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(
            f"{step.directory}: {label} differs\n--- expected\n{_dump(expected)}\n--- actual\n{_dump(actual)}"
        )


def run_scenario(adapter: Adapter, scenario: Path) -> None:
    history: List[Message] = []
    carried: Dict[str, Any] = {}
    for step in load_scenario(scenario):
        carried = {**carried, **{key: value for key, value in step.input.items() if key != "messages"}}
        history = history + codec.from_json(List[Message], step.input.get("messages", []))
        profile = profile_for(step.meta)
        request = request_from(step.meta, carried, history)

        wire = adapter.render(request, profile)
        rendered = scrubbed_request(wire)
        _same(step, "request method", rendered["method"], step.request["method"])
        _same(step, "request url", rendered["url"], step.request["url"])
        _same(step, "request headers", rendered["headers"], step.request["headers"])
        _same(step, "request body", rendered["body"], step.request["body"])
        _same(step, "request bytes", wire.content(), request_bytes(step.request["body"]))

        decoder = adapter.decoder(request, profile)
        if isinstance(step.events, HarnessError):
            history = _expect_error(step, decoder, history)
            continue
        decoded = decode_chunks(decoder, step.chunks)
        _same(step, "decoded events", codec.to_json(decoded), codec.to_json(step.events))

        if step.step is None:
            raise AssertionError(f"{step.directory}: step.json is required when the stream decodes")
        stored = codec.from_json(Message, step.step["message"])
        assembler = StepAssembler(stored.id, origin_for(step.meta, request.config))
        assembler.feed_all(decoded)
        _same(step, "assembled step", step_payload(assembler), step.step)
        history = history + [stored]


def _expect_error(step: CassetteStep, decoder: Any, history: List[Message]) -> List[Message]:
    try:
        decode_chunks(decoder, step.chunks)
    except HarnessError as error:
        _same(step, "decode error", error.to_json(), step.events.to_json())
        return history
    raise AssertionError(f"{step.directory}: expected {step.events.kind} while decoding, nothing was raised")


def hand_built_scenarios(root: Path = CASSETTE_ROOT) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    for adapter_dir in sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []:
        for scenario in scenario_dirs(adapter_dir.name, root):
            sources = {_read_json(step / "meta.json").get("source") for step in step_dirs(scenario)}
            if HAND_BUILT in sources:
                found.append((adapter_dir.name, scenario.name))
    return found

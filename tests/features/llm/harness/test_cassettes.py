import asyncio
import base64
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from src.features.llm.harness.adapter import RenderRequest, WireRequest
from src.features.llm.harness.errors import ServerFailed
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.ndjson import NDJSONFramer
from src.features.llm.harness.profile import ProfileLayer, ProfileSource
from src.features.llm.harness.transport import HttpAdapter
from tests.features.llm.harness import cassettes
from tests.features.llm.harness.wire_peer import chunked, install_peer

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "llm_cassette_record.py"
_spec = importlib.util.spec_from_file_location("llm_cassette_record", SCRIPT)
recorder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recorder)

SECRET = "sekret-key-123"


class _LineDecoder:
    def __init__(self):
        self._framer = NDJSONFramer()
        self._calls = 0

    def feed(self, chunk):
        return self._events(self._framer.feed(chunk))

    def close(self):
        return self._events(self._framer.close())

    def _events(self, values):
        events = []
        for value in values:
            if "t" in value:
                events.append(TextDelta(block=0, text=value["t"]))
            elif "call" in value:
                call = value["call"]
                block = 1 + self._calls
                self._calls += 1
                events.append(ToolCallStart(block=block, call_id=call["id"], name=call["name"]))
                events.append(ToolArgsDelta(block=block, text=json.dumps(call["args"])))
                events.append(ToolCallEnd(block=block))
            elif "error" in value:
                raise ServerFailed(value["error"], code="in_band")
            elif value.get("done"):
                reason = FinishReason.TOOL_CALLS if self._calls else FinishReason.STOP
                events.append(UsageReported(Usage(input_tokens=3, output_tokens=2, total_tokens=5)))
                events.append(Finish(reason=reason, raw_reason="done"))
        return events


class LineAdapter(HttpAdapter):
    name = "line-test"

    def render(self, request: RenderRequest, profile) -> WireRequest:
        messages = []
        for message in request.messages:
            entry = {"role": message.role.value, "text": message.text()}
            if message.tool_calls():
                entry["calls"] = [{"id": call.call_id, "name": call.name, "args": call.args} for call in message.tool_calls()]
            if message.tool_results():
                entry["results"] = [{"id": item.call_id, "text": item.text()} for item in message.tool_results()]
            messages.append(entry)
        headers = {"Content-Type": "application/json"}
        if request.config.api_key:
            headers["Authorization"] = f"Bearer {request.config.api_key}"
        return WireRequest(
            method="POST",
            url=f"{request.config.base_url}/line?key={request.config.api_key}",
            headers=headers,
            body={"model": request.config.model, "system": request.system, "messages": messages},
        )

    def decoder(self, request, profile):
        return _LineDecoder()

    async def discover(self, config):
        return ProfileLayer(ProfileSource.DISCOVERED, {})

    async def list_models(self, base_url, api_key=None):
        return []


def _ndjson(*objects) -> bytes:
    return b"".join(json.dumps(item).encode() + b"\n" for item in objects)


TOOL_REPLY = _ndjson({"t": "Looking. "}, {"call": {"id": "c1", "name": "lookup", "args": {"q": "x"}}}, {"done": True})
FINAL_REPLY = _ndjson({"t": "Found "}, {"t": "it."}, {"done": True})

SPEC = {
    "adapter": "line-test",
    "scenario": "tool_then_answer",
    "model": "line-1",
    "doc_url": "https://example.invalid/docs",
    "config": {"base_url": "http://peer.invalid:9", "api_key_env": "LINE_TEST_KEY", "max_tokens": 50},
    "system": "SYS",
    "tools": [{"name": "lookup", "description": "look", "parameters": {"type": "object"}}],
    "turns": ["find x"],
    "tool_results": {"lookup": "result text"},
}


def _record(monkeypatch, out: Path, replies, spec=None):
    monkeypatch.setenv("LINE_TEST_KEY", SECRET)
    install_peer(monkeypatch, *[chunked([reply[:11], reply[11:]]) for reply in replies])
    return asyncio.run(recorder.record(spec or SPEC, out, LineAdapter, force=True))


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    _record(monkeypatch, tmp_path, [TOOL_REPLY, FINAL_REPLY])
    return tmp_path / "line-test" / "tool_then_answer"


def _edit(path: Path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_a_recorded_scenario_replays_cleanly(recorded):
    assert [p.name for p in cassettes.step_dirs(recorded)] == ["01", "02"]

    cassettes.run_scenario(LineAdapter(), recorded)


def test_the_recorder_keeps_chunk_boundaries_and_the_tool_round_trip(recorded):
    first, second = cassettes.load_scenario(recorded)

    assert first.chunks == [TOOL_REPLY[:11], TOOL_REPLY[11:]]
    assert first.meta["source"] == "recorded"
    assert [m["role"] for m in second.request["body"]["messages"]] == ["user", "assistant", "tool"]
    assert second.request["body"]["messages"][2]["results"] == [{"id": "c1", "text": "result text"}]


def test_no_secret_reaches_any_cassette_file(recorded):
    for path in recorded.rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text(encoding="utf-8")
    request = json.loads((recorded / "01" / "request.json").read_text(encoding="utf-8"))
    assert request["headers"]["Authorization"] == cassettes.SCRUBBED
    assert request["url"].endswith(f"key={cassettes.SCRUBBED}")
    stored = json.loads((recorded / "01" / "input.json").read_text(encoding="utf-8"))
    assert stored["config"]["api_key"] == cassettes.CASSETTE_API_KEY


def test_a_changed_request_body_fails_the_replay(recorded):
    _edit(recorded / "01" / "request.json", lambda data: data["body"].update(system="OTHER"))

    with pytest.raises(AssertionError, match="request body differs"):
        cassettes.run_scenario(LineAdapter(), recorded)


def test_a_changed_url_fails_the_replay(recorded):
    _edit(recorded / "01" / "request.json", lambda data: data.update(url="http://elsewhere.invalid/line"))

    with pytest.raises(AssertionError, match="request url differs"):
        cassettes.run_scenario(LineAdapter(), recorded)


def test_a_changed_header_fails_the_replay(recorded):
    _edit(recorded / "01" / "request.json", lambda data: data["headers"].update({"X-Extra": "1"}))

    with pytest.raises(AssertionError, match="request headers differ"):
        cassettes.run_scenario(LineAdapter(), recorded)


def test_a_reordered_request_body_fails_the_byte_comparison(recorded):
    path = recorded / "01" / "request.json"
    _edit(path, lambda data: data.update(body=dict(reversed(list(data["body"].items())))))

    with pytest.raises(AssertionError, match="request bytes differ"):
        cassettes.run_scenario(LineAdapter(), recorded)


def test_a_changed_chunk_fails_the_decode_comparison(recorded):
    path = recorded / "01" / "chunks.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps({"b64": base64.b64encode(_ndjson({"t": "Different. "})[:11]).decode()})
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="decoded events differ"):
        cassettes.run_scenario(LineAdapter(), recorded)


def test_a_changed_assembled_step_fails_the_assembly_comparison(recorded):
    def change(data):
        data["message"]["parts"][0]["text"] = "Edited. "

    _edit(recorded / "01" / "step.json", change)

    with pytest.raises(AssertionError, match="assembled step differs"):
        cassettes.run_scenario(LineAdapter(), recorded)


def test_a_later_request_that_no_longer_matches_the_stored_history_fails(monkeypatch, tmp_path, recorded):
    other = tmp_path / "other"
    _record(
        monkeypatch,
        other,
        [_ndjson({"t": "Elsewhere. "}, {"call": {"id": "c9", "name": "lookup", "args": {"q": "y"}}}, {"done": True}), FINAL_REPLY],
    )
    spliced = tmp_path / "spliced"
    shutil.copytree(other / "line-test" / "tool_then_answer", spliced)
    shutil.copy(recorded / "02" / "request.json", spliced / "02" / "request.json")
    shutil.copy(recorded / "02" / "input.json", spliced / "02" / "input.json")

    with pytest.raises(AssertionError, match="02: request body differs"):
        cassettes.run_scenario(LineAdapter(), spliced)


def test_an_in_band_error_is_recorded_and_replays_as_the_same_error(monkeypatch, tmp_path):
    _record(monkeypatch, tmp_path, [_ndjson({"t": "Hi"}, {"error": "overloaded"})])
    scenario = tmp_path / "line-test" / "tool_then_answer"

    assert not (scenario / "01" / "step.json").exists()
    assert json.loads((scenario / "01" / "events.json").read_text(encoding="utf-8"))["error"]["code"] == "in_band"
    cassettes.run_scenario(LineAdapter(), scenario)

    _edit(scenario / "01" / "events.json", lambda data: data["error"].update(code="other"))
    with pytest.raises(AssertionError, match="decode error differs"):
        cassettes.run_scenario(LineAdapter(), scenario)


def test_a_stream_that_should_have_failed_but_decodes_is_reported(recorded):
    (recorded / "02" / "events.json").write_text(
        json.dumps({"error": {"kind": "server_error", "message": "x", "status": None, "code": None, "retry_after": None, "detail": {}}}),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="expected server_error while decoding"):
        cassettes.run_scenario(LineAdapter(), recorded)


class LeakyAdapter(LineAdapter):
    def render(self, request, profile):
        wire = super().render(request, profile)
        return WireRequest(wire.method, wire.url, wire.headers, {**wire.body, "note": request.config.api_key})


def test_the_recorder_refuses_to_keep_a_recording_that_contains_the_key(monkeypatch, tmp_path):
    monkeypatch.setenv("LINE_TEST_KEY", SECRET)
    install_peer(monkeypatch, chunked([FINAL_REPLY]))

    with pytest.raises(SystemExit, match="contains the API key"):
        asyncio.run(recorder.record(SPEC, tmp_path, LeakyAdapter, force=True))


def test_the_recorder_needs_the_key_in_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("LINE_TEST_KEY", raising=False)

    with pytest.raises(SystemExit, match="LINE_TEST_KEY is not set"):
        asyncio.run(recorder.record(SPEC, tmp_path, LineAdapter, force=True))


def test_the_recorder_will_not_replace_a_scenario_without_force(monkeypatch, recorded, tmp_path):
    monkeypatch.setenv("LINE_TEST_KEY", SECRET)

    with pytest.raises(SystemExit, match="exists"):
        asyncio.run(recorder.record(SPEC, tmp_path, LineAdapter, force=False))


def test_hand_built_scenarios_are_listed(recorded, tmp_path):
    root = tmp_path
    assert cassettes.hand_built_scenarios(root) == []

    _edit(recorded / "01" / "meta.json", lambda data: data.update(source=cassettes.HAND_BUILT))

    assert cassettes.hand_built_scenarios(root) == [("line-test", "tool_then_answer")]


def test_scrub_helpers_hide_credentials_and_nothing_else():
    headers = {"Authorization": "Bearer k", "x-api-key": "k", "X-Goog-Api-Key": "k", "OpenAI-Organization": "org", "Accept": "text/event-stream"}

    assert cassettes.scrub_headers(headers) == {
        "Authorization": cassettes.SCRUBBED,
        "x-api-key": cassettes.SCRUBBED,
        "X-Goog-Api-Key": cassettes.SCRUBBED,
        "OpenAI-Organization": cassettes.SCRUBBED,
        "Accept": "text/event-stream",
    }
    assert cassettes.scrub_url("https://h/x?alt=sse&key=abc") == f"https://h/x?alt=sse&key={cassettes.SCRUBBED}"
    assert cassettes.scrub_url("https://h/x") == "https://h/x"


def _all_steps():
    root = cassettes.CASSETTE_ROOT
    return [
        step
        for adapter in (sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else [])
        for scenario in cassettes.scenario_dirs(adapter.name)
        for step in cassettes.step_dirs(scenario)
    ]


@pytest.mark.parametrize("step", _all_steps(), ids=lambda path: f"{path.parent.parent.name}/{path.parent.name}/{path.name}")
def test_every_stored_cassette_says_where_it_came_from(step):
    meta = json.loads((step / "meta.json").read_text(encoding="utf-8"))

    assert meta["source"] in (cassettes.HAND_BUILT, cassettes.RECORDED)
    assert meta["adapter"] == step.parent.parent.name
    if meta["source"] == cassettes.HAND_BUILT:
        assert meta["doc_url"].startswith("https://")
    else:
        assert meta["recorded_at"]

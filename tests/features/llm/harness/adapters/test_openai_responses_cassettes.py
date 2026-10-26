import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.features.llm.harness import codec
from src.features.llm.harness.adapter import RenderRequest, Settings, ToolSpec
from src.features.llm.harness.assembler import StepAssembler
from src.features.llm.harness.errors import HarnessError, ServerFailed
from src.features.llm.harness.messages import Message, ReasoningPart, TextPart, ToolCallPart
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.harness.turn_state import NoticeKind
from src.features.llm.repository import LLMConfig
from tests.features.llm.harness.adapters.openai_responses_support import ADAPTER

ROOT = Path(__file__).resolve().parents[4] / "fixtures" / "llm_cassettes" / "openai-responses"
SCENARIOS = ["thinking_parallel_tools", "approval_pause", "credential_switch", "in_band_error"]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Played:
    folder: Path
    meta: Dict[str, Any]
    config: LLMConfig
    request: RenderRequest
    profile: CapabilityProfile
    chunks: List[bytes]
    wire: Any
    events: List[Any]
    error: Optional[HarnessError]
    message: Optional[Message]
    step: Optional[Dict[str, Any]]


def golden(wire) -> Dict[str, Any]:
    headers = {key: ("[scrubbed]" if key.lower() == "authorization" else value) for key, value in wire.headers.items()}
    return {"method": wire.method, "url": wire.url, "headers": headers, "body": wire.body}


def chunks_of(folder: Path) -> List[bytes]:
    lines = (folder / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return [base64.b64decode(json.loads(line)["b64"]) for line in lines]


def play(name: str) -> List[Played]:
    history: List[Message] = []
    played: List[Played] = []
    for number, folder in enumerate(sorted(path for path in (ROOT / name).iterdir() if path.is_dir()), start=1):
        meta = read(folder / "meta.json")
        inputs = read(folder / "input.json")
        config = LLMConfig(**inputs["config"])
        history.extend(codec.from_json(Message, item) for item in inputs["append"])
        request = RenderRequest(
            config=config,
            system=inputs["system"],
            messages=tuple(history),
            tools=tuple(codec.from_json(ToolSpec, item) for item in inputs["tools"]),
            settings=codec.from_json(Settings, inputs["settings"]),
            stream=True,
        )
        profile = CapabilityProfile(adapter=meta["adapter"], model=meta["model"]).layered(
            [ProfileLayer(ProfileSource.OVERRIDE, meta["profile_overrides"])]
        )
        wire = ADAPTER.render(request, profile)
        chunks = chunks_of(folder)
        decoder = ADAPTER.decoder(request, profile)
        events: List[Any] = []
        error = None
        try:
            for chunk in chunks:
                events.extend(decoder.feed(chunk))
            events.extend(decoder.close())
        except HarnessError as raised:
            error = raised
        message = step = None
        if error is None:
            assembler = StepAssembler(f"msg_{number:02d}", ADAPTER.origin(config))
            assembler.feed_all(events)
            message = assembler.message()
            step = {
                "message": codec.to_json(message),
                "usage": codec.to_json(assembler.usage),
                "finish": codec.to_json(assembler.finish),
            }
            history.append(codec.from_json(Message, step["message"]))
        played.append(Played(folder, meta, config, request, profile, chunks, wire, events, error, message, step))
    return played


@pytest.fixture(scope="module")
def plays() -> Dict[str, List[Played]]:
    return {name: play(name) for name in SCENARIOS}


class TestRecordedShape:
    @pytest.mark.parametrize("name", SCENARIOS)
    def test_every_step_is_labelled_and_cites_the_docs(self, name, plays):
        for step in plays[name]:
            assert step.meta["adapter"] == "openai-responses"
            assert step.meta["source"] == "hand-built"
            assert step.meta["doc_url"].startswith("https://developers.openai.com/api/docs/")

    @pytest.mark.parametrize("name", SCENARIOS)
    def test_the_chunks_cut_through_events(self, name, plays):
        for step in plays[name]:
            assert len(step.chunks) > 3
            assert any(not chunk.endswith(b"\n\n") for chunk in step.chunks)

    def test_a_chunk_boundary_falls_inside_a_multibyte_character(self, plays):
        broken = []
        for chunk in plays["thinking_parallel_tools"][0].chunks:
            try:
                chunk.decode("utf-8")
            except UnicodeDecodeError:
                broken.append(chunk)

        assert broken


class TestGoldens:
    @pytest.mark.parametrize("name", SCENARIOS)
    def test_each_step_renders_to_its_recorded_request(self, name, plays):
        for step in plays[name]:
            assert golden(step.wire) == read(step.folder / "request.json")

    @pytest.mark.parametrize("name", SCENARIOS)
    def test_each_step_decodes_to_its_recorded_events(self, name, plays):
        for step in plays[name]:
            assert codec.to_json(step.events) == read(step.folder / "events.json")

    @pytest.mark.parametrize("name", [name for name in SCENARIOS if name != "in_band_error"])
    def test_each_step_assembles_to_its_recorded_message(self, name, plays):
        for step in plays[name]:
            assert step.step == read(step.folder / "step.json")

    def test_the_request_bytes_are_the_stored_json(self, plays):
        for step in plays["thinking_parallel_tools"]:
            stored = json.dumps(read(step.folder / "request.json")["body"], ensure_ascii=False, separators=(",", ":"))

            assert step.wire.content() == stored.encode("utf-8")


class TestThinkingWithParallelTools:
    def test_the_first_step_keeps_reasoning_commentary_and_both_calls_in_order(self, plays):
        message = plays["thinking_parallel_tools"][0].message

        kinds = [type(part) for part in message.parts]

        assert kinds == [ReasoningPart, TextPart, ToolCallPart, ToolCallPart]
        assert message.parts[0].summary == "Checking both cities.\n\nCalling the tool twice."
        assert message.parts[1].phase == "commentary"
        assert [part.args for part in message.parts[2:]] == [{"city": "Paris"}, {"city": "Zürich"}]

    def test_the_second_request_replays_the_first_output_untouched(self, plays):
        first, second = plays["thinking_parallel_tools"]

        replayed = second.wire.body["input"]

        assert replayed[1:5] == first.message.raw
        assert replayed[1]["encrypted_content"].startswith("gAAAA")
        assert replayed[2]["phase"] == "commentary"
        assert [item["type"] for item in replayed[5:]] == ["function_call_output", "function_call_output"]
        assert [item["call_id"] for item in replayed[5:]] == ["call_paris_01", "call_zurich_02"]

    def test_usage_reports_cached_and_reasoning_tokens(self, plays):
        first, second = plays["thinking_parallel_tools"]

        assert (first.step["usage"]["reasoning_tokens"], first.step["usage"]["cached_input_tokens"]) == (128, 0)
        assert second.step["usage"]["cached_input_tokens"] == 384
        assert second.step["finish"]["reason"] == "stop"
        assert first.step["finish"]["reason"] == "tool_calls"

    def test_an_empty_summary_still_leaves_a_reasoning_part(self, plays):
        part = plays["thinking_parallel_tools"][1].message.parts[0]

        assert isinstance(part, ReasoningPart)
        assert part.summary == ""
        assert part.opaque["encrypted_content"]


class TestApprovalPause:
    def test_the_paused_call_is_replayed_with_its_reasoning_after_the_decision(self, plays):
        first, second = plays["approval_pause"]

        replayed = second.wire.body["input"]

        assert replayed[1:3] == first.message.raw
        assert replayed[1]["type"] == "reasoning"
        assert replayed[2]["type"] == "function_call"

    def test_the_denial_is_sent_as_the_tool_output(self, plays):
        output = plays["approval_pause"][1].wire.body["input"][-1]

        assert output == {
            "type": "function_call_output",
            "call_id": "call_delete_01",
            "output": "The user declined this call: it would remove 214 files.",
        }

    def test_arguments_streamed_in_pieces_are_reassembled(self, plays):
        call = plays["approval_pause"][0].message.parts[-1]

        assert call.args == {"path": "build/"}
        assert call.args_raw == '{"path":"build/"}'


class TestCredentialSwitch:
    def test_the_second_request_carries_no_opaque_data_from_the_first_key(self, plays):
        first, second = plays["credential_switch"]

        assert first.message.origin.credential != ADAPTER.origin(second.config).credential
        assert "encrypted_content" not in json.dumps(second.wire.body["input"])
        assert not any("id" in item for item in second.wire.body["input"])

    def test_the_visible_answer_is_still_replayed(self, plays):
        replayed = plays["credential_switch"][1].wire.body["input"]

        assert replayed[1] == {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "101 is prime."}],
            "phase": "final_answer",
        }

    def test_the_user_is_told_which_reasoning_was_left_out(self, plays):
        second = plays["credential_switch"][1]

        notices = ADAPTER.reasoning_notices(second.request, second.profile)

        assert [(notice.kind, notice.detail["reason"]) for notice in notices] == [(NoticeKind.REASONING_DROPPED, "credential")]
        assert notices[0].detail["message_id"] == "msg_01"

    def test_the_first_step_replays_nothing_to_report(self, plays):
        first = plays["credential_switch"][0]

        assert ADAPTER.reasoning_notices(first.request, first.profile) == ()


class TestInBandError:
    def test_the_failure_is_raised_while_decoding_with_the_earlier_events_delivered(self, plays):
        step = plays["in_band_error"][0]

        assert isinstance(step.error, ServerFailed)
        assert step.error.retryable is True
        assert step.error.detail["in_stream"] is True
        assert step.error.to_json() == read(step.folder / "error.json")
        assert len(step.events) == 2

    def test_no_message_is_made_from_a_failed_stream(self, plays):
        assert plays["in_band_error"][0].message is None

from typing import Any, Dict, List

import pytest

from src.features.llm.harness import codec
from src.features.llm.harness.errors import ServerFailed
from src.features.llm.harness.events import StreamEvent
from src.features.llm.harness.messages import Message, ReasoningPart, TextPart, ToolCallPart
from src.features.llm.harness.turn_state import NoticeKind
from tests.features.llm.harness import cassettes
from tests.features.llm.harness.adapters.openai_responses_support import ADAPTER

ROOT = cassettes.CASSETTE_ROOT / "openai-responses"
SCENARIOS = ["thinking_parallel_tools", "approval_pause", "credential_switch", "in_band_error"]


def steps(name: str) -> List[cassettes.CassetteStep]:
    return cassettes.load_scenario(ROOT / name)


def message_of(step: cassettes.CassetteStep) -> Message:
    return codec.from_json(Message, step.step["message"])


def body_of(step: cassettes.CassetteStep) -> Dict[str, Any]:
    return step.request["body"]


def request_and_profile(name: str, index: int):
    carried: Dict[str, Any] = {}
    history: List[Message] = []
    for number, step in enumerate(steps(name)):
        carried = {**carried, **{key: value for key, value in step.input.items() if key != "messages"}}
        history = history + codec.from_json(List[Message], step.input.get("messages", []))
        if number == index:
            return cassettes.request_from(step.meta, carried, history), cassettes.profile_for(step.meta)
        if step.step is not None:
            history.append(message_of(step))
    raise IndexError(index)


def test_the_scenarios_on_disk_are_the_expected_ones():
    assert [path.name for path in cassettes.scenario_dirs("openai-responses")] == sorted(SCENARIOS)


@pytest.mark.parametrize("name", SCENARIOS)
def test_a_scenario_replays_cleanly_through_the_shared_runner(name):
    cassettes.run_scenario(ADAPTER, ROOT / name)


class TestRecordedShape:
    @pytest.mark.parametrize("name", SCENARIOS)
    def test_every_step_is_labelled_and_cites_the_docs(self, name):
        for step in steps(name):
            assert step.meta["adapter"] == "openai-responses"
            assert step.meta["source"] == cassettes.HAND_BUILT
            assert step.meta["doc_url"].startswith("https://developers.openai.com/api/docs/")

    @pytest.mark.parametrize("name", SCENARIOS)
    def test_the_chunks_cut_through_events(self, name):
        for step in steps(name):
            assert len(step.chunks) > 3
            assert any(not chunk.endswith(b"\n\n") for chunk in step.chunks)

    def test_a_chunk_boundary_falls_inside_a_multibyte_character(self):
        broken = []
        for chunk in steps("thinking_parallel_tools")[0].chunks:
            try:
                chunk.decode("utf-8")
            except UnicodeDecodeError:
                broken.append(chunk)

        assert broken


class TestThinkingWithParallelTools:
    def test_the_first_step_keeps_reasoning_commentary_and_both_calls_in_order(self):
        message = message_of(steps("thinking_parallel_tools")[0])

        assert [type(part) for part in message.parts] == [ReasoningPart, TextPart, ToolCallPart, ToolCallPart]
        assert message.parts[0].summary == "Checking both cities.\n\nCalling the tool twice."
        assert message.parts[1].phase == "commentary"
        assert [part.args for part in message.parts[2:]] == [{"city": "Paris"}, {"city": "Zürich"}]

    def test_the_second_request_replays_the_first_output_untouched(self):
        first, second = steps("thinking_parallel_tools")

        replayed = body_of(second)["input"]

        assert replayed[1:5] == message_of(first).raw
        assert replayed[1]["encrypted_content"].startswith("gAAAA")
        assert replayed[2]["phase"] == "commentary"
        assert [item["type"] for item in replayed[5:]] == ["function_call_output", "function_call_output"]
        assert [item["call_id"] for item in replayed[5:]] == ["call_paris_01", "call_zurich_02"]

    def test_usage_reports_cached_and_reasoning_tokens(self):
        first, second = steps("thinking_parallel_tools")

        assert (first.step["usage"]["reasoning_tokens"], first.step["usage"]["cached_input_tokens"]) == (128, 0)
        assert second.step["usage"]["cached_input_tokens"] == 384
        assert first.step["finish"]["reason"] == "tool_calls"
        assert second.step["finish"]["reason"] == "stop"

    def test_an_empty_summary_still_leaves_a_reasoning_part(self):
        part = message_of(steps("thinking_parallel_tools")[1]).parts[0]

        assert isinstance(part, ReasoningPart)
        assert part.summary == ""
        assert part.opaque["encrypted_content"]


class TestApprovalPause:
    def test_the_paused_call_is_replayed_with_its_reasoning_after_the_decision(self):
        first, second = steps("approval_pause")

        replayed = body_of(second)["input"]

        assert replayed[1:3] == message_of(first).raw
        assert [item["type"] for item in replayed[1:3]] == ["reasoning", "function_call"]

    def test_the_denial_is_sent_as_the_tool_output(self):
        assert body_of(steps("approval_pause")[1])["input"][-1] == {
            "type": "function_call_output",
            "call_id": "call_delete_01",
            "output": "The user declined this call: it would remove 214 files.",
        }

    def test_arguments_streamed_in_pieces_are_reassembled(self):
        call = message_of(steps("approval_pause")[0]).parts[-1]

        assert call.args == {"path": "build/"}
        assert call.args_raw == '{"path":"build/"}'


class TestCredentialSwitch:
    def test_the_second_request_carries_no_opaque_data_from_the_first_key(self):
        first, second = steps("credential_switch")
        request, _ = request_and_profile("credential_switch", 1)

        assert message_of(first).origin.credential != ADAPTER.origin(request.config).credential
        assert "encrypted_content" not in str(body_of(second)["input"])
        assert not any("id" in item for item in body_of(second)["input"])

    def test_the_visible_answer_is_still_replayed(self):
        assert body_of(steps("credential_switch")[1])["input"][1] == {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "101 is prime."}],
            "phase": "final_answer",
        }

    def test_the_user_is_told_which_reasoning_was_left_out(self):
        request, profile = request_and_profile("credential_switch", 1)

        notices = ADAPTER.reasoning_notices(request, profile)

        assert [(notice.kind, notice.detail["reason"]) for notice in notices] == [(NoticeKind.REASONING_DROPPED, "credential")]
        assert notices[0].detail["message_id"] == "msg_01"

    def test_the_first_step_replays_nothing_to_report(self):
        request, profile = request_and_profile("credential_switch", 0)

        assert ADAPTER.reasoning_notices(request, profile) == ()


class TestInBandError:
    def test_the_stored_failure_is_the_classified_in_stream_error(self):
        step = steps("in_band_error")[0]

        assert isinstance(step.events, ServerFailed)
        assert step.events.retryable is True
        assert step.events.detail["in_stream"] is True
        assert step.step is None

    def test_the_events_before_the_failure_reach_the_caller(self):
        step = steps("in_band_error")[0]
        request, profile = request_and_profile("in_band_error", 0)
        decoder = ADAPTER.decoder(request, profile)
        delivered: List[StreamEvent] = []

        with pytest.raises(ServerFailed):
            for chunk in step.chunks:
                delivered.extend(decoder.feed(chunk))

        assert len(delivered) == 2

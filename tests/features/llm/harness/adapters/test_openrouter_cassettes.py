from typing import Any, Dict, List

import pytest

from src.features.llm.harness import codec
from src.features.llm.harness.adapters.openrouter import DETAILS_KEY
from src.features.llm.harness.errors import ServerFailed
from src.features.llm.harness.events import Finish, FinishReason, TextDelta, UsageReported
from src.features.llm.harness.messages import Message, Origin, ReasoningPart, TextPart, ToolCallPart
from tests.features.llm.harness import cassettes
from tests.features.llm.harness.adapters.openrouter_support import ADAPTER
from tests.fixtures.llm_harness import decode_chunks

ROOT = cassettes.CASSETTE_ROOT / "openrouter"
PLANNED = ["thinking_parallel_tools", "in_band_error", "credential_switch"]
PRESENT = [path.name for path in cassettes.scenario_dirs("openrouter")]
DOC_PREFIX = "https://openrouter.ai/docs/"


def steps(name: str) -> List[cassettes.CassetteStep]:
    return cassettes.load_scenario(ROOT / name)


def hand_built(name: str) -> bool:
    return name in PRESENT and all(step.meta["source"] == cassettes.HAND_BUILT for step in steps(name))


def only_while_hand_built(name: str):
    return pytest.mark.skipif(not hand_built(name), reason=f"{name} was replaced by a recording")


def message_of(step: cassettes.CassetteStep) -> Message:
    return codec.from_json(Message, step.step["message"])


def body_of(step: cassettes.CassetteStep) -> Dict[str, Any]:
    return step.request["body"]


def assistant_of(step: cassettes.CassetteStep) -> Dict[str, Any]:
    return next(entry for entry in body_of(step)["messages"] if entry["role"] == "assistant")


def request_of(name: str, index: int):
    carried: Dict[str, Any] = {}
    history: List[Message] = []
    for number, step in enumerate(steps(name)):
        carried = {**carried, **{key: value for key, value in step.input.items() if key != "messages"}}
        history = history + codec.from_json(List[Message], step.input.get("messages", []))
        if number == index:
            return cassettes.request_from(step.meta, carried, history)
        if step.step is not None:
            history.append(message_of(step))
    raise IndexError(index)


def test_every_planned_scenario_is_present():
    assert set(PLANNED) <= set(PRESENT)


@pytest.mark.parametrize("name", PRESENT)
def test_a_scenario_replays_cleanly_through_the_shared_runner(name):
    cassettes.run_scenario(ADAPTER, ROOT / name)


class TestEveryRecording:
    @pytest.mark.parametrize("name", PRESENT)
    def test_every_step_names_its_adapter_and_source(self, name):
        for step in steps(name):
            assert step.meta["adapter"] == "openrouter"
            assert step.meta["source"] in (cassettes.HAND_BUILT, cassettes.RECORDED)
            if step.meta["source"] == cassettes.HAND_BUILT:
                assert step.meta["doc_url"].startswith(DOC_PREFIX)

    @pytest.mark.parametrize("name", PRESENT)
    def test_streams_are_cut_mid_record_and_carry_comment_lines(self, name):
        for step in steps(name):
            assert len(step.chunks) > 3
            assert any(not chunk.endswith(b"\n\n") for chunk in step.chunks)
            assert b": OPENROUTER PROCESSING" in b"".join(step.chunks)


@only_while_hand_built("thinking_parallel_tools")
class TestThinkingWithParallelTools:
    def test_the_first_reply_is_two_calls_after_signed_reasoning(self):
        message = message_of(steps("thinking_parallel_tools")[0])

        assert [type(part) for part in message.parts] == [ReasoningPart, TextPart, ToolCallPart, ToolCallPart]
        assert message.parts[0].text == "Two cities, so two independent lookups."
        assert [part.args for part in message.parts[2:]] == [{"city": "Paris"}, {"city": "Zurich"}]
        assert message.raw["provider"] == "Anthropic"
        assert message.raw[DETAILS_KEY][0]["signature"].startswith("EqQB")

    def test_the_usage_carries_cost_cache_and_reasoning_tokens(self):
        first = steps("thinking_parallel_tools")[0]
        usage = next(event.usage for event in first.events if isinstance(event, UsageReported))

        assert usage.cost == 0.002676
        assert usage.cache_write_tokens == 384
        assert usage.reasoning_tokens == 41
        assert first.step["finish"]["reason"] == FinishReason.TOOL_CALLS.value

    def test_the_second_request_replays_the_stored_blocks_unmodified(self):
        first, second = steps("thinking_parallel_tools")

        sent = assistant_of(second)

        assert sent["reasoning_details"] == message_of(first).raw[DETAILS_KEY]
        assert "reasoning" not in sent
        assert [call["id"] for call in sent["tool_calls"]] == ["toolu_01A", "toolu_01B"]

    def test_the_second_request_pins_the_provider_that_signed_the_reasoning(self):
        assert body_of(steps("thinking_parallel_tools")[1])["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}
        assert "provider" not in body_of(steps("thinking_parallel_tools")[0])

    def test_the_final_answer_ends_the_turn(self):
        second = steps("thinking_parallel_tools")[1]
        events = second.events

        assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "Paris is sunny at 21C, Zurich is rainy at 14C."
        assert next(event for event in events if isinstance(event, Finish)).reason is FinishReason.STOP


@only_while_hand_built("in_band_error")
class TestInBandError:
    def test_the_stored_failure_is_the_classified_in_stream_error(self):
        step = steps("in_band_error")[0]

        assert isinstance(step.events, ServerFailed)
        assert step.events.retryable is True
        assert step.events.status == 502
        assert step.events.detail["in_stream"] is True
        assert step.events.detail["metadata"]["provider_name"] == "Anthropic"
        assert step.step is None

    def test_the_text_before_the_failure_reached_the_caller(self):
        step = steps("in_band_error")[0]
        decoder = ADAPTER.decoder(request_of("in_band_error", 0), cassettes.profile_for(step.meta))
        delivered = []

        with pytest.raises(ServerFailed):
            for chunk in step.chunks:
                delivered.extend(decoder.feed(chunk))

        assert "".join(event.text for event in delivered if isinstance(event, TextDelta)) == "The plan is to "

    def test_the_failure_arrives_as_a_finish_reason_error_event_inside_the_stream(self):
        assert b"finish_reason\":\"error\"" in b"".join(steps("in_band_error")[0].chunks)


@only_while_hand_built("credential_switch")
class TestCredentialSwitch:
    def test_the_second_request_carries_no_opaque_data_from_the_first_key(self):
        first, second = steps("credential_switch")
        request = request_of("credential_switch", 1)

        assert message_of(first).origin.credential != Origin.for_config(ADAPTER.name, request.config).credential
        sent = assistant_of(second)
        assert "reasoning_details" not in sent
        assert "provider" not in body_of(second)
        assert "R0VOQVJJQ0FMTFkt" not in str(body_of(second))

    def test_the_visible_text_and_tool_calls_are_still_replayed(self):
        sent = assistant_of(steps("credential_switch")[1])

        assert sent["content"] == "Checking."
        assert sent["tool_calls"][0]["id"] == "toolu_01A"

    def test_the_first_reply_stored_only_encrypted_blocks_and_pinned_nothing(self):
        first = steps("credential_switch")[0]

        assert message_of(first).raw[DETAILS_KEY][0]["type"] == "reasoning.encrypted"
        assert "provider" not in body_of(first)

    def test_decoding_still_works_after_the_switch(self):
        second = steps("credential_switch")[1]

        decoded = decode_chunks(ADAPTER.decoder(request_of("credential_switch", 1), cassettes.profile_for(second.meta)), second.chunks)

        assert "".join(event.text for event in decoded if isinstance(event, TextDelta)) == "It is fine."

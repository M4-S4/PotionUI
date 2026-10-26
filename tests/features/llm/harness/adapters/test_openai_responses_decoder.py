import json

import pytest

from src.features.llm.harness.errors import (
    MalformedStream,
    OpaqueViolation,
    RateLimited,
    ServerFailed,
)
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    OpaqueDelta,
    ReasoningDelta,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import ReasoningPart, TextPart, ToolCallPart
from tests.features.llm.harness.adapters.openai_responses_support import (
    ADAPTER,
    assistant_message,
    call_item,
    completed,
    item_events,
    make_profile,
    make_request,
    message_item,
    reasoning_item,
    sse,
    stream_of,
    user_message,
)
from tests.fixtures.llm_harness import assemble, decode_chunks

CREATED = {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}


def request():
    return make_request([user_message()])


def decode(data, chunks=None):
    decoder = ADAPTER.decoder(request(), make_profile())
    return decode_chunks(decoder, chunks if chunks is not None else [data])


def finish_of(events):
    return events[-1]


class TestStreaming:
    def test_a_streamed_answer_becomes_text_deltas_with_the_phase(self):
        item = message_item("Hello there", "final_answer")
        opening = {**item, "status": "in_progress", "content": []}
        data = sse(
            CREATED,
            {"type": "response.output_item.added", "output_index": 0, "item": opening},
            {"type": "response.content_part.added", "output_index": 0, "content_index": 0, "part": {"type": "output_text"}},
            {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hello "},
            {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "there"},
            {"type": "response.output_text.done", "output_index": 0, "content_index": 0, "text": "Hello there"},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            completed([item]),
        )

        events = decode(data)

        assert events[:2] == [
            TextDelta(block=0, text="Hello ", phase="final_answer"),
            TextDelta(block=0, text="there", phase="final_answer"),
        ]
        assert isinstance(finish_of(events), Finish)

    def test_a_phase_that_only_shows_up_on_the_finished_item_still_reaches_the_part(self):
        item = message_item("Hi", "commentary")
        bare = {k: v for k, v in item.items() if k not in ("phase", "content")} | {"content": []}
        data = sse(
            CREATED,
            {"type": "response.output_item.added", "output_index": 0, "item": bare},
            {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hi"},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            completed([item]),
        )

        message = assemble(decode(data))

        assert message.parts == (TextPart("Hi", phase="commentary"),)

    def test_reasoning_summaries_are_joined_and_the_opaque_data_is_attached(self):
        item = reasoning_item("rs_1", "enc-1", ["First.", "Second."])
        data = sse(
            CREATED,
            {"type": "response.output_item.added", "output_index": 0, "item": {"id": "rs_1", "type": "reasoning", "summary": []}},
            {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "First."},
            {"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 1, "delta": "Second."},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            completed([item]),
        )

        events = decode(data)
        message = assemble(events)

        assert events[0] == ReasoningDelta(block=0, text="")
        assert OpaqueDelta(block=0, key="encrypted_content", value="enc-1") in events
        assert message.parts == (
            ReasoningPart(text="", summary="First.\n\nSecond.", opaque={"id": "rs_1", "encrypted_content": "enc-1"}),
        )

    def test_open_weight_reasoning_text_is_kept_apart_from_summaries(self):
        item = {"id": "rs_1", "type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "raw thoughts"}]}
        data = sse(
            CREATED,
            {"type": "response.output_item.added", "output_index": 0, "item": {"id": "rs_1", "type": "reasoning", "summary": []}},
            {"type": "response.reasoning_text.delta", "output_index": 0, "content_index": 0, "delta": "raw "},
            {"type": "response.reasoning_text.delta", "output_index": 0, "content_index": 0, "delta": "thoughts"},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            completed([item]),
        )

        part = assemble(decode(data)).parts[0]

        assert part.text == "raw thoughts"
        assert part.summary == ""

    def test_a_reasoning_item_with_nothing_to_show_still_makes_a_part(self):
        message = assemble(decode(stream_of([reasoning_item("rs_1", "enc-1")])))

        assert message.parts == (ReasoningPart(text="", summary="", opaque={"id": "rs_1", "encrypted_content": "enc-1"}),)

    def test_tool_calls_stream_their_arguments_and_end_with_the_item(self):
        item = call_item("call_1", "get_weather", '{"city":"Paris"}', "fc_1")
        data = sse(
            CREATED,
            {"type": "response.output_item.added", "output_index": 0, "item": {**item, "arguments": "", "status": "in_progress"}},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"city"'},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": ':"Paris"}'},
            {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": '{"city":"Paris"}'},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            completed([item]),
        )

        events = decode(data)

        assert events[:4] == [
            ToolCallStart(block=0, call_id="call_1", name="get_weather", wire_id="fc_1"),
            ToolArgsDelta(block=0, text='{"city"'),
            ToolArgsDelta(block=0, text=':"Paris"}'),
            ToolCallEnd(block=0),
        ]
        part = assemble(events).parts[0]
        assert part == ToolCallPart(
            call_id="call_1", name="get_weather", args={"city": "Paris"}, args_raw='{"city":"Paris"}', wire_id="fc_1",
        )

    def test_arguments_that_arrive_only_on_the_finished_item_are_not_lost(self):
        part = assemble(decode(stream_of([call_item("call_1", "get_weather", '{"city":"Rome"}')]))).parts[0]

        assert part.args == {"city": "Rome"}
        assert part.args_raw == '{"city":"Rome"}'

    def test_a_call_without_arguments_runs_with_an_empty_object(self):
        part = assemble(decode(stream_of([call_item("call_1", "refresh", "")]))).parts[0]

        assert part.args == {}
        assert part.args_raw is None

    def test_arguments_that_are_not_an_object_leave_the_call_unrunnable(self):
        part = assemble(decode(stream_of([call_item("call_1", "refresh", "[1, 2]")]))).parts[0]

        assert part.args is None
        assert part.args_raw == "[1, 2]"

    def test_blocks_follow_the_output_order(self):
        output = [reasoning_item(), message_item("Checking", "commentary"), call_item()]

        message = assemble(decode(stream_of(output)))

        assert [type(part) for part in message.parts] == [ReasoningPart, TextPart, ToolCallPart]


class TestFraming:
    def test_one_byte_at_a_time_decodes_like_one_chunk(self):
        output = [reasoning_item("rs_1", "enc-é", ["Héllo."]), message_item("Zürich ✓"), call_item()]
        data = stream_of(output, {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7})

        whole = decode(data)
        bytewise = decode(data, [data[i:i + 1] for i in range(len(data))])

        assert bytewise == whole

    def test_carriage_return_line_feeds_frame_like_line_feeds(self):
        events = [CREATED, *item_events(0, message_item("Hi")), completed([message_item("Hi")])]

        assert decode(sse(*events, newline="\r\n")) == decode(sse(*events))

    def test_comments_done_markers_and_unknown_events_are_skipped(self):
        item = message_item("Hi")
        data = (
            b": keepalive\n\n"
            + sse(
                CREATED,
                {"type": "response.in_progress", "response": {}},
                {"type": "response.future_thing", "payload": 1},
                *item_events(0, item),
                {"type": "response.content_part.done", "output_index": 0},
                completed([item]),
            )
            + b"data: [DONE]\n\n"
        )

        assert decode(data) == decode(stream_of([item]))

    def test_events_after_the_response_completed_are_ignored(self):
        data = stream_of([message_item("Hi")]) + sse({"type": "response.output_text.delta", "output_index": 0, "delta": "late"})

        assert decode(data) == decode(stream_of([message_item("Hi")]))

    def test_a_final_event_cut_off_before_its_blank_line_is_not_trusted(self):
        data = stream_of([message_item("Hi")]).rstrip(b"\n")

        with pytest.raises(MalformedStream):
            decode(data)


class TestFinish:
    def test_the_finish_carries_the_assembled_output_array(self):
        output = [reasoning_item(), message_item("Hi"), call_item()]

        finish = finish_of(decode(stream_of(output)))

        assert finish.raw == output
        assert finish.reason is FinishReason.TOOL_CALLS
        assert finish.raw_reason == "completed"

    def test_an_answer_without_calls_stops(self):
        assert finish_of(decode(stream_of([message_item("Hi")]))).reason is FinishReason.STOP

    def test_items_only_present_in_the_completed_response_are_still_decoded(self):
        output = [reasoning_item("rs_1", "enc-1"), message_item("Only at the end", "final_answer")]
        data = sse(CREATED, completed(output))

        events = decode(data)
        message = assemble(events)

        assert message.text() == "Only at the end"
        assert message.raw == output

    def test_streamed_items_win_over_the_completed_copy(self):
        streamed = message_item("Streamed")
        data = sse(CREATED, *item_events(0, streamed), completed([message_item("Different")]))

        assert finish_of(decode(data)).raw == [streamed]

    def test_a_response_cut_off_by_the_token_limit_is_a_length_finish(self):
        item = {**message_item("Half an ans"), "status": "incomplete"}
        data = sse(
            CREATED,
            *item_events(0, item),
            {"type": "response.incomplete", "response": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": [item]}},
        )

        finish = finish_of(decode(data))

        assert finish.reason is FinishReason.LENGTH
        assert finish.raw_reason == "max_output_tokens"
        assert finish.raw == [item]

    def test_a_truncated_call_never_ends(self):
        item = {**call_item("call_1", "get_weather", '{"city":"Par'), "status": "incomplete"}
        data = sse(
            CREATED,
            *item_events(0, item),
            {"type": "response.incomplete", "response": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": [item]}},
        )

        events = decode(data)
        part = assemble(events).parts[0]

        assert not any(isinstance(event, ToolCallEnd) for event in events)
        assert part.args is None
        assert part.args_raw == '{"city":"Par'

    @pytest.mark.parametrize("reason, expected", [
        ("content_filter", FinishReason.CONTENT_FILTER),
        ("something_new", FinishReason.UNKNOWN),
        (None, FinishReason.UNKNOWN),
    ])
    def test_other_incomplete_reasons(self, reason, expected):
        details = {"reason": reason} if reason else None
        data = sse(CREATED, {"type": "response.incomplete", "response": {"status": "incomplete", "incomplete_details": details, "output": []}})

        finish = finish_of(decode(data))

        assert finish.reason is expected
        assert finish.raw_reason == (reason or "incomplete")

    def test_a_refusal_is_shown_as_the_answer_and_finishes_as_a_refusal(self):
        item = {
            "id": "msg_1", "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "refusal", "refusal": "I can't help with that."}],
        }
        data = sse(
            CREATED,
            {"type": "response.output_item.added", "output_index": 0, "item": {**item, "content": []}},
            {"type": "response.refusal.delta", "output_index": 0, "content_index": 0, "delta": "I can't help "},
            {"type": "response.refusal.delta", "output_index": 0, "content_index": 0, "delta": "with that."},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            completed([item]),
        )

        events = decode(data)

        assert assemble(events).text() == "I can't help with that."
        assert finish_of(events).reason is FinishReason.REFUSAL

    def test_an_item_that_never_finished_leaves_no_raw_payload(self):
        opening = {**message_item("Part"), "status": "in_progress", "content": []}
        data = sse(
            CREATED,
            {"type": "response.output_item.added", "output_index": 0, "item": opening},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "Part"},
            {"type": "response.incomplete", "response": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}},
        )

        finish = finish_of(decode(data))

        assert finish.raw is None
        assert assemble(decode(data)).raw is None


class TestUsage:
    def test_usage_maps_cached_written_and_reasoning_tokens(self):
        usage = {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 12},
            "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 25},
            "total_tokens": 140,
        }

        events = decode(stream_of([message_item("Hi")], usage))

        assert events[-2] == UsageReported(Usage(
            input_tokens=100, output_tokens=40, total_tokens=140, reasoning_tokens=25,
            cached_input_tokens=64, cache_write_tokens=12, raw=usage,
        ))

    def test_missing_details_stay_unknown(self):
        events = decode(stream_of([message_item("Hi")], {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11}))

        usage = events[-2].usage
        assert (usage.cached_input_tokens, usage.cache_write_tokens, usage.reasoning_tokens) == (None, None, None)

    def test_no_usage_means_no_usage_event(self):
        assert not any(isinstance(event, UsageReported) for event in decode(stream_of([message_item("Hi")])))


class TestFailures:
    def test_a_failed_response_raises_the_classified_error_from_feed(self):
        data = sse(
            CREATED,
            {"type": "response.failed", "response": {"status": "failed", "error": {"code": "server_error", "message": "Boom"}}},
        )
        decoder = ADAPTER.decoder(request(), make_profile())

        with pytest.raises(ServerFailed) as caught:
            decoder.feed(data)

        assert caught.value.message == "Boom"
        assert caught.value.detail["in_stream"] is True

    def test_events_before_the_failure_are_delivered(self):
        decoder = ADAPTER.decoder(request(), make_profile())
        opened = sse(CREATED, {"type": "response.output_item.added", "output_index": 0, "item": reasoning_item("rs_1", None)})

        assert decoder.feed(opened) == [ReasoningDelta(block=0, text="")]

    def test_an_error_event_is_classified_by_its_code(self):
        data = sse({"type": "error", "code": "rate_limit_exceeded", "message": "Slow down", "param": None})

        with pytest.raises(RateLimited):
            ADAPTER.decoder(request(), make_profile()).feed(data)

    def test_an_error_event_may_nest_its_error_object(self):
        data = sse({"type": "error", "error": {"code": "invalid_encrypted_content", "message": "Could not verify"}})

        with pytest.raises(OpaqueViolation):
            ADAPTER.decoder(request(), make_profile()).feed(data)

    def test_a_stream_that_stops_before_completing_is_malformed(self):
        data = sse(CREATED, *item_events(0, message_item("Hi")))

        with pytest.raises(MalformedStream):
            decode(data)

    def test_a_frame_with_invalid_json_is_malformed(self):
        with pytest.raises(MalformedStream):
            ADAPTER.decoder(request(), make_profile()).feed(b"data: {not json\n\n")

    def test_a_frame_that_is_not_utf8_is_malformed(self):
        with pytest.raises(MalformedStream):
            ADAPTER.decoder(request(), make_profile()).feed(b"data: \xff\xfe\n\n")

    def test_a_payload_that_is_not_an_object_is_malformed(self):
        with pytest.raises(MalformedStream):
            ADAPTER.decoder(request(), make_profile()).feed(b"data: [1, 2]\n\n")

    def test_an_item_event_without_an_index_is_malformed(self):
        data = b'data: {"type":"response.output_item.added","item":{"type":"message"}}\n\n'

        with pytest.raises(MalformedStream):
            ADAPTER.decoder(request(), make_profile()).feed(data)

    def test_an_item_event_without_an_item_is_malformed(self):
        data = b'data: {"type":"response.output_item.done","output_index":0}\n\n'

        with pytest.raises(MalformedStream):
            ADAPTER.decoder(request(), make_profile()).feed(data)


class TestBufferedBodies:
    def body(self, output, status="completed", **extra):
        return json.dumps({"id": "resp_1", "status": status, "output": output, **extra}).encode("utf-8")

    def decode_body(self, body):
        decoder = ADAPTER.decoder(make_request([user_message()], stream=False), make_profile())
        return decode_chunks(decoder, [body[:7], body[7:]])

    def test_a_buffered_response_decodes_like_the_stream(self):
        output = [reasoning_item("rs_1", "enc-1", ["Why."]), message_item("Hi"), call_item()]
        usage = {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}

        buffered = self.decode_body(self.body(output, usage=usage))

        assert assemble(buffered) == assemble(decode(stream_of(output, usage)))
        assert finish_of(buffered).raw == output

    def test_a_buffered_failure_raises(self):
        with pytest.raises(ServerFailed):
            self.decode_body(self.body([], "failed", error={"code": "server_error", "message": "Boom"}))

    def test_a_buffered_incomplete_response_is_a_length_finish(self):
        body = self.body([message_item("Hal")], "incomplete", incomplete_details={"reason": "max_output_tokens"})

        assert finish_of(self.decode_body(body)).reason is FinishReason.LENGTH

    @pytest.mark.parametrize("body", [b"", b"not json", b"[]", b'{"status":"queued"}'])
    def test_an_unusable_buffered_body_is_malformed(self, body):
        with pytest.raises(MalformedStream):
            self.decode_body(body)


class TestRoundTrip:
    def test_a_decoded_message_replays_into_the_next_request(self):
        output = [reasoning_item("rs_1", "enc-1", ["Plan."]), call_item("call_1", "get_weather", '{"city":"Oslo"}', "fc_1")]
        assistant = assistant_message(output)

        wire = ADAPTER.render(make_request([user_message(), assistant]), make_profile(reasoning="effort"))

        assert wire.body["input"][1:] == output

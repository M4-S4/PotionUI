import json

import pytest

from src.features.llm.harness.adapters.openrouter import DETAILS_KEY, DIGEST_KEY, details_digest
from src.features.llm.harness.errors import (
    AuthFailed,
    ConnectionFailed,
    MalformedStream,
    QuotaExhausted,
    RateLimited,
    ServerFailed,
)
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    OpaqueDelta,
    ReasoningDelta,
    TextDelta,
    UsageReported,
)
from tests.features.llm.harness.adapters.openrouter_support import ADAPTER, chunk, make_profile, make_request, sse, user
from tests.fixtures.llm_harness import assemble, decode_chunks


def decode(*chunks, split=None):
    decoder = ADAPTER.decoder(make_request([user()]), make_profile())
    data = b"".join(chunks)
    pieces = [data] if split is None else [data[i:i + split] for i in range(0, len(data), split)]
    return decode_chunks(decoder, pieces)


def of_type(events, kind):
    return [event for event in events if isinstance(event, kind)]


def streamed_reasoning():
    return sse(
        chunk({"role": "assistant", "content": ""}, provider="Anthropic"),
        chunk({"reasoning": "Both cities ", "reasoning_details": [
            {"type": "reasoning.text", "text": "Both cities ", "format": "anthropic-claude-v1", "index": 0},
        ]}),
        chunk({"reasoning": "need a lookup.", "reasoning_details": [
            {"type": "reasoning.text", "text": "need a lookup.", "format": "anthropic-claude-v1", "index": 0},
        ]}),
        chunk({"reasoning_details": [
            {"type": "reasoning.text", "text": "", "signature": "sig-1", "format": "anthropic-claude-v1", "index": 0},
            {"type": "reasoning.encrypted", "id": "rd-9", "data": "Zm9v", "format": "anthropic-claude-v1", "index": 1},
        ]}),
        chunk({"content": "Done."}),
        chunk({}, "stop"),
        "[DONE]",
        comments=["OPENROUTER PROCESSING"],
    )


class TestReasoningDetails:
    def test_the_blocks_are_merged_by_index_and_kept_in_order_on_the_finish(self):
        events = decode(streamed_reasoning())

        raw = of_type(events, Finish)[0].raw
        assert raw["provider"] == "Anthropic"
        assert raw[DETAILS_KEY] == [
            {
                "type": "reasoning.text",
                "text": "Both cities need a lookup.",
                "format": "anthropic-claude-v1",
                "index": 0,
                "signature": "sig-1",
            },
            {"type": "reasoning.encrypted", "id": "rd-9", "data": "Zm9v", "format": "anthropic-claude-v1", "index": 1},
        ]

    def test_the_visible_reasoning_comes_from_the_text_field_once(self):
        events = decode(streamed_reasoning())

        assert "".join(event.text for event in of_type(events, ReasoningDelta)) == "Both cities need a lookup."

    def test_the_digest_of_the_blocks_is_stored_on_the_reasoning_block(self):
        events = decode(streamed_reasoning())

        raw = of_type(events, Finish)[0].raw
        digests = [event for event in of_type(events, OpaqueDelta) if event.key == DIGEST_KEY]
        assert [event.value for event in digests] == [details_digest(raw[DETAILS_KEY])]

    def test_the_assembled_message_keeps_the_blocks_as_raw(self):
        message = assemble(decode(streamed_reasoning()))

        assert message.raw["provider"] == "Anthropic"
        assert len(message.raw[DETAILS_KEY]) == 2
        assert message.reasoning()[0].opaque[DIGEST_KEY] == details_digest(message.raw[DETAILS_KEY])

    def test_the_stream_decodes_the_same_when_cut_into_tiny_pieces(self):
        assert decode(streamed_reasoning(), split=7) == decode(streamed_reasoning())

    def test_blocks_without_a_text_field_still_show_their_text_and_summary(self):
        events = decode(sse(
            chunk({"reasoning_details": [{"type": "reasoning.text", "text": "Plain.", "index": 0}]}),
            chunk({"reasoning_details": [{"type": "reasoning.summary", "summary": "Short.", "index": 1}]}),
            chunk({}, "stop"),
            "[DONE]",
        ))

        shown = of_type(events, ReasoningDelta)
        assert [(event.text, event.summary) for event in shown] == [("Plain.", False), ("Short.", True)]

    def test_encrypted_only_reasoning_is_stored_as_a_redacted_block(self):
        events = decode(sse(
            chunk({"reasoning_details": [{"type": "reasoning.encrypted", "id": "rd-1", "data": "Zm9v", "index": 0}]}),
            chunk({"content": "Hi"}),
            chunk({}, "stop"),
            "[DONE]",
        ))

        message = assemble(events)
        assert message.reasoning()[0].redacted is True
        assert message.raw[DETAILS_KEY][0]["data"] == "Zm9v"

    def test_a_reply_without_blocks_has_no_raw(self):
        events = decode(sse(chunk({"reasoning": "Thinking."}), chunk({"content": "Hi"}), chunk({}, "stop"), "[DONE]"))

        assert of_type(events, Finish)[0].raw is None
        assert not [event for event in events if isinstance(event, OpaqueDelta)]

    def test_blocks_survive_a_tool_call_reply(self):
        events = decode(sse(
            chunk({"reasoning_details": [{"type": "reasoning.encrypted", "id": "rd-1", "data": "Zm9v", "index": 0}]}),
            chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}),
            chunk({}, "tool_calls"),
            "[DONE]",
        ))

        finish = of_type(events, Finish)[0]
        assert finish.reason is FinishReason.TOOL_CALLS
        assert finish.raw[DETAILS_KEY][0]["id"] == "rd-1"


class TestStreamFraming:
    def test_comment_lines_are_ignored(self):
        events = decode(sse(chunk({"content": "Hi"}), chunk({}, "stop"), "[DONE]", comments=["OPENROUTER PROCESSING", "keep-alive"]))

        assert [event.text for event in of_type(events, TextDelta)] == ["Hi"]

    def test_a_comment_only_prelude_before_the_first_event_is_fine(self):
        events = decode(b": OPENROUTER PROCESSING\n\n", sse(chunk({"content": "Hi"}), chunk({}, "stop"), "[DONE]"))

        assert of_type(events, Finish)[0].reason is FinishReason.STOP


class TestUsage:
    def test_cost_and_token_details_are_reported(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
            "cost": 0.00123,
            "prompt_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 16},
            "completion_tokens_details": {"reasoning_tokens": 25},
        }

        events = decode(sse(chunk({"content": "Hi"}), chunk({}, "stop"), {**chunk({}), "choices": [], "usage": usage}, "[DONE]"))

        reported = of_type(events, UsageReported)[0].usage
        assert reported.cost == 0.00123
        assert reported.cached_input_tokens == 64
        assert reported.cache_write_tokens == 16
        assert reported.reasoning_tokens == 25
        assert (reported.input_tokens, reported.output_tokens, reported.total_tokens) == (100, 40, 140)

    def test_a_usage_without_cost_leaves_it_unknown(self):
        events = decode(sse(chunk({"content": "Hi"}, usage={"prompt_tokens": 3, "completion_tokens": 1}), chunk({}, "stop"), "[DONE]"))

        assert of_type(events, UsageReported)[0].usage.cost is None


def midstream_error(code, message="Provider disconnected", **extra):
    return {
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "error": {"code": code, "message": message, **extra},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
    }


class TestInBandErrors:
    @pytest.mark.parametrize("code, expected", [
        (429, RateLimited),
        (402, QuotaExhausted),
        (401, AuthFailed),
        (403, AuthFailed),
        (408, ConnectionFailed),
        (502, ServerFailed),
        ("502", ServerFailed),
    ])
    def test_an_error_event_is_raised_as_its_typed_error(self, code, expected):
        with pytest.raises(expected) as caught:
            decode(sse(chunk({"content": "Par"}), midstream_error(code)))

        assert caught.value.status == int(code)
        assert caught.value.detail["in_stream"] is True

    def test_the_provider_name_from_the_metadata_is_kept(self):
        with pytest.raises(ServerFailed) as caught:
            decode(sse(midstream_error(502, metadata={"provider_name": "Anthropic", "raw": {"x": 1}})))

        assert "Anthropic" in caught.value.message
        assert caught.value.detail["metadata"]["raw"] == {"x": 1}

    def test_a_finish_reason_error_without_an_error_object_is_a_server_failure(self):
        with pytest.raises(ServerFailed):
            decode(sse(chunk({"content": "Par"}), chunk({}, "error")))

    def test_an_error_without_a_usable_code_is_a_server_failure(self):
        with pytest.raises(ServerFailed):
            decode(sse({"error": {"message": "Something broke"}}))

    def test_events_decoded_before_the_failure_were_already_delivered(self):
        decoder = ADAPTER.decoder(make_request([user()]), make_profile())

        first = decoder.feed(sse(chunk({"content": "Par"})))
        with pytest.raises(ServerFailed):
            decoder.feed(sse(midstream_error(502)))

        assert [event.text for event in first if isinstance(event, TextDelta)] == ["Par"]

    def test_a_plain_json_error_body_with_status_200_is_raised_when_it_arrives(self):
        body = json.dumps({"error": {"code": 402, "message": "Insufficient credits"}}).encode()
        decoder = ADAPTER.decoder(make_request([user()]), make_profile())

        with pytest.raises(QuotaExhausted):
            decoder.feed(body)

    def test_a_json_error_body_split_across_chunks_is_raised_once_complete(self):
        body = json.dumps({"error": {"code": 429, "message": "Slow down"}}).encode()
        decoder = ADAPTER.decoder(make_request([user()]), make_profile())

        assert decoder.feed(body[:10]) == []
        with pytest.raises(RateLimited):
            decoder.feed(body[10:])

    def test_a_plain_json_completion_to_a_streaming_request_is_malformed(self):
        body = json.dumps({"choices": [{"message": {"content": "Hi"}}]}).encode()
        decoder = ADAPTER.decoder(make_request([user()]), make_profile())

        with pytest.raises(MalformedStream):
            decoder.feed(body)

    def test_a_truncated_json_body_is_malformed_at_the_end(self):
        decoder = ADAPTER.decoder(make_request([user()]), make_profile())
        decoder.feed(b'{"error": {"code": 4')

        with pytest.raises(MalformedStream):
            decoder.close()

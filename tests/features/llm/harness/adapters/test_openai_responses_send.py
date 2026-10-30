import json

import httpx
import pytest

from src.features.llm.harness.adapter import Adapter, Settings
from src.features.llm.harness.errors import ConnectionFailed, OpaqueViolation, QuotaExhausted
from src.features.llm.harness.events import FinishReason
from src.features.llm.harness.messages import Origin, TextPart
from src.features.llm.harness.transport import HttpAdapter
from tests.features.llm.harness.adapters.openai_responses_support import (
    ADAPTER,
    REASONING_PROFILE,
    call_item,
    make_profile,
    make_request,
    message_item,
    reasoning_item,
    stream_of,
    user_message,
)
from tests.features.llm.harness.wire_peer import chunked, install_peer
from tests.fixtures.llm_harness import assemble


async def run(request, profile):
    wire = ADAPTER.render(request, profile)
    decoder = ADAPTER.decoder(request, profile)
    events = []
    async for chunk in ADAPTER.send(wire, request):
        events.extend(decoder.feed(chunk))
    events.extend(decoder.close())
    return events


def test_the_adapter_is_an_http_adapter_and_satisfies_the_adapter_protocol():
    assert isinstance(ADAPTER, HttpAdapter)
    assert isinstance(ADAPTER, Adapter)
    assert ADAPTER.name == "openai-responses"


async def test_a_streamed_answer_travels_through_send_render_and_decoder(monkeypatch):
    output = [reasoning_item("rs_1", "enc-1", ["Plan."]), message_item("Hi", "final_answer"), call_item()]
    data = stream_of(output, {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11})
    peer = install_peer(monkeypatch, chunked([data[i:i + 37] for i in range(0, len(data), 37)]))
    request = make_request([user_message()], settings=Settings(max_tokens=300, reasoning="low"))

    events = await run(request, make_profile(**REASONING_PROFILE))

    message = assemble(events)
    assert message.text() == "Hi"
    assert message.raw == output
    assert events[-1].reason is FinishReason.TOOL_CALLS
    sent = peer.requests[0]
    assert str(sent.url) == "https://api.openai.com/v1/responses"
    assert sent.headers["authorization"] == "Bearer test-key-a"
    assert sent.headers["accept"] == "text/event-stream"
    assert json.loads(sent.content)["max_output_tokens"] == 300
    assert sent.content == ADAPTER.render(request, make_profile(**REASONING_PROFILE)).content()


async def test_the_next_request_replays_what_the_first_response_returned(monkeypatch):
    output = [reasoning_item("rs_1", "enc-1"), message_item("Hi", "final_answer")]
    install_peer(monkeypatch, chunked([stream_of(output)]))
    profile = make_profile(**REASONING_PROFILE)
    first = make_request([user_message()])

    assistant = assemble(await run(first, profile), "asst_1", Origin.for_config(ADAPTER.name, first.config))
    second = make_request([user_message(), assistant, user_message("More", "user_2")])

    assert ADAPTER.render(second, profile).body["input"][1:3] == output


@pytest.mark.parametrize("status, body, expected", [
    (400, {"error": {"code": "invalid_encrypted_content", "message": "Encrypted content organization_id did not match."}}, OpaqueViolation),
    (429, {"error": {"code": "insufficient_quota", "message": "You exceeded your current quota."}}, QuotaExhausted),
])
async def test_a_refused_request_raises_the_typed_openai_error(monkeypatch, status, body, expected):
    install_peer(monkeypatch, chunked([json.dumps(body).encode("utf-8")], status=status, headers={"x-request-id": "req_7"}))

    with pytest.raises(expected) as caught:
        await run(make_request([user_message()]), make_profile())

    assert caught.value.status == status
    assert caught.value.detail["request_id"] == "req_7"


async def test_a_connection_that_drops_mid_stream_is_marked_as_such(monkeypatch):
    async def broken():
        yield stream_of([message_item("Hi")])[:60]
        raise httpx.ReadError("peer reset")

    install_peer(monkeypatch, lambda request: httpx.Response(200, content=broken()))

    with pytest.raises(ConnectionFailed) as caught:
        await run(make_request([user_message()]), make_profile())

    assert caught.value.retryable
    assert caught.value.detail["in_stream"] is True


async def test_a_buffered_request_goes_through_the_same_path(monkeypatch):
    body = json.dumps({"id": "resp_1", "status": "completed", "output": [message_item("Hi")]}).encode("utf-8")
    peer = install_peer(monkeypatch, chunked([body[:10], body[10:]]))
    request = make_request([user_message()], stream=False)

    events = await run(request, make_profile())

    assert assemble(events).parts == (TextPart("Hi", phase="final_answer"),)
    assert peer.requests[0].headers["accept"] != "text/event-stream"
    assert peer.body["stream"] is False

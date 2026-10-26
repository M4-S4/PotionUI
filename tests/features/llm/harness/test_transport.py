import json
from typing import List

import httpx
import pytest

from src.features.llm.harness.adapter import RenderRequest, WireRequest
from src.features.llm.harness.errors import (
    AuthFailed,
    ConnectionFailed,
    ContextOverflow,
    InvalidRequest,
    Overloaded,
    QuotaExhausted,
    RateLimited,
    ServerFailed,
    WireFailure,
)
from src.features.llm.harness.profile import ProfileLayer, ProfileSource
from src.features.llm.harness.transport import HttpAdapter, error_code, error_text, stream_timeout
from src.features.llm.repository import LLMConfig
from tests.features.llm.harness.wire_peer import chunked, install_peer


class EchoAdapter(HttpAdapter):
    name = "echo"

    def render(self, request, profile):
        raise NotImplementedError

    def decoder(self, request, profile):
        raise NotImplementedError

    async def discover(self, config):
        return ProfileLayer(ProfileSource.DISCOVERED, {})

    async def list_models(self, base_url, api_key=None):
        return []


CONFIG = LLMConfig(
    id="c", name="c", type="echo", enabled=True, base_url="http://peer.invalid:1", model="m",
    system_message="", timeout=7,
)
REQUEST = RenderRequest(config=CONFIG, system="", messages=())
WIRE = WireRequest(
    method="POST",
    url="http://peer.invalid:1/chat",
    headers={"Content-Type": "application/json", "Authorization": "Bearer k"},
    body={"model": "m", "text": "café"},
)


async def collect(adapter: HttpAdapter) -> List[bytes]:
    return [chunk async for chunk in adapter.send(WIRE, REQUEST)]


async def test_chunks_arrive_in_order_with_the_exact_request_bytes(monkeypatch):
    peer = install_peer(monkeypatch, chunked([b"one", b"two", b"three"]))

    assert await collect(EchoAdapter()) == [b"one", b"two", b"three"]

    sent = peer.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "http://peer.invalid:1/chat"
    assert sent.content == WIRE.content() == json.dumps(WIRE.body, ensure_ascii=False, separators=(",", ":")).encode()
    assert sent.headers["authorization"] == "Bearer k"
    assert sent.headers["content-type"] == "application/json"


async def test_the_read_timeout_is_unbounded_and_the_rest_follow_the_config(monkeypatch):
    peer = install_peer(monkeypatch, chunked([b"x"]))

    await collect(EchoAdapter())

    timeout = peer.timeouts[0]
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (7, None, 7, 7)
    assert stream_timeout(CONFIG).read is None


@pytest.mark.parametrize(
    "status, body, headers, expected",
    [
        (401, b'{"error":{"message":"bad key"}}', {}, AuthFailed),
        (403, b"forbidden", {}, AuthFailed),
        (429, b'{"error":{"message":"slow down"}}', {"Retry-After": "7"}, RateLimited),
        (429, b'{"error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}', {}, QuotaExhausted),
        (400, b'{"error":{"code":"context_length_exceeded","message":"too long"}}', {}, ContextOverflow),
        (400, b'{"type":"error","error":{"message":"prompt is too long: 250000 tokens"}}', {}, ContextOverflow),
        (413, b'{"error":{"message":"prompt is too long"}}', {}, ContextOverflow),
        (422, b'{"detail":"exceed_context_size_error"}', {}, ContextOverflow),
        (400, b'{"error":{"message":"bad field"}}', {}, InvalidRequest),
        (404, b"not found", {}, InvalidRequest),
        (408, b"", {}, ConnectionFailed),
        (500, b"boom", {}, ServerFailed),
        (502, b"", {}, ServerFailed),
        (503, b"busy", {"Retry-After": "3"}, Overloaded),
        (529, b'{"error":{"type":"overloaded_error","message":"Overloaded"}}', {}, Overloaded),
    ],
)
async def test_a_non_success_status_raises_the_matching_harness_error(monkeypatch, status, body, headers, expected):
    install_peer(monkeypatch, chunked([body], status=status, headers=headers))

    with pytest.raises(expected) as caught:
        await collect(EchoAdapter())

    assert type(caught.value) is expected
    assert caught.value.status == status


async def test_the_error_message_comes_from_the_provider_json_and_the_body_is_kept_as_detail(monkeypatch):
    install_peer(monkeypatch, chunked([b'{"error":{"message":"bad key","code":"invalid_api_key"}}'], status=401))

    with pytest.raises(AuthFailed) as caught:
        await collect(EchoAdapter())

    assert caught.value.message == "bad key"
    assert caught.value.code == "invalid_api_key"
    assert "invalid_api_key" in caught.value.detail["body"]


async def test_retry_after_is_read_from_the_response_headers(monkeypatch):
    install_peer(monkeypatch, chunked([b"slow"], status=429, headers={"Retry-After": "7"}))

    with pytest.raises(RateLimited) as caught:
        await collect(EchoAdapter())

    assert caught.value.retry_after == 7.0
    assert caught.value.retryable


async def test_a_refused_connection_becomes_a_retryable_connection_error(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    install_peer(monkeypatch, refuse)

    with pytest.raises(ConnectionFailed) as caught:
        await collect(EchoAdapter())

    assert caught.value.retryable
    assert "connection refused" in caught.value.message


async def test_a_failure_after_the_first_chunk_keeps_the_delivered_chunks_and_raises_a_connection_error(monkeypatch):
    async def body():
        yield b"first"
        raise httpx.ReadError("peer reset")

    install_peer(monkeypatch, lambda request: httpx.Response(200, content=body()))
    received: List[bytes] = []

    with pytest.raises(ConnectionFailed):
        async for chunk in EchoAdapter().send(WIRE, REQUEST):
            received.append(chunk)

    assert received == [b"first"]


async def test_closing_the_stream_midway_releases_the_response(monkeypatch):
    released: List[bool] = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"first"
            yield b"second"

        async def aclose(self):
            released.append(True)

    install_peer(monkeypatch, lambda request: httpx.Response(200, stream=Body()))
    stream = EchoAdapter().send(WIRE, REQUEST)

    assert await stream.__anext__() == b"first"
    await stream.aclose()

    assert released == [True]


def test_error_helpers_read_the_common_provider_shapes():
    assert error_text(b'{"error":{"message":"a"}}') == "a"
    assert error_text(b'{"error":"b"}') == "b"
    assert error_text(b'{"message":"c"}') == "c"
    assert error_text(b'{"detail":[{"msg":"x"},"d"]}') == "d"
    assert error_text(b"plain text") == "plain text"
    assert error_code(b'{"error":{"code":"x","type":"y"}}') == "x"
    assert error_code(b'{"error":{"type":"y"}}') == "y"
    assert error_code(b"not json") is None


def test_a_wire_failure_without_a_status_is_a_connection_error():
    error = EchoAdapter().classify_error(WireFailure(status=None, cause=TimeoutError("slow")))

    assert isinstance(error, ConnectionFailed)
    assert error.message == "slow"

import asyncio
import gzip
import json
import random
import string

import pytest
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from src.bootstrap.middleware import CompressibleGZipMiddleware


def _text(size: int) -> str:
    rng = random.Random(7)
    return " ".join("".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))) for _ in range(size // 6))


async def _call(app, path="/", accept="gzip", headers=()):
    raw = [(b"accept-encoding", accept.encode())] if accept else []
    raw.extend((k.encode(), v.encode()) for k, v in headers)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": raw,
        "client": ("test", 1),
        "server": ("test", 80),
    }
    sent = []

    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.sleep(3600)

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], {k.decode().lower(): v.decode() for k, v in start["headers"]}, body


def _app(*routes):
    return CompressibleGZipMiddleware(Starlette(routes=list(routes)))


def test_large_json_is_gzipped_at_level_six():
    payload = {"text": _text(60000)}
    encoded = json.dumps(payload, separators=(",", ":")).encode()

    async def endpoint(request):
        return Response(encoded, media_type="application/json")

    status, headers, body = asyncio.run(_call(_app(Route("/", endpoint))))

    assert headers["content-encoding"] == "gzip"
    assert "accept-encoding" in headers["vary"].lower()
    assert gzip.decompress(body) == encoded
    assert len(body) == len(gzip.compress(encoded, 6))
    assert body[8] == gzip.compress(encoded, 6)[8]
    assert body[8] != gzip.compress(encoded, 9)[8]


def test_small_json_is_not_compressed():
    async def endpoint(request):
        return JSONResponse({"ok": True})

    status, headers, body = asyncio.run(_call(_app(Route("/", endpoint))))

    assert "content-encoding" not in headers
    assert json.loads(body) == {"ok": True}


def test_text_and_svg_are_compressed():
    async def text(request):
        return PlainTextResponse(_text(5000))

    async def svg(request):
        return Response(_text(5000), media_type="image/svg+xml")

    app = _app(Route("/t", text), Route("/s", svg))

    assert asyncio.run(_call(app, "/t"))[1]["content-encoding"] == "gzip"
    assert asyncio.run(_call(app, "/s"))[1]["content-encoding"] == "gzip"


def test_file_response_keeps_identity_and_content_length(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(bytes(range(256)) * 8192)

    async def endpoint(request):
        return FileResponse(media, media_type="video/mp4")

    status, headers, body = asyncio.run(_call(_app(Route("/", endpoint))))

    assert status == 200
    assert "content-encoding" not in headers
    assert headers["content-length"] == str(media.stat().st_size)
    assert body == media.read_bytes()


def test_range_response_is_untouched(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(bytes(range(256)) * 8192)

    async def endpoint(request):
        return FileResponse(media, media_type="video/mp4")

    status, headers, body = asyncio.run(_call(_app(Route("/", endpoint)), headers=[("range", "bytes=0-1999")]))

    assert status == 206
    assert "content-encoding" not in headers
    assert headers["content-range"].startswith("bytes 0-1999/")
    assert body == media.read_bytes()[:2000]


def test_text_response_with_content_range_is_untouched():
    async def endpoint(request):
        return Response(_text(5000), status_code=206, media_type="text/plain", headers={"Content-Range": "bytes 0-4999/9999"})

    status, headers, body = asyncio.run(_call(_app(Route("/", endpoint))))

    assert status == 206
    assert "content-encoding" not in headers


def test_already_encoded_response_is_untouched():
    payload = gzip.compress(_text(5000).encode())

    async def endpoint(request):
        return Response(payload, media_type="text/plain", headers={"Content-Encoding": "gzip"})

    status, headers, body = asyncio.run(_call(_app(Route("/", endpoint))))

    assert headers["content-encoding"] == "gzip"
    assert body == payload


def test_no_compression_without_accept_encoding():
    async def endpoint(request):
        return Response(_text(5000), media_type="application/json")

    status, headers, body = asyncio.run(_call(_app(Route("/", endpoint)), accept=None))

    assert "content-encoding" not in headers
    assert body.decode() == _text(5000)


def test_event_stream_streams_chunk_by_chunk():
    async def scenario():
        gate = asyncio.Event()
        order = []

        async def events():
            yield b"data: " + b"x" * 1500 + b"\n\n"
            await asyncio.wait_for(gate.wait(), timeout=5)
            yield b"data: second\n\n"

        async def endpoint(request):
            return StreamingResponse(events(), media_type="text/event-stream")

        app = _app(Route("/", endpoint))
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
            "scheme": "http", "path": "/", "raw_path": b"/", "query_string": b"",
            "headers": [(b"accept-encoding", b"gzip")], "client": ("t", 1), "server": ("t", 80),
        }

        async def receive():
            await asyncio.sleep(3600)

        async def send(message):
            order.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                gate.set()

        await app(scope, receive, send)
        return order

    order = asyncio.run(scenario())
    start = order[0]
    headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    bodies = [m["body"] for m in order if m["type"] == "http.response.body" and m.get("body")]

    assert "content-encoding" not in headers
    assert bodies[0].startswith(b"data: x")
    assert bodies[1] == b"data: second\n\n"


def test_streamed_json_with_preset_content_length_is_gzipped_without_it():
    payload = json.dumps({"text": _text(60000)}).encode()

    async def endpoint(request):
        async def chunks():
            for start in range(0, len(payload), 10000):
                yield payload[start:start + 10000]

        return StreamingResponse(
            chunks(), media_type="application/json", headers={"Content-Length": str(len(payload))}
        )

    status, headers, body = asyncio.run(_call(_app(Route("/", endpoint))))

    assert headers["content-encoding"] == "gzip"
    assert "content-length" not in headers
    assert gzip.decompress(body) == payload


@pytest.mark.parametrize(
    ("media_type", "encoding"),
    [
        ("application/problem+json", "gzip"),
        ("application/javascript", "gzip"),
        ("application/octet-stream", None),
        ("image/png", None),
    ],
)
def test_compression_follows_the_media_type(media_type, encoding):
    body = _text(5000).encode()

    async def endpoint(request):
        return Response(body, media_type=media_type)

    status, headers, raw = asyncio.run(_call(_app(Route("/", endpoint))))

    assert headers.get("content-encoding") == encoding
    assert (gzip.decompress(raw) if encoding else raw) == body

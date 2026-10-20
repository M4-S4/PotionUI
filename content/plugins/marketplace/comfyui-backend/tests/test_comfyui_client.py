import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from backend.comfyui_backend import ComfyUIBackend
from backend.comfyui_client import ComfyUiCertificateError, ComfyUiClient
from backend.comfyui_config import ComfyUIBackendConfig
from backend.pipes.comfyui.main import ComfyUIPipe
from src.plugin_api import GenerationExecutionError

BASE = "http://127.0.0.1:8188"


class GatedResponse:
    def __init__(self, session, url, payload):
        self.session = session
        self.url = url
        self.payload = payload
        self.status = 200

    def raise_for_status(self):
        return None

    async def json(self):
        return self.payload

    async def read(self):
        return self.payload

    async def __aenter__(self):
        session = self.session
        if self.url.startswith(f"{BASE}/experiment/models/"):
            session.in_flight += 1
            session.max_in_flight = max(session.max_in_flight, session.in_flight)
            if session.in_flight >= session.expected:
                session.gate.set()
            try:
                await asyncio.wait_for(session.gate.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
            finally:
                session.in_flight -= 1
        return self

    async def __aexit__(self, *exc_info):
        return False


class GatedSession:
    def __init__(self, routes, expected):
        self.routes = routes
        self.expected = expected
        self.gate = asyncio.Event()
        self.in_flight = 0
        self.max_in_flight = 0
        self.kwargs = []

    def get(self, url, **kwargs):
        self.kwargs.append(kwargs)
        return GatedResponse(self, url, self.routes[url])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _backend():
    config = ComfyUIBackendConfig(
        id="c1", name="Comfy", host="127.0.0.1", port=8188, secure=False, timeout_seconds=30
    )
    return ComfyUIBackend(config)


@pytest.mark.asyncio
async def test_list_models_requests_folders_concurrently():
    routes = {
        f"{BASE}/models": ["checkpoints", "loras", "vae"],
        f"{BASE}/experiment/models/checkpoints": [{"name": "a.safetensors", "size": 1}],
        f"{BASE}/experiment/models/loras": [{"name": "b.safetensors", "size": 2}],
        f"{BASE}/experiment/models/vae": [{"name": "c.safetensors", "size": 3}],
    }
    session = GatedSession(routes, expected=3)
    with patch("aiohttp.ClientSession", return_value=session):
        models = await _backend().list_models()
    assert session.max_in_flight == 3
    assert sorted(m.filename for m in models) == ["a.safetensors", "b.safetensors", "c.safetensors"]


@pytest.mark.asyncio
async def test_client_session_uses_the_shared_connector():
    session = GatedSession({f"{BASE}/models": []}, expected=0)
    marker = object()
    with patch("aiohttp.ClientSession", return_value=session) as session_class, \
            patch("backend.comfyui_client.aiohttp_connector", return_value=marker):
        async with ComfyUiClient(BASE, 5) as client:
            await client.folders()
    assert session_class.call_args.kwargs["connector"] is marker


@pytest.mark.asyncio
async def test_pipe_session_uses_the_shared_connector():
    marker = object()
    pipe = ComfyUIPipe({"host": "127.0.0.1", "port": 8188, "workflow_file": "w.json", "field_mappings": []})
    with patch("aiohttp.ClientSession") as session_class, \
            patch("backend.pipes.comfyui.main.aiohttp_connector", return_value=marker):
        pipe._session()
    assert session_class.call_args.kwargs["connector"] is marker


@pytest.mark.asyncio
async def test_download_writes_bytes_without_blocking_sleep():
    pipe = ComfyUIPipe({"host": "127.0.0.1", "port": 8188, "workflow_file": "w.json", "field_mappings": []})
    pipe._view_bytes = AsyncMock(return_value=b"video-bytes")
    with patch.object(time, "sleep", side_effect=AssertionError("blocking sleep")):
        path = await pipe.download_file_from_comfy({"filename": "out.mp4"}, "video")
    try:
        assert path.suffix == ".mp4"
        assert path.read_bytes() == b"video-bytes"
    finally:
        Path(path).unlink()


@pytest.mark.asyncio
async def test_download_returns_none_when_view_fails():
    pipe = ComfyUIPipe({"host": "127.0.0.1", "port": 8188, "workflow_file": "w.json", "field_mappings": []})
    pipe._view_bytes = AsyncMock(return_value=None)
    assert await pipe.download_file_from_comfy({"filename": "out.mp4"}, "video") is None


class CertificateFailureSession:
    def get(self, url, **kwargs):
        raise aiohttp.ClientError("[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate")

    post = get

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


@pytest.mark.asyncio
async def test_client_turns_a_certificate_failure_into_the_readable_message():
    with patch("aiohttp.ClientSession", return_value=CertificateFailureSession()):
        async with ComfyUiClient("https://comfy.example:8188") as client:
            with pytest.raises(ComfyUiCertificateError, match="comfy.example:8188.*does not trust"):
                await client.folders()


@pytest.mark.asyncio
async def test_list_models_reports_the_certificate_message_as_a_connection_error():
    with patch("aiohttp.ClientSession", return_value=CertificateFailureSession()):
        with pytest.raises(ConnectionError, match="does not trust"):
            await _backend().list_models()


@pytest.mark.asyncio
async def test_pipe_upload_reports_the_certificate_message():
    pipe = ComfyUIPipe({"host": "comfy.example", "port": 8188, "workflow_file": "w.json", "field_mappings": []})
    with patch("aiohttp.ClientSession", return_value=CertificateFailureSession()):
        with pytest.raises(GenerationExecutionError, match="comfy.example:8188.*does not trust"):
            await pipe._upload_media("image", b"x", "a.png", "image/png", MagicMock())


class RecordingResponse:
    def __init__(self, status=200, text=""):
        self.status = status
        self._text = text

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class RecordingSession:
    def __init__(self, response=None, error=None):
        self.response = response or RecordingResponse()
        self.error = error
        self.calls = []

    def _send(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.error is not None:
            raise self.error
        return self.response

    def get(self, url, **kwargs):
        return self._send("get", url, **kwargs)

    def post(self, url, **kwargs):
        return self._send("post", url, **kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


@pytest.mark.asyncio
async def test_clear_vram_posts_free_with_its_body_and_timeout():
    from backend.api import clear_vram

    session = RecordingSession()
    with patch("aiohttp.ClientSession", return_value=session), \
            patch("backend.api._get_comfyui_base_url", return_value=BASE):
        result = await clear_vram(current_user=None)
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("post", f"{BASE}/free")
    assert kwargs["json"] == {"unload_models": True, "free_memory": True}
    assert kwargs["timeout"].total == 10
    assert result["success"] is True


@pytest.mark.asyncio
async def test_clear_vram_reports_a_non_200_response():
    from backend.api import clear_vram

    session = RecordingSession(RecordingResponse(503, "busy"))
    with patch("aiohttp.ClientSession", return_value=session), \
            patch("backend.api._get_comfyui_base_url", return_value=BASE):
        result = await clear_vram(current_user=None)
    assert result == {"success": False, "error": "ComfyUI returned 503: busy"}


@pytest.mark.asyncio
async def test_other_client_errors_propagate_unchanged():
    error = aiohttp.ClientError("connection reset")
    with patch("aiohttp.ClientSession", return_value=RecordingSession(error=error)):
        async with ComfyUiClient(BASE) as client:
            with pytest.raises(aiohttp.ClientError) as raised:
                await client.folders()
    assert raised.value is error
    assert not isinstance(raised.value, ComfyUiCertificateError)


@pytest.mark.asyncio
async def test_request_timeout_reaches_the_session_call():
    session = RecordingSession()
    with patch("aiohttp.ClientSession", return_value=session):
        async with ComfyUiClient(BASE) as client:
            async with client.get("/system_stats", request_timeout=7):
                pass
            async with client.get("/models"):
                pass
    assert session.calls[0][2]["timeout"].total == 7
    assert "timeout" not in session.calls[1][2]

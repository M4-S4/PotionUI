from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp

import pytest
from PIL import Image

from backend.pipes.comfyui.main import ComfyUIPipe
from src.plugin_api import GenerationExecutionError, PipeInput

KINDS = [
    ("image", "upload_image_to_comfyui", ".png"),
    ("video", "upload_video_to_comfyui", ".mp4"),
    ("audio", "upload_audio_to_comfyui", ".wav"),
]


def _pipe():
    return ComfyUIPipe({"host": "127.0.0.1", "port": 8188, "workflow_file": "w.json", "field_mappings": []})


def _media_file(tmp_path, suffix):
    path = tmp_path / f"media{suffix}"
    if suffix == ".png":
        Image.new("RGB", (4, 4)).save(path)
    else:
        path.write_bytes(b"data")
    return str(path)


async def _apply(pipe, value, cast):
    pipe.config["field_mappings"] = [[value, "7.inputs.file", cast]]
    workflow = {"7": {"inputs": {"file": "placeholder"}}}
    return await pipe.apply_field_mappings(workflow, PipeInput(input={}), Mock())


@pytest.mark.parametrize("kind,uploader,suffix", KINDS)
class TestMediaMapping:
    @pytest.mark.asyncio
    async def test_uploaded_filename_replaces_the_placeholder(self, tmp_path, kind, uploader, suffix):
        pipe = _pipe()
        setattr(pipe, uploader, AsyncMock(return_value="remote.bin"))
        result = await _apply(pipe, _media_file(tmp_path, suffix), kind)
        assert result["7"]["inputs"]["file"] == "remote.bin"

    @pytest.mark.asyncio
    async def test_required_missing_file_names_the_node(self, kind, uploader, suffix):
        pipe = _pipe()
        with pytest.raises(GenerationExecutionError, match="7: a required"):
            await _apply(pipe, "missing" + suffix, kind + "_required")

    @pytest.mark.asyncio
    async def test_required_rejected_upload_names_the_node(self, tmp_path, kind, uploader, suffix):
        pipe = _pipe()
        setattr(pipe, uploader, AsyncMock(return_value=None))
        with pytest.raises(GenerationExecutionError, match=f"rejected the required {kind} upload"):
            await _apply(pipe, _media_file(tmp_path, suffix), kind + "_required")

    @pytest.mark.asyncio
    async def test_optional_missing_file_keeps_the_placeholder(self, kind, uploader, suffix):
        pipe = _pipe()
        result = await _apply(pipe, "missing" + suffix, kind)
        assert result["7"]["inputs"]["file"] == "placeholder"

    @pytest.mark.asyncio
    async def test_optional_rejected_upload_keeps_the_placeholder(self, tmp_path, kind, uploader, suffix):
        pipe = _pipe()
        setattr(pipe, uploader, AsyncMock(return_value=None))
        result = await _apply(pipe, _media_file(tmp_path, suffix), kind)
        assert result["7"]["inputs"]["file"] == "placeholder"


class TestImageValueShapes:
    @pytest.mark.asyncio
    async def test_legacy_bytes_dict_is_uploaded(self):
        import io

        buffer = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buffer, format="PNG")
        pipe = _pipe()
        pipe.upload_image_to_comfyui = AsyncMock(return_value="legacy.png")
        result = await _apply(pipe, {"data": buffer.getvalue()}, "image")
        assert result["7"]["inputs"]["file"] == "legacy.png"

    @pytest.mark.asyncio
    async def test_unsupported_value_is_skipped(self):
        pipe = _pipe()
        pipe.upload_image_to_comfyui = AsyncMock()
        result = await _apply(pipe, 42, "image")
        assert result["7"]["inputs"]["file"] == "placeholder"
        pipe.upload_image_to_comfyui.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_video_rejects_non_path_values(self):
        pipe = _pipe()
        result = await _apply(pipe, {"data": b"x"}, "video")
        assert result["7"]["inputs"]["file"] == "placeholder"


class TestUploadContentType:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,uploader,expected", [
        ("clip.webm", "upload_video_to_comfyui", "video/webm"),
        ("clip.MKV", "upload_video_to_comfyui", "video/x-matroska"),
        ("song.wav", "upload_audio_to_comfyui", "audio/wav"),
        ("song.flac", "upload_audio_to_comfyui", "audio/flac"),
        ("song.m4a", "upload_audio_to_comfyui", "audio/mp4"),
        ("clip.unknownext", "upload_video_to_comfyui", "video/mp4"),
        ("song.mp3", "upload_audio_to_comfyui", "audio/mpeg"),
        ("song.unknownext", "upload_audio_to_comfyui", "audio/mpeg"),
    ])
    async def test_content_type_follows_the_extension(self, tmp_path, name, uploader, expected):
        pipe = _pipe()
        pipe._upload_media = AsyncMock(return_value="x")
        path = tmp_path / name
        path.write_bytes(b"data")
        await getattr(pipe, uploader)(path, Mock())
        assert pipe._upload_media.await_args.args[3] == expected


class TestEmitMedia:
    @staticmethod
    def _pipe_with_downloads():
        pipe = _pipe()
        pipe.current_seed = 11
        pipe.download_file_from_comfy = AsyncMock(return_value=Path("v.mp4"))
        pipe.load_image_from_comfy = AsyncMock(return_value=Image.new("RGB", (2, 2)))
        return pipe

    def test_fresh_pipe_has_a_seed_slot(self):
        assert _pipe().current_seed is None

    @pytest.mark.asyncio
    async def test_final_gif_is_collected_with_fps_and_seed(self):
        pipe = self._pipe_with_downloads()
        out = Mock()
        images, videos = [], []
        await pipe._emit_media({"filename": "a.gif", "frame_rate": 12.0}, out, gifs=True, final=True, images=images, videos=videos)
        output = out.call_args.args[0]
        assert videos == [Path("v.mp4")] and not images
        assert (output.temporary, output.fps, output.seed) == (True, 12.0, 11)

    @pytest.mark.asyncio
    async def test_final_video_by_extension_is_not_temporary(self):
        pipe = self._pipe_with_downloads()
        out = Mock()
        images, videos = [], []
        await pipe._emit_media({"filename": "a.webm"}, out, final=True, video_temporary=False, images=images, videos=videos)
        assert videos == [Path("v.mp4")]
        assert out.call_args.args[0].temporary is False
        pipe.load_image_from_comfy.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_final_image_is_collected_temporary_with_seed(self):
        pipe = self._pipe_with_downloads()
        out = Mock()
        images, videos = [], []
        await pipe._emit_media({"filename": "a.png"}, out, final=True, images=images, videos=videos)
        output = out.call_args.args[0]
        assert len(images) == 1 and not videos
        assert (output.temporary, output.seed) == (True, 11)

    @pytest.mark.asyncio
    async def test_preview_image_is_not_collected_and_has_no_seed(self):
        pipe = self._pipe_with_downloads()
        out = Mock()
        await pipe._emit_media({"filename": "a.png"}, out)
        output = out.call_args.args[0]
        assert (output.temporary, output.seed) == (True, None)

    @pytest.mark.asyncio
    async def test_failed_download_emits_nothing(self):
        pipe = self._pipe_with_downloads()
        pipe.download_file_from_comfy = AsyncMock(return_value=None)
        out = Mock()
        await pipe._emit_media({"filename": "a.mp4"}, out)
        out.assert_not_called()


class _UploadResponse:
    def __init__(self, status, payload=None, text=""):
        self.status = status
        self._payload = payload
        self._text = text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _upload_session(response=None, error=None):
    session = MagicMock()
    if error is not None:
        session.post.side_effect = error
    else:
        session.post.return_value = response
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


class TestUploadMedia:
    @pytest.mark.asyncio
    async def test_success_returns_the_server_name(self):
        session = _upload_session(_UploadResponse(200, {"name": "server.png"}))
        with patch("aiohttp.ClientSession", return_value=session):
            name = await _pipe()._upload_media("image", b"x", "a.png", "image/png", Mock())
        assert name == "server.png"
        assert session.post.call_args.args[0] == "http://127.0.0.1:8188/upload/image"

    @pytest.mark.asyncio
    async def test_rejection_returns_none_and_reports_progress(self):
        out = Mock()
        session = _upload_session(_UploadResponse(500, text="boom"))
        with patch("aiohttp.ClientSession", return_value=session):
            name = await _pipe()._upload_media("video", b"x", "a.mp4", "video/mp4", out)
        assert name is None
        assert "Failed to upload video: 500" in out.call_args.args[0].state

    @pytest.mark.asyncio
    async def test_transport_error_names_the_server(self):
        session = _upload_session(error=aiohttp.ClientConnectionError("refused"))
        with patch("aiohttp.ClientSession", return_value=session):
            with pytest.raises(GenerationExecutionError, match="127.0.0.1:8188.*upload the audio"):
                await _pipe()._upload_media("audio", b"x", "a.wav", "audio/wav", Mock())


class TestTimelineMedia:
    def test_absolute_path_wins(self, tmp_path):
        target = tmp_path / "a.png"
        target.write_bytes(b"x")
        assert _pipe()._resolve_timeline_media({"path": str(target)}) == target

    def test_relative_path_is_used_when_path_is_missing(self, tmp_path):
        target = tmp_path / "b.png"
        target.write_bytes(b"x")
        media = {"path": str(tmp_path / "gone.png"), "relative_path": str(target)}
        assert _pipe()._resolve_timeline_media(media) == target

    def test_storage_directory_from_backend_config_is_searched(self, tmp_path):
        (tmp_path / "uploads").mkdir()
        (tmp_path / "uploads" / "c.png").write_bytes(b"x")
        pipe = _pipe()
        pipe.config["backend_config"] = {"file_storage_directory": str(tmp_path)}
        assert pipe._resolve_timeline_media({"relative_path": "uploads/c.png"}) == tmp_path / "uploads" / "c.png"

    def test_nothing_found_is_none(self):
        assert _pipe()._resolve_timeline_media({"path": "nope/x.png"}) is None


class TestImageValueEdges:
    @pytest.mark.asyncio
    async def test_pil_image_value_is_uploaded(self):
        pipe = _pipe()
        pipe.upload_image_to_comfyui = AsyncMock(return_value="pil.png")
        result = await _apply(pipe, Image.new("RGB", (2, 2)), "image")
        assert result["7"]["inputs"]["file"] == "pil.png"

    @pytest.mark.asyncio
    async def test_pil_upload_failure_propagates(self):
        pipe = _pipe()
        pipe.upload_image_to_comfyui = AsyncMock(side_effect=RuntimeError("disk"))
        with pytest.raises(RuntimeError):
            await _apply(pipe, Image.new("RGB", (2, 2)), "image")

    @pytest.mark.asyncio
    async def test_dict_without_bytes_is_skipped(self):
        pipe = _pipe()
        pipe.upload_image_to_comfyui = AsyncMock()
        result = await _apply(pipe, {"data": "not-bytes"}, "image")
        assert result["7"]["inputs"]["file"] == "placeholder"
        pipe.upload_image_to_comfyui.assert_not_awaited()

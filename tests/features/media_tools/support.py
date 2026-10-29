import shutil
import threading
import time
import unittest
import uuid
from datetime import datetime
from pathlib import Path

import psutil
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.bootstrap.errors import limit_exceeded_handler
from src.features.generation.file_repository import FileRepository
from src.features.generation.records import File
from src.features.media.media_types import MediaTypeResolver
from src.features.media.records import Upload
from src.features.media.upload_repository import UploadRepository
from src.features.media_tools.jobs import MediaToolJobManager
from src.features.media_tools.repository import MediaToolParamsRepository
from src.features.media_tools.routes import MediaToolController, build_router
from src.features.media_tools.service import MediaToolService
from src.features.plans.errors import LimitExceeded
from src.features.tags.repository import TagRepository
from src.platform.filesystem.storage_driver import LocalFileStorageDriver
from src.platform.media_tools import (
    FilterFragment,
    MediaTool,
    MediaToolRegistry,
    ToolCancelled,
    ToolSpec,
)
from src.platform.security.current_user import get_current_active_user
from src.platform.security.user import AccountType, User
from src.platform.util.ids import generate_ulid
from tests.fixtures.persistence_base import PersistenceTestBase
from tests.platform.media_tools.media_factory import ffmpeg_available, make_audio, make_image, make_video

needs_ffmpeg = unittest.skipUnless(ffmpeg_available(), "ffmpeg is not available")

TERMINAL = ("completed", "failed", "cancelled")
WAIT_SECONDS = 20.0


def limit_error() -> LimitExceeded:
    return LimitExceeded(
        [{
            "message": "Upload limit reached",
            "code": "storage_full",
            "kind": "storage",
            "label": "Storage",
            "format": "bytes",
            "used": 1,
            "limit": 1,
            "percent": 100,
            "resets_at": None,
        }],
        "upload",
        None,
    )


class FakeGuard:
    def __init__(self):
        self.refuse_check = False
        self.refuse_admit = False
        self.checks = []
        self.admits = []

    def check(self, request):
        self.checks.append(request)
        if self.refuse_check:
            raise limit_error()

    def admit(self, request):
        self.admits.append(request)
        if self.refuse_admit:
            raise limit_error()
        return []


class Gate:
    def __init__(self):
        self.entered = threading.Semaphore(0)
        self.release = threading.Event()
        self.seen = []
        self.lock = threading.Lock()


class Shrink(MediaTool):
    def image(self, image, params):
        return image.resize((params["size"], params["size"]))


class Grow(MediaTool):
    def __init__(self):
        self.seen = []

    def image(self, image, params):
        self.seen.append(image.size)
        return image.resize((params["size"], params["size"]))


class Rot(MediaTool):
    def image(self, image, params):
        return image.transpose(Image.Transpose.ROTATE_90)


class KeepAlpha(MediaTool):
    def image(self, image, params):
        return image.copy()


class BlockingTool(MediaTool):
    def __init__(self, gate):
        self.gate = gate

    def run(self, inputs, params, ctx):
        with self.gate.lock:
            self.gate.seen.append(inputs[0].path.name)
        self.gate.entered.release()
        while not self.gate.release.wait(0.01):
            if ctx.is_cancelled():
                raise ToolCancelled()
        out = ctx.new_path(inputs[0].path.suffix)
        shutil.copyfile(inputs[0].path, out)
        return [ctx.probe(out, inputs[0].kind)]


class Exploding(MediaTool):
    def run(self, inputs, params, ctx):
        raise RuntimeError("secret internal path /srv/private/thing")


class LongEncode(MediaTool):
    marker = "killme"

    def __init__(self):
        self.out = None
        self.token = uuid.uuid4().hex

    def run(self, inputs, params, ctx):
        self.out = ctx.new_path(".mp4", stem=f"{self.marker}-{self.token}")
        ctx.ffmpeg([
            "-re", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=30",
            "-c:v", "libx264", "-preset", "ultrafast", str(self.out),
        ])
        return [ctx.probe(self.out, "video")]


class SplitTwo(MediaTool):
    def run(self, inputs, params, ctx):
        outputs = []
        for index in range(2):
            out = ctx.new_path(inputs[0].path.suffix, stem=f"part{index}")
            shutil.copyfile(inputs[0].path, out)
            outputs.append(ctx.probe(out, "audio"))
        return outputs


class Stack(MediaTool):
    def run(self, inputs, params, ctx):
        images = [Image.open(media.path).convert("RGB") for media in inputs]
        width = sum(image.width for image in images)
        height = max(image.height for image in images)
        canvas = Image.new("RGB", (width, height))
        offset = 0
        for image in images:
            canvas.paste(image, (offset, 0))
            offset += image.width
        out = ctx.new_path(".png")
        canvas.save(out)
        return [ctx.probe(out, "image")]


class Noop(MediaTool):
    def video_filter(self, media, params):
        return FilterFragment()


class PadPx(MediaTool):
    def __init__(self):
        self.seen = []

    def image(self, image, params):
        self.seen.append((params["width"], params["height"], image.size))
        return Image.new("RGB", (params["width"], params["height"]), (10, 200, 10))


class FanMany(MediaTool):
    def run(self, inputs, params, ctx):
        outputs = []
        for index in range(params["n"]):
            out = ctx.new_path(inputs[0].path.suffix, stem=f"fan{index}")
            shutil.copyfile(inputs[0].path, out)
            outputs.append(ctx.probe(out, inputs[0].kind))
        return outputs


def build_registry(gate):
    registry = MediaToolRegistry()
    size = {"name": "size", "type": "integer", "label": "Size", "default": 8, "min": 1, "max": 64}
    registry.register(ToolSpec(
        id="shrink", label="Shrink", accepts=("image",), params=[size], preview=True,
        name_suffix="-shrink", impl=Shrink(),
    ))
    registry.register(ToolSpec(
        id="grow", label="Grow", accepts=("image",), preview=True, impl=Grow(),
        params=[{"name": "size", "type": "integer", "label": "Size", "default": 900, "min": 1, "max": 4000}],
    ))
    registry.register(ToolSpec(
        id="rot", label="Rot", accepts=("image", "video"), preview=True, impl=Rot(),
    ))
    registry.register(ToolSpec(
        id="shrink2", label="Shrink again", accepts=("image",), params=[size], name_suffix="-again",
        impl=Shrink(),
    ))
    registry.register(ToolSpec(
        id="needsalpha", label="Needs alpha", accepts=("image",), requires={"has_alpha": True},
        impl=KeepAlpha(),
    ))
    registry.register(ToolSpec(
        id="block", label="Block", accepts=("image",), impl=BlockingTool(gate),
    ))
    registry.register(ToolSpec(
        id="explode", label="Explode", accepts=("image",), impl=Exploding(),
    ))
    registry.register(ToolSpec(
        id="longencode", label="Long encode", accepts=("image",), produces="video", impl=LongEncode(),
    ))
    registry.register(ToolSpec(
        id="splittwo", label="Split two", accepts=("audio",), cardinality="one_to_many",
        name_suffix="-part", impl=SplitTwo(),
    ))
    registry.register(ToolSpec(
        id="stack", label="Stack", accepts=("image",), cardinality="many_to_one",
        selection={"min": 2, "max": 3}, impl=Stack(),
    ))
    registry.register(ToolSpec(
        id="toimage", label="To image", accepts=("video",), produces="image", impl=Noop(),
    ))
    pixel = {"type": "integer", "default": 64, "min": 1, "max": 8192, "unit": "px"}
    registry.register(ToolSpec(
        id="padpx", label="Pad px", accepts=("image",), preview=True, impl=PadPx(),
        params=[{"name": "width", "label": "Width", **pixel}, {"name": "height", "label": "Height", **pixel}],
    ))
    registry.register(ToolSpec(
        id="fanmany", label="Fan many", accepts=("image",), cardinality="one_to_many", name_suffix="-fan",
        impl=FanMany(),
        params=[{"name": "n", "type": "integer", "label": "Copies", "default": 2, "min": 1, "max": 600}],
    ))
    registry.register(ToolSpec(
        id="acme.panel", label="Panel", accepts=("image", "video"), component="plugin:acme:panel",
        source="acme", scopes=("history",), selection={"min": 1, "max": 4},
    ))
    registry.register(ToolSpec(
        id="acme.mirror", label="Mirror", accepts=("image",), source="acme", impl=Shrink(),
        params=[size],
    ))
    return registry


class MediaToolsTestBase(PersistenceTestBase):
    jobs_clock = None

    def setUp(self):
        super().setUp()
        self.addCleanup(self.kill_stray_processes)
        self.storage_dir = Path(self.temp_dir) / "storage"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_root = Path(self.temp_dir) / "scratch-root"
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self.uploads_dir = self.storage_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        self.upload_repo = UploadRepository()
        self.file_repo = FileRepository()
        self.tag_repo = TagRepository()
        self.params_repo = MediaToolParamsRepository()
        self.driver = LocalFileStorageDriver(str(self.storage_dir))
        self.guard = FakeGuard()
        self.gate = Gate()
        self.registry = build_registry(self.gate)
        self.clock = FakeClock()
        self.jobs = MediaToolJobManager(clock=self.clock)
        self.service = self.make_service(self.driver)

        self.user_id = self.create_test_user()
        self.other_user_id = self.create_test_user("other_user", "other", "other@example.com")
        self.acting_user_id = self.user_id
        self.client = self.make_client(self.service)

    def make_service(self, driver):
        return MediaToolService(
            registry=self.registry,
            params_repository=self.params_repo,
            upload_repository=self.upload_repo,
            file_repository=self.file_repo,
            media_type_resolver=MediaTypeResolver(),
            storage_driver=driver,
            limit_guard=self.guard,
            jobs=self.jobs,
            scratch_root=self.scratch_root,
        )

    def make_client(self, service):
        class Container:
            media_tool_controller = MediaToolController(service)

        app = FastAPI()
        app.include_router(build_router(Container()))
        app.add_exception_handler(LimitExceeded, limit_exceeded_handler)

        async def current_user():
            return self.make_user(self.acting_user_id)

        app.dependency_overrides[get_current_active_user] = current_user
        return TestClient(app)

    def kill_stray_processes(self):
        marker = str(self.temp_dir)
        for process in psutil.process_iter(["cmdline"]):
            try:
                if any(marker in part for part in (process.info["cmdline"] or [])):
                    process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def tearDown(self):
        self.gate.release.set()
        self.kill_stray_processes()
        self.jobs.join_all()
        self.client.close()
        self.remove_scratch_files()
        super().tearDown()

    def make_user(self, user_id):
        return User(
            id=user_id, username=user_id, email=f"{user_id}@example.com",
            password_hash="h", account_type=AccountType.USER,
        )

    def as_user(self, user_id):
        self.acting_user_id = user_id

    def image_item(self, name="pic.png", user_id=None, size=(40, 30), mode="RGB", color=(200, 40, 40)):
        path = make_image(self.uploads_dir / name, size, mode, color)
        return self.record(path, "image", user_id, width=size[0], height=size[1])

    def video_item(self, name="clip.mp4", user_id=None, seconds=1.0):
        path = make_video(self.uploads_dir / name, seconds=seconds, size=(64, 48))
        return self.record(path, "video", user_id, width=64, height=48, duration_seconds=seconds)

    def audio_item(self, name="sound.wav", user_id=None, seconds=1.0):
        path = make_audio(self.uploads_dir / name, seconds=seconds)
        return self.record(path, "audio", user_id, duration_seconds=seconds)

    def mesh_item(self, name="thing.glb", user_id=None):
        path = self.uploads_dir / name
        path.write_bytes(b"glTF-ish bytes")
        return self.record(path, "model", user_id)

    def record(self, path, media_type, user_id=None, **metadata):
        return self.upload_repo.create(Upload(
            user_id=user_id or self.user_id,
            filename=path.name,
            original_filename=f"original-{path.name}",
            media_type=media_type,
            mime_type=MediaTypeResolver().get_media_type(path.suffix),
            file_size=path.stat().st_size,
            **metadata,
        ))

    def generation_file(self, user_id=None, suffix=".png", file_type="IMAGE", size=(32, 24)):
        owner = user_id or self.user_id
        generation_id = generate_ulid()
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO generations (id, preset_id, form_data, user_id, status) VALUES (?, ?, ?, ?, ?)",
                (generation_id, "test_preset", '{"prompt": "a cat"}', owner, "completed"),
            )
        directory = self.storage_dir / "generations" / datetime.now().strftime("%Y-%m-%d") / generation_id
        directory.mkdir(parents=True, exist_ok=True)
        on_disk = directory / f"0{suffix}"
        if file_type == "IMAGE":
            make_image(on_disk, size)
        else:
            on_disk.write_bytes(b"glTF-ish bytes")
        record = self.file_repo.create(File(
            file_path=on_disk.relative_to(self.storage_dir).as_posix(),
            file_type=file_type,
            user_id=owner,
            mime_type="image/png",
            file_size=on_disk.stat().st_size,
            is_final=True,
            width=size[0],
            height=size[1],
        ))
        self.file_repo.associate_with_generation(generation_id, record.id)
        return record, on_disk

    def run_body(self, steps, inputs, mode="new"):
        return {"steps": steps, "inputs": inputs, "output": {"mode": mode}}

    def start(self, steps, inputs, mode="new"):
        return self.client.post("/api/media-tools/runs", json=self.run_body(steps, inputs, mode))

    def start_ok(self, steps, inputs, mode="new"):
        response = self.start(steps, inputs, mode)
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["data"]["run_id"]

    def wait_run(self, run_id, user_id=None, timeout=WAIT_SECONDS):
        run = self.jobs.wait(user_id or self.user_id, run_id, timeout)
        self.assertIsNotNone(run)
        self.assertIn(run.status, TERMINAL, f"run still {run.status}")
        return run

    def finish(self, steps, inputs, mode="new"):
        return self.wait_run(self.start_ok(steps, inputs, mode))

    def upload_rows(self, user_id=None):
        return self.upload_repo.list_for_user(user_id or self.user_id, limit=500)

    def stored_files(self):
        return sorted(path.name for path in self.uploads_dir.iterdir() if path.is_file())

    def thumbnail_files(self):
        directory = self.uploads_dir / "thumbnails"
        return sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []

    def scratch_entries(self):
        return sorted(path.name for path in self.scratch_root.iterdir())

    def error_of(self, response):
        return response.json()["detail"]

    def wait_for(self, condition, timeout=WAIT_SECONDS):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = condition()
            if value:
                return value
            threading.Event().wait(0.02)
        self.fail("condition not met in time")

    def encode_processes(self, marker):
        found = []
        for process in psutil.process_iter(["cmdline"]):
            try:
                if any(marker in part for part in (process.info["cmdline"] or [])):
                    found.append(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return found


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

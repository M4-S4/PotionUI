import shutil
import tempfile
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

from src.pipelines.outputs import Icon, Progress, ProgressGenerationOutput
from src.platform.media_tools import MediaToolRegistry, ToolContext
from src.platform.media_tools.ops import register_core_tools
from src.platform.media_tools.ops.concat import ClipTrims, join_clips

SCRATCH_PREFIX = "potionui-media-tool-"
DEFAULT_ICON = "scissors"


@lru_cache(maxsize=1)
def core_registry() -> MediaToolRegistry:
    registry = MediaToolRegistry()
    register_core_tools(registry)
    return registry


def resolve_registry(service: object) -> MediaToolRegistry:
    return service if isinstance(service, MediaToolRegistry) else core_registry()


@contextmanager
def scratch_dir() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def keep_file(path: Path, prefix: str) -> str:
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=path.suffix, delete=False)
    handle.close()
    shutil.move(str(path), handle.name)
    return handle.name


class ProgressRelay:
    def __init__(self, emit: Callable, label: str, icon: str = DEFAULT_ICON, effect: Optional[str] = None):
        self._emit = emit
        self._label = label
        self._icon = Icon(icon, effect)
        self._lock = threading.Lock()
        self._latest = (0.0, label)
        self._sent: Optional[tuple[int, str]] = None

    def update(self, fraction: float, label: str = "") -> None:
        with self._lock:
            self._latest = (fraction, label or self._label)

    def flush(self) -> None:
        with self._lock:
            fraction, label = self._latest
        marker = (int(max(0.0, min(1.0, fraction)) * 100), label)
        if marker == self._sent:
            return
        self._sent = marker
        self._emit(ProgressGenerationOutput(state=label, icon=self._icon, progress=Progress(marker[0], 100)))

    def context(self, scratch: Path, is_cancelled: Optional[Callable[[], bool]] = None) -> ToolContext:
        def poll() -> bool:
            self.flush()
            return bool(is_cancelled is not None and is_cancelled())

        return ToolContext(scratch, progress=self.update, is_cancelled=poll)


def join_video_files(
    clips: Sequence[str],
    out_path: str | Path,
    *,
    emit: Callable,
    is_cancelled: Optional[Callable[[], bool]] = None,
    label: str = "Joining the clips",
    trims: Optional[ClipTrims] = None,
    crossfades: Optional[Sequence[float]] = None,
    canvas: int = 0,
    fit: str = "pad",
    audio: bool = True,
) -> Path:
    relay = ProgressRelay(emit, label, icon="film", effect="pulse")
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with scratch_dir() as scratch:
        ctx = relay.context(scratch, is_cancelled)
        media = [ctx.probe(clip, "video") for clip in clips]
        joined = join_clips(media, ctx, trims=trims, crossfades=crossfades, canvas=canvas, fit=fit, audio=audio)
        relay.flush()
        shutil.move(str(joined.path), str(destination))
    return destination

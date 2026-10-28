import logging
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

from src.platform.util.video_transcode import VideoProbe, find_ffmpeg, probe_video

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
SAMPLE_RATE = 48000
FFMPEG_TIMEOUT_SECONDS = 1800
DEFAULT_FPS = 24.0
POLL_SECONDS = 0.2
TERMINATE_GRACE_SECONDS = 5.0
EDGE_NUDGE_SECONDS = 0.001
NO_SOUND_NOTICE = "ffmpeg could not join the clips, so the joined video has no sound"


def as_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def video_paths(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(item) for item in value if item]


class VideoConcatError(RuntimeError):
    pass


class VideoConcatCancelled(Exception):
    pass


@dataclass(frozen=True)
class FinishedProcess:
    returncode: int
    stderr: bytes


@dataclass(frozen=True)
class ClipTrim:
    start: float = 0.0
    end: float = 0.0


def _stop(process: "subprocess.Popen") -> None:
    process.terminate()
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_cancellable(
    command: Sequence[str],
    *,
    timeout: float = FFMPEG_TIMEOUT_SECONDS,
    is_cancelled: Optional[Callable[[], bool]] = None,
    **_ignored,
) -> FinishedProcess:
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(list(command), stdout=subprocess.DEVNULL, stderr=errors)
        try:
            while True:
                try:
                    process.wait(timeout=POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if is_cancelled is not None and is_cancelled():
                    _stop(process)
                    raise VideoConcatCancelled()
                if time.monotonic() >= deadline:
                    _stop(process)
                    raise subprocess.TimeoutExpired(list(command), timeout)
        except BaseException:
            if process.poll() is None:
                _stop(process)
            raise
        errors.seek(0)
        return FinishedProcess(returncode=process.returncode, stderr=errors.read()[-400:])


@dataclass(frozen=True)
class ClipInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool


def probe_clip(
    path: PathLike, *, probe: Callable[[Path], Optional[VideoProbe]] = probe_video,
) -> Optional[ClipInfo]:
    found = probe(Path(path))
    if found is None or found.width <= 0 or found.height <= 0 or found.duration <= 0:
        return None
    return ClipInfo(
        width=found.width,
        height=found.height,
        fps=found.fps if found.fps > 0 else DEFAULT_FPS,
        duration=found.duration,
        has_audio=found.has_audio,
    )


def kept_seconds(info: ClipInfo, trim: ClipTrim) -> float:
    return info.duration - trim.start - trim.end


def _check_layout(infos: Sequence[ClipInfo], trims: Sequence[ClipTrim], crossfades: Sequence[float]) -> None:
    if len(trims) != len(infos):
        raise VideoConcatError(f"expected {len(infos)} trim(s), got {len(trims)}")
    if len(crossfades) != len(infos) - 1:
        raise VideoConcatError(f"expected {len(infos) - 1} crossfade(s), got {len(crossfades)}")
    lengths = [kept_seconds(info, trim) for info, trim in zip(infos, trims)]
    if any(trim.start < 0 or trim.end < 0 for trim in trims) or any(length <= 0 for length in lengths):
        raise VideoConcatError("a clip has nothing left after trimming")
    joined = lengths[0]
    for fade, length in zip(crossfades, lengths[1:]):
        if fade < 0 or fade >= min(joined, length):
            raise VideoConcatError(f"a {fade:.3f}s crossfade is longer than the clips it joins")
        joined += length - fade


def _trim_filter(name: str, info: ClipInfo, trim: ClipTrim) -> str:
    if trim.start <= 0 and trim.end <= 0:
        return ""
    parts = []
    if trim.start > 0:
        parts.append(f"start={max(0.0, trim.start - EDGE_NUDGE_SECONDS):.6f}")
    if trim.end > 0:
        parts.append(f"end={info.duration - trim.end - EDGE_NUDGE_SECONDS:.6f}")
    reset = "asetpts" if name == "atrim" else "setpts"
    return f"{name}={':'.join(parts)},{reset}=PTS-STARTPTS,"


def concat_command(
    ffmpeg: str,
    clips: Sequence[PathLike],
    infos: Sequence[ClipInfo],
    out_path: PathLike,
    *,
    trims: Optional[Sequence[ClipTrim]] = None,
    crossfades: Optional[Sequence[float]] = None,
    audio: bool = True,
    canvas: int = 0,
) -> List[str]:
    trims = list(trims) if trims is not None else [ClipTrim()] * len(clips)
    crossfades = list(crossfades) if crossfades is not None else [0.0] * (len(clips) - 1)
    _check_layout(infos, trims, crossfades)
    first = infos[canvas]
    width, height = first.width - first.width % 2, first.height - first.height % 2
    with_audio = audio and any(info.has_audio for info in infos)
    blended = any(fade > 0 for fade in crossfades)
    command = [ffmpeg, "-y", "-loglevel", "error"]
    for clip in clips:
        command += ["-i", str(clip)]
    silence_inputs = {}
    for index, info in enumerate(infos):
        if with_audio and not info.has_audio:
            silence_inputs[index] = len(clips) + len(silence_inputs)
            command += [
                "-f", "lavfi", "-t", f"{info.duration:.3f}",
                "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
            ]
    filters: List[str] = []
    joined = ""
    timebase = ",settb=AVTB" if blended else ""
    for index, (info, trim) in enumerate(zip(infos, trims)):
        filters.append(
            f"[{index}:v]{_trim_filter('trim', info, trim)}"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={first.fps:g},format=yuv420p{timebase}[v{index}]"
        )
        joined += f"[v{index}]"
        if with_audio:
            source = f"{silence_inputs[index]}:a" if index in silence_inputs else f"{index}:a"
            filters.append(
                f"[{source}]{_trim_filter('atrim', info, trim)}"
                f"aresample={SAMPLE_RATE},aformat=channel_layouts=stereo[a{index}]"
            )
            joined += f"[a{index}]"
    if blended:
        filters += _blend_chain(infos, trims, crossfades, with_audio)
    else:
        filters.append(
            f"{joined}concat=n={len(clips)}:v=1:a={1 if with_audio else 0}[v]" + ("[a]" if with_audio else "")
        )
    command += ["-filter_complex", ";".join(filters), "-map", "[v]"]
    if with_audio:
        command += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    command += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_path),
    ]
    return command


def _blend_chain(
    infos: Sequence[ClipInfo], trims: Sequence[ClipTrim], crossfades: Sequence[float], with_audio: bool,
) -> List[str]:
    filters: List[str] = []
    video, sound = "v0", "a0"
    elapsed = kept_seconds(infos[0], trims[0])
    last = len(infos) - 1
    for index in range(1, len(infos)):
        fade = crossfades[index - 1]
        video_out = "v" if index == last else f"vj{index}"
        sound_out = "a" if index == last else f"aj{index}"
        if fade > 0:
            filters.append(
                f"[{video}][v{index}]xfade=transition=fade:duration={fade:.6f}:offset={elapsed - fade:.6f}[{video_out}]"
            )
            if with_audio:
                filters.append(f"[{sound}][a{index}]acrossfade=d={fade:.6f}[{sound_out}]")
        else:
            filters.append(f"[{video}][v{index}]concat=n=2:v=1:a=0[{video_out}]")
            if with_audio:
                filters.append(f"[{sound}][a{index}]concat=n=2:v=0:a=1[{sound_out}]")
        elapsed += kept_seconds(infos[index], trims[index]) - fade
        video, sound = video_out, sound_out
    return filters


def _ffmpeg_concat(
    clips: Sequence[PathLike],
    out_path: Path,
    *,
    trims: Optional[Sequence[ClipTrim]],
    crossfades: Optional[Sequence[float]],
    audio: bool,
    canvas: int,
    run: Callable,
    find: Callable[[], Optional[str]],
    probe: Callable,
    is_cancelled: Optional[Callable[[], bool]],
) -> bool:
    ffmpeg = find()
    if ffmpeg is None:
        return False
    infos = [probe(clip) for clip in clips]
    if any(info is None for info in infos):
        return False
    command = concat_command(
        ffmpeg, clips, infos, out_path, trims=trims, crossfades=crossfades, audio=audio, canvas=canvas,
    )
    try:
        result = run(command, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS, is_cancelled=is_cancelled)
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning(f"[VIDEO_CONCAT] ffmpeg could not join the clips: {type(error).__name__}")
        return False
    if result.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
        stderr = (result.stderr or b"")[-400:]
        logger.warning(f"[VIDEO_CONCAT] ffmpeg could not join the clips: {stderr!r}")
        return False
    return True


def _opencv_concat(
    clips: Sequence[PathLike], out_path: Path, trims: Sequence[ClipTrim], crossfades: Sequence[float],
) -> None:
    import cv2

    writer = None
    size = None
    fps = DEFAULT_FPS
    try:
        for index, clip in enumerate(clips):
            capture = cv2.VideoCapture(str(clip))
            if not capture.isOpened():
                raise VideoConcatError(f"could not open clip {clip}")
            try:
                clip_fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
                if writer is None:
                    fps = clip_fps
                overlap = crossfades[index - 1] if index > 0 else 0.0
                skip = round((trims[index].start + overlap) * clip_fps)
                held: deque = deque()
                hold = round(trims[index].end * clip_fps)
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if skip > 0:
                        skip -= 1
                        continue
                    held.append(frame)
                    if len(held) <= hold:
                        continue
                    frame = held.popleft()
                    if writer is None:
                        size = (frame.shape[1], frame.shape[0])
                        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                        if not writer.isOpened():
                            raise VideoConcatError("could not write the joined video")
                    if (frame.shape[1], frame.shape[0]) != size:
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    writer.write(frame)
            finally:
                capture.release()
    finally:
        if writer is not None:
            writer.release()
    if writer is None or not out_path.is_file() or out_path.stat().st_size == 0:
        raise VideoConcatError("the clips had no frames to join")


def concat_videos(
    clips: Sequence[PathLike],
    out_path: PathLike,
    *,
    trims: Optional[Sequence[ClipTrim]] = None,
    crossfades: Optional[Sequence[float]] = None,
    audio: bool = True,
    canvas: int = 0,
    run: Callable = run_cancellable,
    find: Callable[[], Optional[str]] = find_ffmpeg,
    probe: Optional[Callable[[PathLike], Optional[ClipInfo]]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    notify: Optional[Callable[[str], None]] = None,
) -> Path:
    if not clips:
        raise VideoConcatError("there are no clips to join")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _ffmpeg_concat(
        clips, out_path, trims=trims, crossfades=crossfades, audio=audio, canvas=canvas,
        run=run, find=find, probe=probe or probe_clip, is_cancelled=is_cancelled,
    ):
        logger.info(f"[VIDEO_CONCAT] joined {len(clips)} clip(s) with ffmpeg")
        return out_path
    logger.warning("[VIDEO_CONCAT] ffmpeg was not usable; joining the clips without their sound")
    if notify is not None and audio:
        notify(NO_SOUND_NOTICE)
    _opencv_concat(
        clips, out_path,
        list(trims) if trims is not None else [ClipTrim()] * len(clips),
        list(crossfades) if crossfades is not None else [0.0] * (len(clips) - 1),
    )
    return out_path


def trim_command(
    ffmpeg: str, clip: PathLike, out_path: PathLike, *, keep: str, frames: int, fps: float = 0.0, audio: bool = True,
) -> List[str]:
    if keep not in ("head", "tail"):
        raise VideoConcatError(f"keep must be 'head' or 'tail', got {keep!r}")
    if frames < 1:
        raise VideoConcatError("a trimmed clip needs at least one frame")
    resample = f"fps={fps:g}," if fps > 0 else ""
    even = "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"
    command = [ffmpeg, "-y", "-loglevel", "error"]
    if keep == "tail":
        window = (frames + 2) / (fps if fps > 0 else DEFAULT_FPS) + 1.0
        command += ["-sseof", f"-{window:.3f}"]
        video = f"{resample}reverse,trim=end_frame={frames},reverse,setpts=PTS-STARTPTS,{even}"
    else:
        video = f"{resample}trim=end_frame={frames},setpts=PTS-STARTPTS,{even}"
    command += ["-i", str(clip), "-map", "0:v:0", "-vf", video]
    if audio and fps > 0:
        seconds = frames / fps
        sound = (
            f"areverse,atrim=end={seconds:.6f},areverse,asetpts=PTS-STARTPTS" if keep == "tail"
            else f"atrim=end={seconds:.6f},asetpts=PTS-STARTPTS"
        )
        command += ["-map", "0:a:0?", "-af", sound, "-c:a", "aac", "-b:a", "192k"]
    else:
        command += ["-an"]
    command += [
        "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path),
    ]
    return command


def trim_video(
    clip: PathLike,
    out_path: PathLike,
    *,
    keep: str,
    frames: int,
    fps: float = 0.0,
    audio: bool = True,
    run: Callable = run_cancellable,
    find: Callable[[], Optional[str]] = find_ffmpeg,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    ffmpeg = find()
    if ffmpeg is None:
        raise VideoConcatError("ffmpeg is needed to trim a video")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = trim_command(ffmpeg, clip, out_path, keep=keep, frames=frames, fps=fps, audio=audio)
    try:
        result = run(command, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS, is_cancelled=is_cancelled)
    except (OSError, subprocess.SubprocessError) as error:
        raise VideoConcatError(f"ffmpeg could not trim the video: {type(error).__name__}") from error
    if result.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
        stderr = (result.stderr or b"")[-400:]
        raise VideoConcatError(f"ffmpeg could not trim the video: {stderr!r}")
    return out_path

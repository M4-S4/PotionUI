from __future__ import annotations

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.pipelines.pipes._shared.media.video_read import read_video_frames
from src.pipelines.pipes.generator.video_minimax_h3.conditioning import (
    normalize_reference_video,
    reference_decode_seconds,
)


def _write_video(path, fps: int, frame_count: int, size=(32, 24)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    assert writer.isOpened()
    rng = np.random.default_rng(fps)
    for index in range(frame_count):
        frame = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        frame[:, :, 0] = index * 3 % 256
        writer.write(frame)
    writer.release()
    return path


def test_max_seconds_stops_decoding_after_the_requested_head(tmp_path):
    path = _write_video(tmp_path / "clip.mp4", 30, 60)
    full, fps = read_video_frames(path)
    head, head_fps = read_video_frames(path, max_seconds=0.5)
    assert len(full) == 60
    assert len(head) == math.ceil(0.5 * fps) == 15
    assert head_fps == fps
    for a, b in zip(head, full):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_max_seconds_beyond_the_clip_returns_every_frame(tmp_path):
    path = _write_video(tmp_path / "clip.mp4", 30, 20)
    frames, _ = read_video_frames(path, max_seconds=10.0)
    assert len(frames) == 20


@pytest.mark.parametrize("fps", [30, 25, 24, 60])
@pytest.mark.parametrize("num_frames", [5, 22])
def test_bounded_decode_normalizes_to_the_same_frames_as_a_full_decode(tmp_path, fps, num_frames):
    path = _write_video(tmp_path / f"clip_{fps}.mp4", fps, fps * 2)

    full_frames, full_fps = read_video_frames(path)
    expected = normalize_reference_video(full_frames, fps=full_fps, num_frames=num_frames)

    bounded_frames, bounded_fps = read_video_frames(path, max_seconds=reference_decode_seconds(num_frames))
    actual = normalize_reference_video(bounded_frames, fps=bounded_fps, num_frames=num_frames)

    assert len(bounded_frames) < len(full_frames)
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)

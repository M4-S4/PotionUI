import numpy as np
import pytest

from src.pipelines.pipes._shared.video.frames import probe_video_stream


def test_the_probe_reads_size_rate_and_frame_count(tmp_path):
    cv2 = pytest.importorskip("cv2", reason="the probe is cv2-backed", exc_type=ImportError)
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (40, 30))
    if not writer.isOpened():
        pytest.skip("no mp4v encoder available to synthesize a clip")
    for i in range(7):
        writer.write(np.full((30, 40, 3), i * 20, dtype=np.uint8))
    writer.release()

    assert probe_video_stream(str(path)) == (40, 30, 12.0, 7)


def test_a_missing_clip_is_an_error(tmp_path):
    pytest.importorskip("cv2", reason="the probe is cv2-backed", exc_type=ImportError)
    with pytest.raises(ValueError):
        probe_video_stream(str(tmp_path / "missing.mp4"))

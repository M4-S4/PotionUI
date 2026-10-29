"""Read a whole video into an in-memory frame sequence + its native fps.

Complements ``frame_extract.extract_frame`` (ONE frame at an exact index): it
returns *every* frame together with the source frame rate, which is what a
frame-for-frame video transform (e.g. SeedVR2 upscale) needs: every source
frame is upscaled and the clip is re-encoded at the original fps.
"""

from __future__ import annotations

import logging
from pathlib import Path
import math
from typing import List, Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_FPS = 24.0


def read_video_frames(
    video_path: Union[str, Path], *, max_seconds: Optional[float] = None,
) -> Tuple[List[Image.Image], float]:
    """Read all frames of ``video_path`` as RGB PIL Images plus the native fps.

    Returns ``(frames, fps)``. ``fps`` falls back to :data:`DEFAULT_FPS` when the
    container reports a non-positive rate. ``max_seconds`` stops decoding after
    ``ceil(max_seconds * fps)`` source frames, so a caller that only needs the head
    of a long clip never decodes the rest. Raises ``ValueError`` if the file can't
    be opened or yields no frames.
    """
    import cv2

    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else DEFAULT_FPS

        limit = max(1, math.ceil(max_seconds * fps)) if max_seconds is not None else None
        frames: List[Image.Image] = []
        while limit is None or len(frames) < limit:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"no frames decoded from video: {path}")

    logger.info("[VIDEO_READ] %s: %d frame(s) @ %.3f fps", Path(path).name, len(frames), fps)
    return frames, fps

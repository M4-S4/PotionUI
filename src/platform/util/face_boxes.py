"""Face boxes found by OpenCV's YuNet detector, at the size of each image."""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from PIL import Image

Box = Dict[str, float]

NMS = 0.3
TOP_K = 5000
SCORE = 14


def _rgb(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as opened:
        return opened.convert("RGB")


def boxes(faces: Any, min_score: float) -> List[Box]:
    """The rows YuNet answers (x, y, width, height, five landmarks, score) as boxes, best first, the weak and the empty ones left out."""
    found = []
    for face in [] if faces is None else faces:
        x, y, width, height, score = (float(value) for value in (face[0], face[1], face[2], face[3], face[SCORE]))
        if score >= min_score and width > 0 and height > 0:
            found.append({"x": x, "y": y, "width": width, "height": height, "score": round(score, 4)})
    return sorted(found, key=lambda box: box["score"], reverse=True)


def detect(model: Path, images: Sequence[Union[str, Path, Image.Image]], min_score: float = 0.5) -> List[List[Box]]:
    """The faces YuNet finds in each image, in pixels of the image, best first, read at the image's own size so that the small face of a wide shot is found; the model file is given, never downloaded."""
    import cv2
    import numpy as np

    detector = None
    found = []
    for image in images:
        picture = np.ascontiguousarray(cv2.cvtColor(np.asarray(_rgb(image)), cv2.COLOR_RGB2BGR))
        size = (picture.shape[1], picture.shape[0])
        if detector is None:
            detector = cv2.FaceDetectorYN.create(str(model), "", size, min_score, NMS, TOP_K)
        detector.setInputSize(size)
        _, faces = detector.detect(picture)
        found.append(boxes(faces, min_score))
    return found

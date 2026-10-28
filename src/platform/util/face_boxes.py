"""Face boxes found by MediaPipe's BlazeFace in whole images and in overlapping tiles of them."""

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

from PIL import Image

Box = Dict[str, float]

TILE_SHARE = 2 / 3
OVERLAP = 0.3


def tiles(width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """The whole image, then square tiles of two thirds of its short side that overlap by half, left to right and top to bottom, as (left, top, right, bottom): BlazeFace looks at a 128 px square, so a face of a wide shot is found only in a tile."""
    side = max(1, round(min(width, height) * TILE_SHARE))
    found = [(0, 0, width, height)]

    def starts(length: int) -> List[int]:
        if length <= side:
            return [0]
        count = math.ceil((length - side) / (side / 2)) + 1
        return [round(index * (length - side) / (count - 1)) for index in range(count)]

    for top in starts(height):
        for left in starts(width):
            found.append((left, top, min(width, left + side), min(height, top + side)))
    return found


def overlap(a: Box, b: Box) -> float:
    """The intersection over union of two boxes."""
    left, top = max(a["x"], b["x"]), max(a["y"], b["y"])
    right, bottom = min(a["x"] + a["width"], b["x"] + b["width"]), min(a["y"] + a["height"], b["y"] + b["height"])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def merged(boxes: Iterable[Box]) -> List[Box]:
    """The boxes best first, each dropped when it overlaps a better one by more than OVERLAP."""
    kept: List[Box] = []
    for box in sorted(boxes, key=lambda item: item["score"], reverse=True):
        if all(overlap(box, other) <= OVERLAP for other in kept):
            kept.append(box)
    return kept


def _rgb(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as opened:
        return opened.convert("RGB")


def _boxes(detector: Any, picture: Image.Image, left: int, top: int, min_score: float) -> List[Box]:
    import mediapipe as mp
    import numpy as np

    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(picture))))
    found = []
    for detection in result.detections:
        score = float(detection.categories[0].score) if detection.categories else 0.0
        box = detection.bounding_box
        if score >= min_score and box.width > 0 and box.height > 0:
            found.append({"x": float(left + box.origin_x), "y": float(top + box.origin_y), "width": float(box.width), "height": float(box.height), "score": round(score, 4)})
    return found


def detect(model: Path, images: Sequence[Union[str, Path, Image.Image]], min_score: float = 0.5) -> List[List[Box]]:
    """The faces BlazeFace finds in each image, in pixels of the image, best first, searched in the whole image and in its tiles; the model file is given, never downloaded."""
    from mediapipe.tasks import python as tasks
    from mediapipe.tasks.python import vision

    options = vision.FaceDetectorOptions(base_options=tasks.BaseOptions(model_asset_path=str(model)), min_detection_confidence=min_score)
    with vision.FaceDetector.create_from_options(options) as detector:
        found = []
        for image in images:
            picture = _rgb(image)
            boxes = []
            for left, top, right, bottom in tiles(*picture.size):
                boxes.extend(_boxes(detector, picture.crop((left, top, right, bottom)), left, top, min_score))
            found.append(merged(boxes))
        return found

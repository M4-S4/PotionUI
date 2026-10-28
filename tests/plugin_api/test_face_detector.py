import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from src.platform.util import face_boxes
from src.plugin_api import media


def _world(*folders):
    entries = [SimpleNamespace(path=Path(folder)) for folder in folders]
    return SimpleNamespace(model_roots=SimpleNamespace(type_dirs=lambda model_type: entries if model_type == "mediapipe" else []))


def _box(x, y, side, score):
    return {"x": float(x), "y": float(y), "width": float(side), "height": float(side), "score": score}


def test_the_face_detector_says_which_file_it_reads_and_whether_it_is_here(tmp_path):
    empty, home = tmp_path / "a", tmp_path / "b"
    empty.mkdir()
    home.mkdir()
    with patch("src.plugin_api.media.get_container", return_value=_world(empty, home)):
        before = media.face_detector()
        (home / media.FACE_MODEL).write_bytes(b"model")
        after = media.face_detector()

    assert before == {"model": "blaze_face_short_range.tflite", "available": False}
    assert after == {"model": "blaze_face_short_range.tflite", "available": True}


def test_without_its_model_file_detecting_faces_says_so_and_downloads_nothing(tmp_path):
    with patch("src.plugin_api.media.get_container", return_value=_world(tmp_path)), patch.object(face_boxes, "detect") as detect:
        with pytest.raises(FileNotFoundError):
            media.detect_faces(["a.png"])

    detect.assert_not_called()


def test_detecting_faces_reads_the_model_file_of_the_first_mediapipe_folder_that_has_it(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (second / media.FACE_MODEL).write_bytes(b"model")
    with patch("src.plugin_api.media.get_container", return_value=_world(first, second)), patch.object(face_boxes, "detect", return_value=[[]]) as detect:
        found = media.detect_faces(("a.png",), min_score=0.6)

    assert found == [[]]
    detect.assert_called_once_with(second / media.FACE_MODEL, ["a.png"], 0.6)


def test_an_image_is_searched_whole_then_in_square_tiles_of_two_thirds_of_its_short_side_that_overlap_by_half():
    found = face_boxes.tiles(832, 480)

    assert found[0] == (0, 0, 832, 480)
    assert {right - left for left, _, right, _ in found[1:]} == {320} and {bottom - top for _, top, _, bottom in found[1:]} == {320}
    assert [left for left, top, _, _ in found[1:] if top == 0] == [0, 128, 256, 384, 512]
    assert sorted({top for _, top, _, _ in found[1:]}) == [0, 160]
    assert face_boxes.tiles(100, 100) == [(0, 0, 100, 100), (0, 0, 67, 67), (33, 0, 100, 67), (0, 33, 67, 100), (33, 33, 100, 100)]


def test_a_face_found_twice_is_kept_once_with_its_best_score_and_two_faces_stay_two():
    boxes = [_box(10, 10, 50, 0.7), _box(12, 11, 48, 0.9), _box(200, 40, 60, 0.8)]

    kept = face_boxes.merged(boxes)

    assert [box["score"] for box in kept] == [0.9, 0.8]
    assert face_boxes.overlap(_box(0, 0, 10, 1), _box(5, 0, 10, 1)) == pytest.approx(50 / 150)
    assert face_boxes.overlap(_box(0, 0, 10, 1), _box(20, 0, 10, 1)) == 0.0


class Detector:
    def __init__(self, faces):
        self.faces = faces
        self.sizes = []

    def detect(self, image):
        height, width = image.numpy_view().shape[:2]
        self.sizes.append((width, height))
        found = self.faces.get((width, height), [])
        return SimpleNamespace(detections=[SimpleNamespace(categories=[SimpleNamespace(score=score)], bounding_box=SimpleNamespace(origin_x=x, origin_y=y, width=side, height=side)) for x, y, side, score in found])


def test_the_boxes_of_a_tile_are_moved_to_the_image_and_a_weak_one_is_dropped():
    pytest.importorskip("mediapipe")
    detector = Detector({(320, 320): [(40, 50, 30, 0.9), (5, 5, 10, 0.2)]})

    boxes = face_boxes._boxes(detector, Image.new("RGB", (320, 320)), 171, 160, 0.5)

    assert boxes == [{"x": 211.0, "y": 210.0, "width": 30.0, "height": 30.0, "score": 0.9}]


@pytest.mark.requires_models
def test_the_real_detector_finds_no_face_in_a_plain_image_and_reads_its_model_from_the_model_roots():
    folder = Path(os.environ.get("POTIONUI_MEDIAPIPE_DIR", "models/mediapipe"))
    if not (folder / media.FACE_MODEL).is_file():
        pytest.skip("blaze_face_short_range.tflite is not on this machine")

    with patch("src.plugin_api.media.get_container", return_value=_world(folder)):
        found = media.detect_faces([Image.new("RGB", (832, 480), (120, 130, 140))])

    assert found == [[]]


def test_the_names_are_listed_by_the_media_module():
    assert {"detect_faces", "face_detector"} <= set(media.__all__)

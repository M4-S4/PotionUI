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
    return SimpleNamespace(model_roots=SimpleNamespace(type_dirs=lambda model_type: entries if model_type == "detection_bbox" else []))


def _row(x, y, width, height, score, points=None):
    return [x, y, width, height, *(points or [0.0] * 10), score]


NO_POINTS = [[0.0, 0.0]] * 5


def test_the_face_detector_says_which_file_it_reads_and_whether_it_is_here(tmp_path):
    empty, home = tmp_path / "a", tmp_path / "b"
    empty.mkdir()
    home.mkdir()
    with patch("src.plugin_api.media.get_container", return_value=_world(empty, home)):
        before = media.face_detector()
        (home / media.FACE_MODEL).write_bytes(b"model")
        after = media.face_detector()

    assert before == {"model": "face_detection_yunet_2023mar.onnx", "available": False}
    assert after == {"model": "face_detection_yunet_2023mar.onnx", "available": True}


def test_without_its_model_file_detecting_faces_says_so_and_downloads_nothing(tmp_path):
    with patch("src.plugin_api.media.get_container", return_value=_world(tmp_path)), patch.object(face_boxes, "detect") as detect:
        with pytest.raises(FileNotFoundError):
            media.detect_faces(["a.png"])

    detect.assert_not_called()


def test_detecting_faces_reads_the_model_file_of_the_first_detection_bbox_folder_that_has_it(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (second / media.FACE_MODEL).write_bytes(b"model")
    with patch("src.plugin_api.media.get_container", return_value=_world(first, second)), patch.object(face_boxes, "detect", return_value=[[]]) as detect:
        found = media.detect_faces(("a.png",), min_score=0.6)

    assert found == [[]]
    detect.assert_called_once_with(second / media.FACE_MODEL, ["a.png"], 0.6)


def test_the_rows_yunet_answers_become_boxes_best_first_without_the_weak_and_empty_ones():
    rows = [_row(10, 20, 30, 40, 0.7), _row(100, 20, 50, 50, 0.95), _row(5, 5, 8, 8, 0.3), _row(1, 1, 0, 9, 0.9)]

    found = face_boxes.boxes(rows, 0.5)

    assert found == [
        {"x": 100.0, "y": 20.0, "width": 50.0, "height": 50.0, "score": 0.95, "landmarks": NO_POINTS},
        {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0, "score": 0.7, "landmarks": NO_POINTS},
    ]
    assert face_boxes.boxes(None, 0.5) == []


def test_a_box_keeps_the_five_landmarks_yunet_answers_in_their_order():
    points = [11.0, 12.0, 21.0, 12.5, 16.0, 18.0, 12.0, 24.0, 20.0, 24.5]

    (found,) = face_boxes.boxes([_row(5, 5, 20, 24, 0.9, points)], 0.5)

    assert found["landmarks"] == [[11.0, 12.0], [21.0, 12.5], [16.0, 18.0], [12.0, 24.0], [20.0, 24.5]]


class Detector:
    def __init__(self):
        self.sizes = []

    def setInputSize(self, size):
        self.sizes.append(size)

    def detect(self, picture):
        return 1, [_row(4, 6, 20, 20, 0.88)] if picture.shape[1] > 100 else None


def test_each_image_is_read_at_its_own_size_by_one_detector():
    cv2 = pytest.importorskip("cv2")
    detector = Detector()

    with patch.object(cv2.FaceDetectorYN, "create", return_value=detector) as create:
        found = face_boxes.detect(Path("m.onnx"), [Image.new("RGB", (832, 480)), Image.new("RGB", (64, 64))], 0.5)

    assert found == [[{"x": 4.0, "y": 6.0, "width": 20.0, "height": 20.0, "score": 0.88, "landmarks": NO_POINTS}], []]
    assert detector.sizes == [(832, 480), (64, 64)]
    create.assert_called_once_with("m.onnx", "", (832, 480), 0.5, face_boxes.NMS, face_boxes.TOP_K)


@pytest.mark.requires_models
def test_the_real_detector_finds_no_face_in_a_plain_image():
    folder = Path(os.environ.get("POTIONUI_FACE_MODEL_DIR", "models/detection_bbox"))
    if not (folder / media.FACE_MODEL).is_file():
        pytest.skip("face_detection_yunet_2023mar.onnx is not on this machine")

    with patch("src.plugin_api.media.get_container", return_value=_world(folder)):
        found = media.detect_faces([Image.new("RGB", (832, 480), (120, 130, 140))])

    assert found == [[]]


def test_the_names_are_listed_by_the_media_module():
    assert {"detect_faces", "face_detector", "face_identifier", "identify_faces"} <= set(media.__all__)

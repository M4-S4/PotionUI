import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from src.platform.util import face_identity
from src.plugin_api import media


def _world(home):
    return SimpleNamespace(model_roots=SimpleNamespace(asset_dir=lambda subdir: Path(home) / subdir))


def _points(scale=1.0, dx=40.0, dy=40.0):
    return [[x * scale + dx, y * scale + dy] for x, y in face_identity.TEMPLATE]


def test_the_face_identifier_is_looked_for_in_the_home_model_root_unless_a_file_is_named(tmp_path, monkeypatch):
    monkeypatch.delenv(media.IDENTITY_ENV, raising=False)
    target = tmp_path / "face_recognition" / "auraface" / "glintr100.onnx"
    with patch("src.plugin_api.media.get_container", return_value=_world(tmp_path)):
        before = media.face_identifier()
        target.parent.mkdir(parents=True)
        target.write_bytes(b"model")
        after = media.face_identifier()

    named = tmp_path / "etudes" / "glintr100.onnx"
    monkeypatch.setenv(media.IDENTITY_ENV, str(named))
    with patch("src.plugin_api.media.get_container", return_value=_world(tmp_path)):
        missing = media.face_identifier()
        named.parent.mkdir()
        named.write_bytes(b"model")
        found = media.face_identifier()

    assert before == {"model": "glintr100.onnx", "path": str(target), "available": False}
    assert after == {"model": "glintr100.onnx", "path": str(target), "available": True}
    assert missing == {"model": "glintr100.onnx", "path": str(named), "available": False}
    assert found == {"model": "glintr100.onnx", "path": str(named), "available": True}


def test_without_its_model_file_identifying_faces_says_so_and_downloads_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(media.IDENTITY_ENV, raising=False)
    with patch("src.plugin_api.media.get_container", return_value=_world(tmp_path)), patch.object(face_identity, "embed") as embed:
        with pytest.raises(FileNotFoundError):
            media.identify_faces([("a.png", _points())])

    embed.assert_not_called()


def test_identifying_faces_reads_the_named_file_and_hands_each_image_with_its_landmarks(tmp_path, monkeypatch):
    model = tmp_path / "glintr100.onnx"
    model.write_bytes(b"model")
    monkeypatch.setenv(media.IDENTITY_ENV, str(model))
    with patch.object(face_identity, "embed", return_value=[[1.0, 0.0]]) as embed:
        found = media.identify_faces([("a.png", _points())])

    assert found == [[1.0, 0.0]]
    embed.assert_called_once_with(model, [("a.png", _points())])


def test_a_face_is_aligned_so_that_its_five_landmarks_sit_on_the_arcface_template():
    pytest.importorskip("cv2")
    image = Image.new("RGB", (400, 300), (0, 0, 0))
    points = _points(scale=2.0, dx=60.0, dy=40.0)
    eye_x, eye_y = points[0]
    ImageDraw.Draw(image).rectangle((eye_x - 3, eye_y - 3, eye_x + 3, eye_y + 3), fill=(255, 255, 255))

    aligned = face_identity.align(image, points)

    assert aligned.shape == (112, 112, 3)
    assert aligned[round(face_identity.TEMPLATE[0][1]), round(face_identity.TEMPLATE[0][0])].min() > 200
    assert aligned[round(face_identity.TEMPLATE[2][1]), round(face_identity.TEMPLATE[2][0])].max() < 50


class Session:
    def __init__(self):
        self.inputs = []

    def get_inputs(self):
        return [SimpleNamespace(name="data")]

    def run(self, outputs, feeds):
        import numpy as np

        self.inputs.append(feeds["data"])
        return [np.array([[3.0, 4.0]], dtype=np.float32)]


def test_each_face_is_read_alone_scaled_to_minus_one_one_and_its_vector_normalized(monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    session = Session()
    monkeypatch.setattr(face_identity, "_session", lambda model: session)
    white = Image.new("RGB", (200, 200), (255, 255, 255))

    vectors = face_identity.embed(Path("m.onnx"), [(white, _points()), (white, _points())])

    assert vectors == [pytest.approx([0.6, 0.8]), pytest.approx([0.6, 0.8])]
    assert [tuple(item.shape) for item in session.inputs] == [(1, 3, 112, 112), (1, 3, 112, 112)]
    assert np.allclose(session.inputs[0][0, :, 56, 56], 1.0)
    assert face_identity.embed(Path("m.onnx"), []) == []


@pytest.mark.requires_models
def test_the_real_model_gives_one_unit_vector_of_512_values_per_face():
    path = Path(os.environ.get(media.IDENTITY_ENV) or "models/face_recognition/auraface/glintr100.onnx")
    if not path.is_file():
        pytest.skip("glintr100.onnx is not on this machine")

    (vector,) = face_identity.embed(path, [(Image.new("RGB", (200, 200), (128, 100, 90)), _points())])

    assert len(vector) == 512
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-3

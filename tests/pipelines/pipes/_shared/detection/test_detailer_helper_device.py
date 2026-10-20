import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

pytest.importorskip(
    "cv2",
    reason="cv2 required by ultralytics (detailer helper dependency)",
    exc_type=ImportError,
)

from src.pipelines.pipes._shared.detection import detailer_helper
from src.pipelines.pipes._shared.detection.detailer_helper import DetailerHelper


@pytest.mark.parametrize("cuda_available, expected", [(True, "cuda"), (False, "cpu")])
def test_detection_without_a_device_follows_the_host(monkeypatch, cuda_available, expected):
    monkeypatch.setattr(detailer_helper.torch.cuda, "is_available", lambda: cuda_available)
    helper = DetailerHelper({"detections": {"face": {"confidence": 0.3}}})
    detector = MagicMock()
    detector.predict.return_value = []

    helper.detect_objects_scored(Image.new("RGB", (8, 8)), detector, "face")

    assert detector.predict.call_args.kwargs["device"] == expected


def test_detection_with_a_device_keeps_it(monkeypatch):
    monkeypatch.setattr(detailer_helper.torch.cuda, "is_available", lambda: False)
    helper = DetailerHelper({"detections": {"face": {"confidence": 0.3, "device": "cuda"}}})
    detector = MagicMock()
    detector.predict.return_value = []

    helper.detect_objects_scored(Image.new("RGB", (8, 8)), detector, "face")

    assert detector.predict.call_args.kwargs["device"] == "cuda"


@pytest.mark.parametrize("cuda_available, expected", [(True, "cuda"), (False, "cpu")])
def test_load_detector_without_a_device_follows_the_host(monkeypatch, cuda_available, expected):
    monkeypatch.setattr(detailer_helper.torch.cuda, "is_available", lambda: cuda_available)
    model = MagicMock()
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: model))

    DetailerHelper({}).load_detector("face.pt")

    model.to.assert_called_once_with(expected)

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.pipelines.contracts import IOType, PipeInput
from src.pipelines.outputs import (
    GalleryGenerationOutput,
    ProgressGenerationOutput,
    SplatGenerationOutput,
    TextGenerationOutput,
)
from src.pipelines.pipes.generator.image_to_splat.main import GeneratorImageToSplatPipe
from src.platform.filesystem.splat_formats import inspect_splat
from src.platform.runtime.native.errors import SamplingCancelled
from tests.platform.runtime.native.arch.image_to_splat.test_image_to_splat import FakeEstimator


class _Bundle:
    def __init__(self, estimator, device="cpu"):
        self.estimator = estimator
        self.device = device

    def components(self):
        return self.estimator


def _image(width=96, height=64):
    pixels = np.random.default_rng(1).integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, "RGB")


def _run(images, config=None, is_cancelled=None, estimator=None):
    outputs = []
    pipe = GeneratorImageToSplatPipe({**GeneratorImageToSplatPipe.get_default_config(), "device": "cpu", **(config or {})})
    bundle = _Bundle(estimator or FakeEstimator())
    result = pipe.process(PipeInput(input={"model": bundle, "image": images}), outputs.append, is_cancelled=is_cancelled)
    return result, outputs


@pytest.fixture
def cleanup():
    paths = []
    yield paths
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def test_the_pipe_declares_a_splat_output():
    [output] = GeneratorImageToSplatPipe.outputs()
    assert (output.name, output.io_type, output.is_array) == ("splat", IOType.SPLAT, True)
    assert {spec.name: spec.choices for spec in GeneratorImageToSplatPipe.configuration()}["detail"] == ["standard", "high"]


def test_one_image_becomes_one_contract_ply_in_a_gallery(cleanup):
    result, outputs = _run([_image()])
    [path] = result.output["splat"]
    cleanup.append(path)

    info = inspect_splat(path)
    assert info.format == "ply"
    assert info.view.width == 96 and info.view.height == 64
    assert info.view.depth == 4.0
    [gallery] = [output for output in outputs if isinstance(output, GalleryGenerationOutput)]
    [splat] = gallery.splats
    assert isinstance(splat, SplatGenerationOutput)
    assert splat.splat_path == Path(path)
    assert splat.temporary is False
    assert splat.splat_count == info.count
    assert splat.preview.size == (96, 64)
    assert gallery.images == [] and gallery.meshes == []


def test_progress_follows_the_stages_and_a_summary_is_reported(cleanup):
    result, outputs = _run([_image()])
    cleanup.extend(result.output["splat"])
    states = [output.state for output in outputs if isinstance(output, ProgressGenerationOutput)]
    assert states == ["Estimating depth with MoGe-2", "Building the 3D scene", "Writing the splat", "Built the 3D scene"]
    [summary] = [output for output in outputs if isinstance(output, TextGenerationOutput)]
    assert summary.title == "3D scene"
    assert "Gaussians" in summary.text and "96x64" in summary.text


def test_several_images_give_several_splats(cleanup):
    result, outputs = _run([_image(), _image(64, 64)])
    cleanup.extend(result.output["splat"])
    assert len(result.output["splat"]) == 2
    states = [output.state for output in outputs if isinstance(output, ProgressGenerationOutput)]
    assert states[0] == "Estimating depth with MoGe-2 (<<NUMBER:1>> of <<NUMBER:2>>)"
    assert states[-1] == "Built <<NUMBER:2>> 3D scenes"


def test_detail_reaches_the_inference_resolution(cleanup):
    estimator = FakeEstimator()
    result, _ = _run([_image(2048, 1024)], config={"detail": "standard"}, estimator=estimator)
    cleanup.extend(result.output["splat"])
    assert estimator.calls[0][0] == (1, 3, 512, 1024)


def test_cancellation_removes_the_splats_already_written():
    calls = {"count": 0}

    def cancelled():
        calls["count"] += 1
        return calls["count"] > 3

    written = []
    original = GeneratorImageToSplatPipe._write

    def _spy(lifted):
        path = original(lifted)
        written.append(path)
        return path

    GeneratorImageToSplatPipe._write = staticmethod(_spy)
    try:
        with pytest.raises(SamplingCancelled):
            _run([_image(), _image()], is_cancelled=cancelled)
    finally:
        GeneratorImageToSplatPipe._write = staticmethod(original)
    assert written and not [path for path in written if os.path.exists(path)]


@pytest.mark.parametrize("inputs,message", [
    ({"image": []}, "needs a source image"),
    ({"image": [Image.new("RGB", (8, 8))]}, "needs the model bundle"),
])
def test_missing_inputs_are_explained(inputs, message):
    pipe = GeneratorImageToSplatPipe(GeneratorImageToSplatPipe.get_default_config())
    with pytest.raises(ValueError, match=message):
        pipe.process(PipeInput(input=inputs), lambda output: None)


def test_an_unknown_detail_is_refused():
    with pytest.raises(ValueError, match="unknown detail"):
        _run([_image()], config={"detail": "ultra"})


class _FailingSecondImage(FakeEstimator):
    def estimate_geometry(self, pixels, resolution_level, mask_threshold):
        if self.calls:
            self.calls.append("second")
            raise RuntimeError("CUDA out of memory")
        return super().estimate_geometry(pixels, resolution_level, mask_threshold)


def test_an_error_in_the_batch_removes_the_splats_already_written(monkeypatch):
    written = []
    original = GeneratorImageToSplatPipe._write

    def _spy(lifted):
        path = original(lifted)
        written.append(path)
        return path

    monkeypatch.setattr(GeneratorImageToSplatPipe, "_write", staticmethod(_spy))
    with pytest.raises(RuntimeError, match="out of memory"):
        _run([_image(), _image()], estimator=_FailingSecondImage())
    assert len(written) == 1
    assert not os.path.exists(written[0])


def test_a_cancellation_between_two_images_skips_the_second(monkeypatch):
    estimator = FakeEstimator()
    written = []
    original = GeneratorImageToSplatPipe._write

    def _spy(lifted):
        path = original(lifted)
        written.append(path)
        return path

    monkeypatch.setattr(GeneratorImageToSplatPipe, "_write", staticmethod(_spy))
    with pytest.raises(SamplingCancelled):
        _run([_image(), _image()], estimator=estimator, is_cancelled=lambda: bool(written))
    assert len(estimator.calls) == 1
    assert written and not os.path.exists(written[0])


def test_a_cancellation_after_moge_never_lifts_or_writes(monkeypatch):
    estimator = FakeEstimator()
    written = []
    monkeypatch.setattr(GeneratorImageToSplatPipe, "_write", staticmethod(lambda lifted: written.append(1)))
    with pytest.raises(SamplingCancelled):
        _run([_image()], estimator=estimator, is_cancelled=lambda: bool(estimator.calls))
    assert len(estimator.calls) == 1
    assert written == []

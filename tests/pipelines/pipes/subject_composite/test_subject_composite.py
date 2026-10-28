import numpy as np
from PIL import Image

from src.pipelines.contracts import PipeInput
from src.pipelines.pipes._shared.inpainting.mask_utils import dilate_box
from src.pipelines.pipes.subject_composite.main import SubjectCompositePipe


def _original(size=(200, 160)):
    rng = np.random.default_rng(2)
    return rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)


def _cutout(rgb, box=(50, 110, 60, 140), soft=0):
    alpha = np.zeros(rgb.shape[:2], dtype=np.uint8)
    top, bottom, left, right = box
    alpha[top:bottom, left:right] = 255
    if soft:
        alpha[10:20, 10:30] = soft
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA"), alpha > 200


def _run(config, generated, cutout):
    pipe = SubjectCompositePipe(config)
    return pipe.process(PipeInput(input={"image": generated, "cutout": [cutout]}), None).output["image"]


def test_subject_pixels_are_identical_to_the_original():
    rgb = _original()
    cutout, subject = _cutout(rgb)
    generated = Image.new("RGB", (200, 160), (10, 200, 30))
    out = np.array(_run({}, [generated], cutout)[0])
    np.testing.assert_array_equal(out[subject], rgb[subject])


def test_background_outside_the_feather_comes_from_the_edit():
    rgb = _original()
    cutout, subject = _cutout(rgb)
    generated = Image.new("RGB", (200, 160), (10, 200, 30))
    out = np.array(_run({"feather": 0.05}, [generated], cutout)[0])
    far = ~dilate_box(subject, 12)
    assert (out[far] == (10, 200, 30)).all()


def test_feather_blends_a_ring_outside_the_subject():
    rgb = _original()
    cutout, subject = _cutout(rgb)
    generated = Image.new("RGB", (200, 160), (0, 0, 0))
    wide = np.array(_run({"feather": 0.1}, [generated], cutout)[0]).astype(int)
    none = np.array(_run({"feather": 0.0}, [generated], cutout)[0]).astype(int)
    ring = dilate_box(subject, 7) & ~dilate_box(subject, 3)
    assert (wide[ring].sum(axis=-1) > 0).mean() > 0.5
    assert (none[ring] == 0).all()


def test_generated_image_of_another_size_is_resampled_to_the_original():
    rgb = _original()
    cutout, subject = _cutout(rgb)
    generated = Image.new("RGB", (256, 192), (40, 40, 40))
    out = np.array(_run({}, [generated], cutout)[0])
    assert out.shape == rgb.shape
    np.testing.assert_array_equal(out[subject], rgb[subject])


def test_threshold_decides_which_matte_pixels_are_subject():
    rgb = _original()
    cutout, _ = _cutout(rgb, soft=100)
    generated = Image.new("RGB", (200, 160), (0, 0, 0))
    strict = np.array(_run({"feather": 0.0, "threshold": 200}, [generated], cutout)[0])
    loose = np.array(_run({"feather": 0.0, "threshold": 50}, [generated], cutout)[0])
    soft_rows = (slice(10, 20), slice(10, 30))
    assert (strict[soft_rows] == 0).all()
    np.testing.assert_array_equal(loose[soft_rows], rgb[soft_rows])


def test_empty_matte_returns_the_edit_unchanged():
    rgb = _original()
    cutout = Image.fromarray(np.dstack([rgb, np.zeros(rgb.shape[:2], dtype=np.uint8)]), mode="RGBA")
    generated = Image.new("RGB", (200, 160), (7, 8, 9))
    out = np.array(_run({}, [generated], cutout)[0])
    assert (out == (7, 8, 9)).all()


def test_every_edit_in_the_list_is_composited():
    rgb = _original()
    cutout, subject = _cutout(rgb)
    outs = _run({}, [Image.new("RGB", (200, 160), (1, 1, 1)), Image.new("RGB", (200, 160), (2, 2, 2))], cutout)
    assert len(outs) == 2
    for out in outs:
        np.testing.assert_array_equal(np.array(out)[subject], rgb[subject])

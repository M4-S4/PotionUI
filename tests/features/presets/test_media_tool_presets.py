from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts.preset_render import build_fixture_form_data, build_generation_data, build_processor
from src.features.generation.engine import validate_pipe_configuration
from src.features.presets import PresetTemplateLoader
from src.pipelines.contracts import PipeInput
from src.pipelines.pipes.media_tool.main import MediaToolPipe
from src.platform.media_tools import validate_params
from src.pipelines.pipes._shared.media.tools import core_registry


@pytest.fixture(scope="module")
def presets():
    loader = PresetTemplateLoader(["content/presets"])
    loader.load_presets()
    return {Path(template.path).as_posix().rsplit("marketplace/", 1)[-1]: template for template in loader.presets if "marketplace/" in Path(template.path).as_posix()}


def rendered_media_tools(presets, preset_name, mode, **form):
    template = presets[preset_name]
    form_data = build_fixture_form_data(template, mode)
    form_data.update(form)
    pipes = build_processor().process(template, build_generation_data(mode, form_data, template.modes[mode]))
    return [pipe for pipe in pipes if pipe["name"] == "media_tool"]


def cutout():
    image = Image.new("RGBA", (200, 120), (0, 0, 0, 0))
    image.paste((200, 30, 30, 255), (60, 20, 140, 100))
    return image


def run(pipe, image):
    config = validate_pipe_configuration(MediaToolPipe, pipe["config"])
    emitted = []
    output = MediaToolPipe(config).process(PipeInput(input={"image": [image]}), emitted.append).output
    return output["image"], emitted


CASES = [
    ("ImageTools", "fit_canvas", ["fit_canvas"]),
    ("ImageTools", "color_key", ["color_key"]),
    ("ImageTools", "crop_subject", ["crop_subject"]),
    ("ImageTools", "prepare", ["crop_subject", "fit_canvas"]),
]


@pytest.mark.parametrize("preset, mode, tools", CASES)
def test_every_ported_step_names_a_core_tool_with_params_the_platform_accepts(presets, preset, mode, tools):
    steps = [step for step in rendered_media_tools(presets, preset, mode) if step["enabled"]]

    assert [step["config"]["tool"] for step in steps] == tools
    for step in steps:
        spec = core_registry().get(step["config"]["tool"])
        assert validate_params(spec, step["config"]["params"]) is not None


def test_the_crop_subject_mode_trims_a_cutout_to_its_box_plus_margin(presets):
    (step,) = rendered_media_tools(presets, "ImageTools", "crop_subject", crop_margin=10, crop_alpha_threshold=16)

    (image,), _ = run(step, cutout())

    assert image.size == (100, 100)


def test_the_fit_canvas_mode_places_the_image_on_the_chosen_canvas(presets):
    (step,) = rendered_media_tools(presets, "ImageTools", "fit_canvas", canvas_resolution="64x48", scale_percent=50, fill="#00B140")

    (image,), _ = run(step, Image.new("RGB", (200, 100), (255, 0, 0)))

    assert image.size == (64, 48)
    assert image.getpixel((0, 0)) == (0, 177, 64)
    assert image.getpixel((32, 24)) == (255, 0, 0)


def test_the_color_key_mode_keys_a_flat_background_to_alpha(presets):
    (step,) = rendered_media_tools(presets, "ImageTools", "color_key", key_mode="auto", tolerance=25)
    image = Image.new("RGB", (60, 60), (0, 177, 64))
    image.paste((200, 30, 30), (20, 20, 40, 40))

    (keyed,), _ = run(step, image)

    assert keyed.mode == "RGBA"
    assert keyed.getpixel((2, 2))[3] == 0
    assert keyed.getpixel((30, 30))[3] == 255


def test_the_prepare_mode_crops_then_fits_onto_the_canvas(presets):
    steps = [
        step for step in rendered_media_tools(
            presets, "ImageTools", "prepare",
            remove_background=False, trim_to_subject=True, canvas_resolution="64x64", fill="#101010",
        ) if step["enabled"]
    ]

    image = cutout()
    for step in steps:
        (image,), _ = run(step, image)

    assert image.size == (64, 64)


def test_the_status_line_keeps_the_titles_the_old_pipes_had(presets):
    titles = {
        step["id"]: step["config"]["display_title"]
        for mode in ("fit_canvas", "color_key", "crop_subject", "prepare")
        for step in rendered_media_tools(presets, "ImageTools", mode)
    }

    assert titles == {"canvas_fit": "Fitting canvas", "color_key": "Keying color", "crop_subject": "Cropping subject"}

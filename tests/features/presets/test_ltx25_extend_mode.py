from pathlib import Path
from unittest.mock import Mock

import pytest

from src.features.presets import PresetTemplateLoader
from src.features.presets.processor import PresetProcessor
from src.platform.templating.processor import TemplateProcessor

from ._bound_form import bound_form_data


@pytest.fixture(scope="module")
def ltx25_template():
    loader = PresetTemplateLoader(["content/presets"])
    loader.load_presets()
    template = next((p for p in loader.presets if Path(p.path).as_posix().endswith("marketplace/LTX-2.5")), None)
    if template is None:
        pytest.fail("marketplace/LTX-2.5 preset not present")
    return template


def extend(template, **over):
    form_data = {
        "input_video": "/media/clip.mp4",
        "model": "/models/ltx25_dit.safetensors",
        "text_encoder": "/models/gemma4.safetensors",
        "video_vae": "/models/ltx25_video_vae.safetensors",
        "manual_sigmas": "",
        **over,
    }
    form_data = bound_form_data(template, "extend", form_data)
    processor = PresetProcessor(
        template_processor=TemplateProcessor(settings=Mock()), settings=Mock(), preset_template_loader=Mock(),
    )
    pipes = processor.process(template, {"prompts": [], "mode": "extend", "form_data": form_data})
    return {pipe.get("id") or pipe["name"]: pipe for pipe in pipes}


def inputs(pipe):
    return {(item["name"], item["provider"]) for item in pipe["input"]}


def test_by_default_the_new_footage_matches_the_source_size_and_rate(ltx25_template):
    pipes = extend(ltx25_template, direction="forward", duration=4, context_seconds=1.0)

    context = pipes["context"]["config"]
    assert (context["keep"], context["seconds"], context["frame_step"], context["fps"]) == ("tail", 1.0, 8, 0)
    generator = pipes["generator"]["config"]
    assert generator["match_video"] is True and generator["extend_seconds"] == 4
    assert generator["media_placements"] == [
        {"source": "video", "index": 0, "frame": "first", "strength": 1.0, "role": "keyframe"},
    ]
    assert pipes["joined"]["config"]["extension_position"] == "after"


def test_a_manual_size_and_rate_override_the_source(ltx25_template):
    pipes = extend(ltx25_template, match_source=False, resolution="512x768", fps=24)

    assert pipes["context"]["config"]["fps"] == 24
    generator = pipes["generator"]["config"]
    assert generator["match_video"] is False and generator["resolution"] == "512x768" and generator["fps"] == 24


def test_backward_conditions_on_the_head_and_joins_the_new_footage_before_the_source(ltx25_template):
    pipes = extend(ltx25_template, direction="backward", context_seconds=0.5)

    assert pipes["context"]["config"]["keep"] == "head" and pipes["context"]["config"]["seconds"] == 0.5
    assert pipes["generator"]["config"]["media_placements"][0]["frame"] == "end"
    assert pipes["joined"]["config"]["extension_position"] == "before"


def test_the_carried_over_clip_feeds_the_generator_and_sets_the_seam(ltx25_template):
    pipes = extend(ltx25_template)

    assert ("video", "source") in inputs(pipes["context"])
    assert ("video", "context") in inputs(pipes["generator"])
    assert inputs(pipes["joined"]) >= {("video", "source"), ("extension", "generator"), ("overlap", "context")}
    assert ("video", "joined") in inputs(pipes["gallery"])
    assert pipes["joined"]["config"]["keep_audio"] is True

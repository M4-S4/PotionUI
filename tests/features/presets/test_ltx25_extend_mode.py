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


def test_forward_conditions_on_the_tail_and_joins_the_new_footage_after_the_source(ltx25_template):
    pipes = extend(ltx25_template, direction="forward", fps=24, duration=4, context_seconds=1.0)

    assert pipes["context"]["config"]["keep"] == "tail" and pipes["context"]["config"]["frames"] == 17
    generator = pipes["generator"]["config"]
    assert generator["frames"] == 96 + 17
    assert generator["media_placements"] == [
        {"source": "video", "index": 0, "frame": "first", "strength": 1.0, "role": "keyframe"},
    ]
    joined = pipes["joined"]["config"]
    assert joined["extension_position"] == "after" and joined["overlap_frames"] == 17 and joined["fps"] == 24


def test_backward_conditions_on_the_head_and_joins_the_new_footage_before_the_source(ltx25_template):
    pipes = extend(ltx25_template, direction="backward", fps=25, duration=5, context_seconds=0.5)

    assert pipes["context"]["config"]["keep"] == "head" and pipes["context"]["config"]["frames"] == 9
    assert pipes["generator"]["config"]["media_placements"][0]["frame"] == "end"
    assert pipes["generator"]["config"]["frames"] == 125 + 9
    assert pipes["joined"]["config"]["extension_position"] == "before"


def test_the_carried_over_clip_feeds_the_generator_and_the_source_is_joined_whole(ltx25_template):
    pipes = extend(ltx25_template)

    assert ("video", "source") in inputs(pipes["context"])
    assert ("video", "context") in inputs(pipes["generator"])
    assert inputs(pipes["joined"]) >= {("video", "source"), ("extension", "generator")}
    assert ("video", "joined") in inputs(pipes["gallery"])
    assert pipes["joined"]["config"]["keep_audio"] is True


def test_a_very_short_carry_over_still_keeps_one_frame(ltx25_template):
    pipes = extend(ltx25_template, fps=8, context_seconds=0.25)

    assert pipes["context"]["config"]["frames"] == 1 and pipes["joined"]["config"]["overlap_frames"] == 1

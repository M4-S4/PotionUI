from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.features.forms.binding import bind_form
from src.features.presets import PresetTemplateLoader
from src.features.presets.processor import PresetProcessor
from src.platform.templating.processor import TemplateProcessor

_H3_FAST_PRESET_ID = "01KX47H3FASTH30000000000VA"

_MODELS = {
    "model": "/models/minimax_h3.safetensors",
    "text_encoder": "/models/qwen3_vl.safetensors",
    "video_vae": "/models/h3_video_vae.safetensors",
    "audio_vae": "/models/h3_audio_vae.safetensors",
}


@pytest.fixture(scope="module")
def h3_fast_template():
    loader = PresetTemplateLoader(["content/presets"])
    loader.load_presets()
    return next(p for p in loader.presets if p.id == _H3_FAST_PRESET_ID)


def _pipe(pipes, pid):
    return next(p for p in pipes if p.get("id") == pid or p["name"] == pid)


def _inputs(pipe):
    return {edge["name"]: (edge["provider"], edge["output_var"]) for edge in pipe["input"]}


def _process(template, mode, form, prompt=""):
    processor = PresetProcessor(
        template_processor=TemplateProcessor(settings=Mock()), settings=Mock(), preset_template_loader=Mock(),
    )
    bound = bind_form(
        template, mode, form_name=None, raw_form_data={**_MODELS, **form}, user_id=None, storage_dir=None,
    )
    return processor.process(template, {
        "prompts": [{"positive": prompt, "negative": ""}], "mode": mode, "form_data": dict(bound.values),
    })


def test_extend_is_declared(h3_template):
    assert "extend" in h3_template.modes


def test_extend_after_continues_the_source_and_joins_after_it(h3_template):
    pipes = _process(h3_template, "extend", {"source_video": "/media/clip.mp4", "added_seconds": 4}, "a dragon")
    generator = _pipe(pipes, "generator")
    assert _inputs(generator)["source_video"] == ("source", "video")
    assert generator["config"]["source_video_role"] == "continue"
    assert generator["config"]["frames"] == 96
    assert generator["config"]["resolution"] == "auto"
    join = _pipe(pipes, "join")
    assert _inputs(join) == {"video": ("source", "video"), "extension": ("generator", "video")}
    assert join["config"]["extension_position"] == "after"
    assert _inputs(_pipe(pipes, "gallery"))["video"] == ("join", "video")


def test_extend_before_precedes_the_source_and_joins_before_it(h3_template):
    pipes = _process(h3_template, "extend", {"source_video": "/media/clip.mp4", "direction": "before"}, "a dragon")
    assert _pipe(pipes, "generator")["config"]["source_video_role"] == "precede"
    assert _pipe(pipes, "join")["config"]["extension_position"] == "before"


def test_fast_extend_runs_the_fast_recipe(h3_fast_template):
    assert "extend" in h3_fast_template.modes
    pipes = _process(h3_fast_template, "extend", {"source_video": "/media/clip.mp4"}, "a dragon")
    generator = _pipe(pipes, "generator")
    assert generator["config"]["source_video_role"] == "continue"
    assert generator["config"]["steps"] == 8

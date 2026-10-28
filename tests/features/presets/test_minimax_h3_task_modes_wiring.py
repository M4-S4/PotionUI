from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.features.forms.binding import bind_form
from src.features.presets.processor import PresetProcessor
from src.platform.templating.processor import TemplateProcessor

_MODELS = {
    "model": "/models/minimax_h3.safetensors",
    "text_encoder": "/models/qwen3_vl.safetensors",
    "video_vae": "/models/h3_video_vae.safetensors",
    "audio_vae": "/models/h3_audio_vae.safetensors",
}


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


def _document(pipes):
    encoder = next(p for p in pipes if p["name"] == "prompt_encoder")
    assert encoder["config"]["p_prompt"]["input"] == encoder["config"]["p_prompt"]["output"]
    return encoder["config"]["p_prompt"]["output"]


def test_the_form_only_modes_hide_the_prompt(h3_template):
    assert set(h3_template.vars["promptless_modes"]) == {"swap", "remix", "talk"}


@pytest.mark.parametrize("mode", ["swap", "remix", "talk"])
def test_every_task_mode_is_declared(h3_template, mode):
    assert mode in h3_template.modes


def test_swap_sends_the_video_and_every_character_picture(h3_template):
    pipes = _process(h3_template, "swap", {
        "source_video": "/media/clip.mp4", "character": ["/media/a.png", "/media/b.png"],
        "character_description": "a tall woman in a red coat",
    })
    generator = _pipe(pipes, "generator")
    assert generator["config"]["mode"] == "references"
    assert _inputs(generator)["reference_videos"] == ("reference_videos", "video")
    assert _inputs(generator)["reference_images"] == ("references", "image")
    assert generator["config"]["reference_videos"] == ["/media/clip.mp4"]
    assert generator["config"]["length_from"] == "reference_video"
    assert generator["config"]["audio_source"] == "reference_video"
    document = _document(pipes)
    assert "<Picture 1>, <Picture 2>, a tall woman in a red coat." in document
    assert "[video editing] + [reference generation]" in document
    assert "None" not in document


def test_swap_without_keeping_the_sound_generates_it(h3_template):
    pipes = _process(h3_template, "swap", {
        "source_video": "/media/clip.mp4", "character": ["/media/a.png"], "keep_sound": False,
    })
    assert _pipe(pipes, "generator")["config"]["audio_source"] == "generate"


def test_swap_caps_the_length_at_max_seconds(h3_template):
    pipes = _process(h3_template, "swap", {
        "source_video": "/media/clip.mp4", "character": ["/media/a.png"], "max_seconds": 6,
    })
    assert _pipe(pipes, "generator")["config"]["frames"] == 144
    encoder = next(p for p in pipes if p["name"] == "prompt_encoder")
    assert encoder["config"]["reference_video_frames"] == 144


def test_remix_without_pictures_skips_the_picture_loader(h3_template):
    pipes = _process(h3_template, "remix", {"source_video": "/media/clip.mp4", "instruction": "it is snowing"})
    assert _pipe(pipes, "references")["enabled"] is False
    document = _document(pipes)
    assert "[video editing] Re-create <Video 1> with this change: it is snowing" in document
    assert "<Picture" not in document


def test_remix_with_pictures_cites_them(h3_template):
    pipes = _process(h3_template, "remix", {
        "source_video": "/media/clip.mp4", "instruction": "wear this", "pictures": ["/media/coat.png"],
    })
    assert _pipe(pipes, "references")["enabled"] is True
    document = _document(pipes)
    assert "+ [reference generation]" in document
    assert "<Picture 1> (appears in [Shot 1]): attribute_transfer" in document


def test_talk_reuses_the_voice_and_muxes_it(h3_template):
    pipes = _process(h3_template, "talk", {
        "portrait": "/media/face.png", "voice": "/media/voice.wav", "words": "hello there",
    })
    generator = _pipe(pipes, "generator")
    assert _inputs(generator)["reference_audios"] == ("reference_audios", "audio")
    assert _inputs(generator)["audio"] == ("reference_audios", "audio")
    assert generator["config"]["audio_source"] == "file"
    assert generator["config"]["length_from"] == "reference_audio"
    document = _document(pipes)
    assert "[audio reuse]" in document
    assert '<Subject 1> (S1) says, "<d>hello there</d>"' in document


def test_talk_without_words_or_scene_uses_the_defaults(h3_template):
    document = _document(_process(h3_template, "talk", {"portrait": "/media/face.png", "voice": "/media/v.wav"}))
    assert "speaks the words of <Audio 1>" in document
    assert "A medium close-up of <Subject 1>" in document

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.features.forms.binding import bind_form
from src.features.presets import PresetTemplateLoader
from src.features.presets.processor import PresetProcessor
from src.platform.templating.processor import TemplateProcessor


@pytest.fixture(scope="module")
def flux3_template():
    loader = PresetTemplateLoader(["content/presets"])
    loader.load_presets()
    template = next((p for p in loader.presets if "marketplace/Flux3" in str(p.path)), None)
    if template is None:
        pytest.skip("marketplace/Flux3 preset not present")
    return template


def _process(template, form_over: dict | None = None, mode: str = "txt2img"):
    processor = PresetProcessor(
        template_processor=TemplateProcessor(settings=Mock()),
        settings=Mock(),
        preset_template_loader=Mock(),
    )
    form_data = {
        "diffusion_model": "/models/flux3_dit.safetensors",
        "text_encoder": "/models/qwen3vl_4b.safetensors",
        "vae": "/models/flux3_ae.safetensors",
    }
    if mode == "txt2img":
        form_data["resolution"] = "1024x1024"
    if mode == "edit":
        form_data["source_image"] = ["/uploads/a.png", "/uploads/b.png"]
    if form_over:
        form_data.update(form_over)
    bound = bind_form(
        template, mode, form_name=None, raw_form_data=form_data,
        user_id=None, storage_dir=None,
    )
    generation_data = {"prompts": [], "mode": mode, "form_data": dict(bound.values)}
    return processor.process(template, generation_data)


def _pipe(pipes, name):
    return next(p for p in pipes if p.get("id") == name or p["name"] == name)


def _names(pipes):
    return [p["name"] for p in pipes]


def test_modes_and_display_name(flux3_template):
    assert list(flux3_template.modes.keys()) == ["txt2img", "edit"]
    assert flux3_template.name == "FLUX 3 (experimental)"


def test_pipe_names_are_the_flux3_family(flux3_template):
    pipes = _process(flux3_template)
    assert "model_loader/flux3" in _names(pipes)
    assert "generator/flux3" in _names(pipes)


def test_default_profile_defers_steps_and_cfg_to_the_checkpoint(flux3_template):
    cfg = _pipe(_process(flux3_template), "generator/flux3")["config"]
    assert int(cfg["steps"]) == 0
    assert float(cfg["guidance"]) == 0.0
    assert cfg["sampler"] == "euler"


def test_auto_profile_still_encodes_the_negative_prompt(flux3_template):
    cfg = _pipe(_process(flux3_template), "prompt_encoder")["config"]
    assert float(cfg["guidance_scale"]) == 3.5


def test_cfg_zero_star_is_off(flux3_template):
    for mode in ("txt2img", "edit"):
        cfg = _pipe(_process(flux3_template, mode=mode), "generator/flux3")["config"]
        assert cfg["guidance_options"]["cfg_zero_star"] is False


def test_explicit_values_reach_the_generator(flux3_template):
    cfg = _pipe(_process(flux3_template, {"steps": 30, "cfg": 2.0}), "generator/flux3")["config"]
    assert int(cfg["steps"]) == 30
    assert float(cfg["guidance"]) == 2.0
    assert float(_pipe(_process(flux3_template, {"steps": 30, "cfg": 2.0}), "prompt_encoder")["config"]["guidance_scale"]) == 2.0


def test_blank_shift_is_omitted_and_override_reaches_the_generator(flux3_template):
    blank = _pipe(_process(flux3_template), "generator/flux3")["config"]
    assert blank.get("shift") in (None, "")
    override = _pipe(_process(flux3_template, {"shift": 4.0}), "generator/flux3")["config"]
    assert float(override["shift"]) == 4.0


def test_txt2img_passes_the_resolution(flux3_template):
    cfg = _pipe(_process(flux3_template, {"resolution": "1344x768"}), "generator/flux3")["config"]
    assert cfg["resolution"] == "1344x768"


def test_step_cache_form_values_reach_the_generator(flux3_template):
    cfg = _pipe(_process(flux3_template, {
        "step_cache_threshold": 0.1, "step_cache_warmup_steps": 6, "step_cache_max_skips": 2,
    }), "generator/flux3")["config"]
    assert float(cfg["step_cache"]["rel_threshold"]) == 0.1
    assert int(cfg["step_cache"]["warmup_steps"]) == 6
    assert int(cfg["step_cache"]["max_consecutive_skips"]) == 2


def test_loader_has_no_lora_or_vision_configuration(flux3_template):
    cfg = _pipe(_process(flux3_template), "model_loader/flux3")["config"]
    assert set(cfg) >= {"diffusion_model", "text_encoder", "vae"}
    assert "vision" not in cfg


def test_edit_mode_generator_config(flux3_template):
    cfg = _pipe(_process(flux3_template, mode="edit"), "generator/flux3")["config"]
    assert cfg["mode"] == "edit"
    assert cfg["output_size"] == "match"
    assert "resolution" not in cfg


def test_edit_output_size_override_reaches_the_generator(flux3_template):
    cfg = _pipe(_process(flux3_template, {"output_size": "768x1344"}, mode="edit"), "generator/flux3")["config"]
    assert cfg["output_size"] == "768x1344"


def test_edit_media_loader_keeps_the_upload_order(flux3_template):
    media = _pipe(_process(flux3_template, mode="edit"), "media_loader")["config"]["media"]
    assert [m["path"] for m in media] == ["/uploads/a.png", "/uploads/b.png"]


def test_edit_generator_receives_the_reference_images(flux3_template):
    pipe = _pipe(_process(flux3_template, mode="edit"), "generator/flux3")
    assert any(i["name"] == "image" and i["provider"] == "media_loader" for i in pipe["input"])


def test_edit_prompt_encoder_does_not_take_images(flux3_template):
    pipe = _pipe(_process(flux3_template, mode="edit"), "prompt_encoder")
    assert [i["name"] for i in pipe["input"]] == ["text_encoder"]

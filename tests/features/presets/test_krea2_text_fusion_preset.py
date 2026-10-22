from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.features.forms.binding import bind_form
from src.features.presets import PresetTemplateLoader
from src.features.presets.processor import PresetProcessor
from src.platform.templating.processor import TemplateProcessor

from ._bound_form import bound_form_data

_BASE_FORM = {
    "diffusion_model": "/models/krea2.safetensors",
    "text_encoder": "/models/qwen3vl.safetensors",
    "vae": "/models/qwen_image_vae.safetensors",
    "resolution": "1024x1024",
}


@pytest.fixture(scope="module")
def krea2_template():
    loader = PresetTemplateLoader(["content/presets"])
    loader.load_presets()
    template = next((p for p in loader.presets if "marketplace/Krea2" in Path(p.path).as_posix()), None)
    if template is None:
        pytest.fail("marketplace/Krea2 preset not present")
    return template


def _fields(template, mode):
    return template.modes[mode].forms[0].fields


def _find_field(fields, name):
    for field in fields:
        if field.name == name:
            return field
        children = field.children
        if isinstance(children, list):
            found = _find_field(children, name)
            if found is not None:
                return found
    return None


def _find_field_container(fields, label):
    for field in fields:
        if getattr(field, "label", None) == label:
            yield field
        children = field.children
        if isinstance(children, list):
            yield from _find_field_container(children, label)


def _process(template, mode, form_over=None):
    processor = PresetProcessor(
        template_processor=TemplateProcessor(settings=Mock()),
        settings=Mock(),
        preset_template_loader=Mock(),
    )
    form_data = bound_form_data(template, mode, {**_BASE_FORM, **(form_over or {})})
    return processor.process(template, {"prompts": [], "mode": mode, "form_data": form_data})


def _pipe(pipes, pid):
    return next(p for p in pipes if p.get("id") == pid or p["name"] == pid)


def test_the_enhancer_gate_sits_beside_nag_and_is_off_by_default(krea2_template):
    fields = _fields(krea2_template, "txt2img")
    gate = _find_field(fields, "text_enhancer_enabled")
    nag = _find_field(fields, "nag_enabled")

    assert gate is not None and gate.type == "gate"
    assert gate.default is False
    assert gate.configuration["experimental"] is True
    assert gate.description in (None, "")
    names = [c.name for c in gate.children]
    assert names == ["text_enhancer_strength"]
    strength = gate.children[0]
    assert (strength.default, strength.configuration["min"], strength.configuration["max"]) == (1.0, 0.0, 2.0)
    assert gate.formula == nag.formula


def test_the_text_scale_lives_in_an_advanced_text_conditioning_section(krea2_template):
    fields = _fields(krea2_template, "txt2img")
    advanced = next(_find_field_container(fields, "Text conditioning"))
    scale = _find_field(fields, "text_scale")

    assert advanced.audience == "advanced"
    assert [c.name for c in advanced.children] == ["text_scale"]
    assert scale.description in (None, "")
    assert (scale.default, scale.configuration["min"], scale.configuration["max"]) == (1.0, 0.25, 4.0)


def test_the_enhance_mode_gets_no_new_fields(krea2_template):
    fields = _fields(krea2_template, "enhance")

    assert _find_field(fields, "text_enhancer_enabled") is None
    assert _find_field(fields, "text_enhancer_strength") is None
    assert _find_field(fields, "text_scale") is None


def test_an_untouched_form_renders_the_neutral_settings(krea2_template):
    pipes = _process(krea2_template, "txt2img", {"enhance_enabled": True})

    for pid in ("generator", "enhancer"):
        config = _pipe(pipes, pid)["config"]
        assert config["text_enhancer_strength"] == 0.0
        assert config["text_scale"] == 1.0


def test_a_strength_with_the_gate_off_renders_as_off(krea2_template):
    pipes = _process(krea2_template, "txt2img", {"text_enhancer_enabled": False, "text_enhancer_strength": 1.5})

    assert _pipe(pipes, "generator")["config"]["text_enhancer_strength"] == 0.0


def test_the_gate_on_carries_strength_and_scale_to_the_generator_and_the_enhance_pass(krea2_template):
    pipes = _process(krea2_template, "txt2img", {
        "text_enhancer_enabled": True, "text_enhancer_strength": 1.5, "text_scale": 1.75,
        "enhance_enabled": True,
    })

    for pid in ("generator", "enhancer"):
        config = _pipe(pipes, pid)["config"]
        assert config["text_enhancer_strength"] == 1.5
        assert config["text_scale"] == 1.75


def test_the_text_scale_applies_without_the_enhancer_gate(krea2_template):
    pipes = _process(krea2_template, "txt2img", {"text_scale": 2.0})

    config = _pipe(pipes, "generator")["config"]
    assert config["text_enhancer_strength"] == 0.0
    assert config["text_scale"] == 2.0


def test_the_form_defaults_survive_the_request_boundary(krea2_template):
    bound = bind_form(krea2_template, "txt2img", None, dict(_BASE_FORM))

    assert bound.values["text_enhancer_enabled"] is False
    assert bound.values["text_enhancer_strength"] == 1.0
    assert bound.values["text_scale"] == 1.0

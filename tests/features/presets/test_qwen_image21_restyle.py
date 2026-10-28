from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.features.forms.binding import FormBindingError
from src.features.presets import PresetTemplateLoader
from src.features.presets.loader import read_mode_naming
from src.features.presets.mode_labels import mode_display_icon, mode_display_name
from src.features.presets.processor import PresetProcessor
from src.platform.templating.processor import TemplateProcessor

from ._bound_form import bound_form_data

MATTING = "/models/detection_segm/birefnet.safetensors"
BASE = {
    "diffusion_model": "/models/dit.safetensors",
    "text_encoder": "/models/te.safetensors",
    "vae": "/models/vae.safetensors",
    "source_image": "/uploads/photo.png",
}
STYLE = "Restyle the whole image as watercolour. Keep the composition, subjects and layout unchanged."
BACKGROUND = "Replace the background with a white studio. Keep the subject, its pose and its lighting unchanged."
LIGHTING = "Change the lighting to golden hour. Keep the composition, subjects and colours of objects unchanged."
COLOURISE = "Colourise this black and white photo with natural, realistic colours."
RESTORE = (
    "Restore this old photo: remove scratches, dust and fading, repair damage and sharpen detail, "
    "keeping the people and scene exactly as they are."
)


@pytest.fixture(scope="module")
def template():
    loader = PresetTemplateLoader(["content/presets"])
    loader.load_presets()
    found = next((p for p in loader.presets if "marketplace/QwenImage-2.1" in Path(p.path).as_posix()), None)
    if found is None:
        pytest.fail("marketplace/QwenImage-2.1 preset not present")
    return found


def _render(template, **over):
    processor = PresetProcessor(
        template_processor=TemplateProcessor(settings=Mock()), settings=Mock(), preset_template_loader=Mock(),
    )
    form = bound_form_data(template, "restyle", {**BASE, **over})
    pipes = processor.process(template, {"prompts": [], "mode": "restyle", "form_data": form})
    return {p.get("id") or p["name"]: p for p in pipes}


def _prompt(pipes):
    return pipes["prompt_encoder"]["config"]["p_prompt"]["output"]


def test_mode_follows_text_is_named_restyle_and_needs_no_prompt(template):
    assert list(template.modes)[:5] == ["txt2img", "edit", "combine", "text", "restyle"]
    naming = read_mode_naming(Path("content/presets/marketplace/QwenImage-2.1/modes/restyle"))
    assert mode_display_name("restyle", naming["mode_label"]) == "Restyle"
    assert mode_display_icon("restyle") == "palette"
    assert "restyle" in template.vars["promptless_modes"]


@pytest.mark.parametrize("task, text, expected", [
    ("style", "watercolour", STYLE),
    ("background", "a white studio", BACKGROUND),
    ("lighting", "golden hour", LIGHTING),
    ("colourise", "", COLOURISE),
    ("restore", "", RESTORE),
    ("colourise", "  warm tones ", COLOURISE + " Details: warm tones."),
    ("restore", "faces first", RESTORE + " Details: faces first."),
])
def test_each_task_builds_its_documented_prompt(template, task, text, expected):
    pipes = _render(template, restyle_task=task, restyle_text=text)
    assert _prompt(pipes) == expected


def test_default_task_is_style(template):
    assert _prompt(_render(template, restyle_text="watercolour")) == STYLE


def test_history_records_the_same_prompt(template):
    pipes = _render(template, restyle_task="lighting", restyle_text="golden hour")
    emitted = next(p for p in pipes["param_emitter"]["config"]["parameters"] if p and p[0] == "positive_prompt")
    assert emitted[1] == LIGHTING


@pytest.mark.parametrize("task", ["style", "background", "lighting"])
def test_a_description_is_required_for_the_describing_tasks(template, task):
    with pytest.raises(FormBindingError):
        bound_form_data(template, "restyle", {**BASE, "restyle_task": task, "restyle_text": ""})


@pytest.mark.parametrize("task", ["colourise", "restore"])
def test_a_description_is_optional_for_colourise_and_restore(template, task):
    bound_form_data(template, "restyle", {**BASE, "restyle_task": task, "restyle_text": ""})


def test_without_keep_subject_the_flow_is_a_plain_edit(template):
    pipes = _render(template, restyle_task="background", restyle_text="a white studio")
    assert pipes["matting"]["enabled"] is False
    assert pipes["subject_composite"]["enabled"] is False
    gallery = next(i for i in pipes["gallery"]["input"] if i["name"] == "image")
    assert gallery["provider"] == "generator/qwen_image21"
    assert pipes["generator/qwen_image21"]["config"]["mode"] == "edit"


def test_keep_subject_runs_matting_and_composites_over_the_edit(template):
    pipes = _render(
        template, restyle_task="background", restyle_text="a white studio", keep_subject=True, matting_model=MATTING,
    )
    assert pipes["matting"]["enabled"] is True
    assert pipes["matting"]["config"]["model"]["file_path"] == MATTING
    assert {(i["name"], i["provider"]) for i in pipes["matting"]["input"]} == {("image", "media_loader")}
    composite = {i["name"]: (i["provider"], i["output_var"]) for i in pipes["subject_composite"]["input"]}
    assert composite == {"image": ("generator/qwen_image21", "image"), "cutout": ("matting", "image")}
    gallery = next(i for i in pipes["gallery"]["input"] if i["name"] == "image")
    assert gallery["provider"] == "subject_composite"


@pytest.mark.parametrize("task", ["style", "lighting", "colourise", "restore"])
def test_keep_subject_only_applies_to_the_background_task(template, task):
    pipes = _render(template, restyle_task=task, restyle_text="x", keep_subject=True, matting_model=MATTING)
    assert pipes["matting"]["enabled"] is False
    assert pipes["subject_composite"]["enabled"] is False


def test_matting_model_is_required_only_when_keep_subject_is_on(template):
    bound_form_data(template, "restyle", {**BASE, "restyle_task": "background", "restyle_text": "x"})
    with pytest.raises(FormBindingError):
        bound_form_data(
            template, "restyle", {**BASE, "restyle_task": "background", "restyle_text": "x", "keep_subject": True},
        )


def test_generator_resolves_the_edit_profile_defaults(template):
    cfg = _render(template, restyle_text="x")["generator/qwen_image21"]["config"]
    assert int(cfg["steps"]) == 40
    assert float(cfg["guidance"]) == 4.0

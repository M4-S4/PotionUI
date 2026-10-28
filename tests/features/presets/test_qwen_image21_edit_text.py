from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.features.forms.binding import FormBindingError
from src.features.presets import PresetTemplateLoader
from src.features.presets.loader import read_mode_naming
from src.features.presets.mode_labels import mode_display_name
from src.features.presets.processor import PresetProcessor
from src.platform.templating.processor import TemplateProcessor

from ._bound_form import bound_form_data

MASK = "/uploads/source_mask.png"
KEEP = "Keep the font, size, colour and background."


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
    form = {
        "diffusion_model": "/models/dit.safetensors",
        "text_encoder": "/models/te.safetensors",
        "vae": "/models/vae.safetensors",
        "source_image": "/uploads/sign.png",
        "new_text": "OPEN",
        **over,
    }
    form = bound_form_data(template, "text", form)
    pipes = processor.process(template, {"prompts": [], "mode": "text", "form_data": form})
    return {p.get("id") or p["name"]: p for p in pipes}


def _prompt(pipes):
    return pipes["prompt_encoder"]["config"]["p_prompt"]["output"]


def test_mode_follows_combine_is_named_edit_text_and_needs_no_prompt(template):
    assert list(template.modes)[:4] == ["txt2img", "edit", "combine", "text"]
    naming = read_mode_naming(Path("content/presets/marketplace/QwenImage-2.1/modes/text"))
    assert mode_display_name("text", naming["mode_label"]) == "Edit text"
    assert "text" in template.vars["promptless_modes"]


def test_changing_existing_text_quotes_both_strings(template):
    pipes = _render(template, current_text="CLOSED")
    assert _prompt(pipes) == f'Change the text "CLOSED" to "OPEN". {KEEP}'


def test_adding_text_without_current_text(template):
    pipes = _render(template)
    assert _prompt(pipes) == 'Add the text "OPEN". Keep the rest of the image unchanged.'


def test_hint_is_added_after_the_instruction(template):
    pipes = _render(template, current_text="CLOSED", text_hint="in white capitals at the top")
    assert _prompt(pipes) == (
        f'Change the text "CLOSED" to "OPEN". Placement and style: in white capitals at the top. {KEEP}'
    )


def test_unicode_and_mixed_scripts_are_kept_verbatim(template):
    pipes = _render(template, current_text="Café 你好", new_text="Привет, мир! «Ça va» 開店中")
    assert _prompt(pipes) == f'Change the text "Café 你好" to "Привет, мир! «Ça va» 開店中". {KEEP}'


def test_surrounding_whitespace_is_trimmed_but_inner_text_is_not(template):
    pipes = _render(template, current_text="  A  B ", new_text="  C   D  ")
    assert _prompt(pipes).startswith('Change the text "A  B" to "C   D".')


def test_history_records_the_same_assembled_prompt(template):
    pipes = _render(template, current_text="CLOSED")
    emitted = next(p for p in pipes["param_emitter"]["config"]["parameters"] if p and p[0] == "positive_prompt")
    assert emitted[1] == _prompt(pipes)


def test_no_negative_prompt_and_no_prompt_pairs_are_read(template):
    pipes = _render(template)
    config = pipes["prompt_encoder"]["config"]
    assert config["n_prompt"]["output"] == ""
    assert not config.get("pairs")
    assert "dynamic_prompts_renderer" not in pipes


def test_new_text_is_required(template):
    with pytest.raises(FormBindingError):
        bound_form_data(template, "text", {
            "diffusion_model": "/m/d.safetensors", "text_encoder": "/m/t.safetensors", "vae": "/m/v.safetensors",
            "source_image": "/uploads/sign.png", "new_text": "",
        })


def test_without_a_region_the_flow_is_the_plain_edit_flow(template):
    pipes = _render(template)
    for name in ("mask_loader", "inpaint_region_crop", "inpaint_region_restore"):
        assert pipes[name]["enabled"] is False
    image = next(i for i in pipes["prompt_encoder"]["input"] if i["name"] == "image")
    assert (image["provider"], image["output_var"]) == ("media_loader", "image")
    gallery = next(i for i in pipes["gallery"]["input"] if i["name"] == "image")
    assert gallery["provider"] == "generator/qwen_image21"
    assert pipes["generator/qwen_image21"]["config"]["mode"] == "edit"


def test_a_painted_box_crops_and_composites_back(template):
    pipes = _render(template, source_image_inpaint_mask=MASK)
    for name in ("mask_loader", "inpaint_region_crop", "inpaint_region_restore"):
        assert pipes[name]["enabled"] is True
    assert pipes["inpaint_region_crop"]["config"]["strategy"] == "scaled"
    assert pipes["inpaint_region_restore"]["config"]["strategy"] == "composite"
    for name in ("prompt_encoder", "generator/qwen_image21"):
        image = next(i for i in pipes[name]["input"] if i["name"] == "image")
        assert (image["provider"], image["output_var"]) == ("inpaint_region_crop", "images")
    gallery = next(i for i in pipes["gallery"]["input"] if i["name"] == "image")
    assert gallery["provider"] == "inpaint_region_restore"


def test_generator_resolves_the_edit_profile_defaults(template):
    cfg = _render(template)["generator/qwen_image21"]["config"]
    assert int(cfg["steps"]) == 40
    assert float(cfg["guidance"]) == 4.0


def test_blank_current_text_and_hint_give_the_plain_add_prompt(template):
    pipes = _render(template, current_text="   ", text_hint="  ")
    assert _prompt(pipes) == 'Add the text "OPEN". Keep the rest of the image unchanged.'
    assert "Placement" not in _prompt(pipes)


def test_blank_new_text_is_rejected(template):
    with pytest.raises(FormBindingError):
        bound_form_data(template, "text", {
            "diffusion_model": "/m/d.safetensors", "text_encoder": "/m/t.safetensors", "vae": "/m/v.safetensors",
            "source_image": "/uploads/sign.png", "new_text": "   ",
        })

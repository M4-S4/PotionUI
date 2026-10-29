from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.features.forms.binding import FormBindingError
from src.features.presets import PresetTemplateLoader
from src.features.presets.loader import read_mode_naming
from src.features.presets.mode_labels import mode_display_name
from src.features.presets.processor import PresetProcessor
from src.pipelines.pipes.prompt_encoder.main import PromptEncoderPipe
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


def _render(template, prompt="", prompts=None, **over):
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
    pairs = prompts or [{"positive": prompt, "negative": ""}]
    pipes = processor.process(template, {"prompts": pairs, "mode": "text", "form_data": form})
    return {p.get("id") or p["name"]: p for p in pipes}


def _prompt(pipes):
    return pipes["prompt_encoder"]["config"]["p_prompt"]["output"]


def _emitted(pipes, name):
    parameters = pipes["param_emitter"]["config"]["parameters"]
    flat = [entry for item in parameters for entry in (item if item and isinstance(item[0], list) else [item])]
    return [entry[1] for entry in flat if entry and entry[0] == name]


def _final_prompts(pipes, count):
    config = pipes["prompt_encoder"]["config"]
    pipe = PromptEncoderPipe({**PromptEncoderPipe.get_default_config(), **config})
    return [
        pipe._final_prompt_pair(
            config["p_prompt"]["output"], config["p_prompt"]["input"],
            config["n_prompt"]["output"], config["n_prompt"]["input"], config["pairs"], index,
        )
        for index in range(count)
    ]


def test_mode_follows_combine_is_named_edit_text_and_takes_an_optional_prompt(template):
    assert list(template.modes)[:4] == ["txt2img", "edit", "combine", "text"]
    naming = read_mode_naming(Path("content/presets/marketplace/QwenImage-2.1/modes/text"))
    assert mode_display_name("text", naming["mode_label"]) == "Edit text"
    assert "text" not in template.vars["promptless_modes"]
    assert "text" in template.vars["optional_prompt_modes"]


def test_the_style_or_placement_field_is_gone(template):
    stack = list(template.modes["text"].forms[0].fields)
    names = set()
    while stack:
        node = stack.pop()
        if node.name:
            names.add(node.name)
        if isinstance(node.children, list):
            stack.extend(node.children)
    assert "text_hint" not in names
    assert {"current_text", "new_text"} <= names


def test_changing_existing_text_quotes_both_strings(template):
    pipes = _render(template, current_text="CLOSED")
    assert _prompt(pipes) == f'Change the text "CLOSED" to "OPEN". {KEEP}'


def test_adding_text_without_current_text(template):
    pipes = _render(template)
    assert _prompt(pipes) == 'Add the text "OPEN". Keep the rest of the image unchanged.'


def test_the_prompt_is_added_after_the_instruction(template):
    pipes = _render(template, prompt="in white capitals at the top", current_text="CLOSED")
    expected = f'Change the text "CLOSED" to "OPEN". Placement and style: in white capitals at the top. {KEEP}'
    assert _prompt(pipes) == expected
    assert _final_prompts(pipes, 1)[0][0] == expected


def test_the_exact_strings_keep_dynamic_prompt_syntax_verbatim(template):
    pipes = _render(template, prompt="in red", current_text="{A|B}", new_text="${x} {C|D}")
    assert _prompt(pipes) == f'Change the text "{{A|B}}" to "${{x}} {{C|D}}". Placement and style: in red. {KEEP}'


def test_every_image_gets_its_own_placement_and_style(template):
    prompts = [{"positive": "in red", "negative": "blurry"}, {"positive": "in blue", "negative": "noisy"}]
    pipes = _render(template, prompts=prompts, quantity=2, current_text="CLOSED")
    first, second = _final_prompts(pipes, 2)
    assert first == (f'Change the text "CLOSED" to "OPEN". Placement and style: in red. {KEEP}', "blurry")
    assert second == (f'Change the text "CLOSED" to "OPEN". Placement and style: in blue. {KEEP}', "noisy")
    assert _emitted(pipes, "positive_prompt") == [first[0], second[0]]


def test_a_prompt_equal_to_the_current_text_still_lands_in_its_slot(template):
    prompts = [{"positive": "CLOSED", "negative": ""}, {"positive": "OPEN", "negative": ""}]
    first, second = _final_prompts(_render(template, prompts=prompts, quantity=2, current_text="CLOSED"), 2)
    assert first[0] == f'Change the text "CLOSED" to "OPEN". Placement and style: CLOSED. {KEEP}'
    assert second[0] == f'Change the text "CLOSED" to "OPEN". Placement and style: OPEN. {KEEP}'


def test_unicode_and_mixed_scripts_are_kept_verbatim(template):
    pipes = _render(template, current_text="Café 你好", new_text="Привет, мир! «Ça va» 開店中")
    assert _prompt(pipes) == f'Change the text "Café 你好" to "Привет, мир! «Ça va» 開店中". {KEEP}'


def test_surrounding_whitespace_is_trimmed_but_inner_text_is_not(template):
    pipes = _render(template, current_text="  A  B ", new_text="  C   D  ")
    assert _prompt(pipes).startswith('Change the text "A  B" to "C   D".')


def test_history_records_the_same_assembled_prompt(template):
    pipes = _render(template, prompt="in red", current_text="CLOSED")
    assert _emitted(pipes, "positive_prompt") == [_prompt(pipes)]


def test_the_negative_prompt_and_every_prompt_pair_are_read(template):
    pipes = _render(template, prompts=[{"positive": "in red", "negative": "blurry"}])
    config = pipes["prompt_encoder"]["config"]
    assert config["n_prompt"]["output"] == "blurry"
    assert [pair["negative"] for pair in config["pairs"]] == ["blurry"]
    assert pipes["dynamic_prompts_renderer"]["config"]["pairs"] == [{"positive": "in red", "negative": "blurry"}]


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


def test_blank_current_text_and_prompt_give_the_plain_add_prompt(template):
    pipes = _render(template, prompt="  ", current_text="   ")
    assert _prompt(pipes) == 'Add the text "OPEN". Keep the rest of the image unchanged.'
    assert "Placement" not in _prompt(pipes)


def test_blank_new_text_is_rejected(template):
    with pytest.raises(FormBindingError):
        bound_form_data(template, "text", {
            "diffusion_model": "/m/d.safetensors", "text_encoder": "/m/t.safetensors", "vae": "/m/v.safetensors",
            "source_image": "/uploads/sign.png", "new_text": "   ",
        })

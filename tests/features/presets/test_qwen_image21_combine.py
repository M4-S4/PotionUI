from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.features.presets import PresetTemplateLoader
from src.features.presets.loader import read_mode_naming
from src.features.presets.mode_labels import mode_display_name
from src.features.presets.processor import PresetProcessor
from src.platform.templating.processor import TemplateProcessor

from ._bound_form import bound_form_data


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
        "source_image": "/uploads/base.png",
        **over,
    }
    form = bound_form_data(template, "combine", form)
    pipes = processor.process(template, {"prompts": [], "mode": "combine", "form_data": form})
    return {p.get("id") or p["name"]: p for p in pipes}


def _paths(pipes):
    return [m["path"] for m in pipes["media_loader"]["config"]["media"]]


def _prefix(pipes):
    return pipes["prompt_encoder"]["config"]["p_prompt"]["output"]


def test_mode_follows_edit_and_is_named_combine_images(template):
    assert list(template.modes)[:3] == ["txt2img", "edit", "combine"]
    naming = read_mode_naming(Path("content/presets/marketplace/QwenImage-2.1/modes/combine"))
    assert mode_display_name("combine", naming["mode_label"]) == "Combine images"


def test_base_only_loads_one_image_and_names_it_image_one(template):
    pipes = _render(template)
    assert _paths(pipes) == ["/uploads/base.png"]
    assert _prefix(pipes).startswith("Image 1 is the base image to edit.")
    assert "Image 2" not in _prefix(pipes)


def test_references_load_in_slot_order_after_the_base(template):
    pipes = _render(template, reference_1="/u/a.png", reference_3="/u/c.png", reference_2="/u/b.png")
    assert _paths(pipes) == ["/uploads/base.png", "/u/a.png", "/u/b.png", "/u/c.png"]


def test_empty_slots_are_skipped_and_do_not_leave_gaps_in_the_numbering(template):
    pipes = _render(template, reference_1="/u/a.png", reference_4="/u/d.png", reference_4_role="style")
    assert _paths(pipes) == ["/uploads/base.png", "/u/a.png", "/u/d.png"]
    assert "Image 3 is the style." in _prefix(pipes)


def test_roles_are_folded_into_the_prompt_prefix_deterministically(template):
    pipes = _render(
        template,
        reference_1="/u/a.png", reference_1_role="red jacket",
        reference_2="/u/b.png", reference_2_role="  beach background ",
        reference_3="/u/c.png",
    )
    prefix = _prefix(pipes)
    assert prefix.startswith(
        "Image 1 is the base image to edit. Image 2 is the red jacket. "
        "Image 3 is the beach background. Image 4 is a reference. "
    )


def test_all_nine_references_reach_the_encoder_and_generator(template):
    refs = {f"reference_{i}": f"/u/r{i}.png" for i in range(1, 10)}
    pipes = _render(template, **refs)
    assert len(_paths(pipes)) == 10
    for name in ("prompt_encoder", "generator/qwen_image21"):
        image = next(i for i in pipes[name]["input"] if i["name"] == "image")
        assert (image["provider"], image["output_var"]) == ("media_loader", "image")
    assert pipes["generator/qwen_image21"]["config"]["mode"] == "edit"
    assert pipes["model_loader/qwen_image21"]["config"]["vision"] is True


def test_there_is_no_region_machinery(template):
    pipes = _render(template)
    assert not {"mask_loader", "inpaint_region_crop", "inpaint_region_restore"} & set(pipes)
    gallery = next(i for i in pipes["gallery"]["input"] if i["name"] == "image")
    assert gallery["provider"] == "generator/qwen_image21"


def test_a_role_on_an_empty_slot_is_ignored(template):
    pipes = _render(template, reference_1="/u/a.png", reference_2_role="ghost", reference_3_role="phantom")
    prefix = _prefix(pipes)
    assert "ghost" not in prefix and "phantom" not in prefix
    assert "Image 3" not in prefix
    assert "Image 2 is a reference." in prefix
    assert _paths(pipes) == ["/uploads/base.png", "/u/a.png"]


def test_the_generator_resolves_the_edit_profile_defaults(template):
    cfg = _render(template)["generator/qwen_image21"]["config"]
    assert int(cfg["steps"]) == 40
    assert float(cfg["guidance"]) == 4.0

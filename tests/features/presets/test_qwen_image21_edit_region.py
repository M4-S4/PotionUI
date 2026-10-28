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

MASK = "/uploads/source_mask.png"
UNREGIONED_PIPES = [
    "model_loader/qwen_image21",
    "media_loader",
    "prompt_encoder",
    "seed_generator",
    "dynamic_prompts_renderer",
    "from_iotype",
    "param_emitter",
    "generator/qwen_image21",
    "gallery",
]


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
        "source_image": "/uploads/source.png",
        **over,
    }
    form = bound_form_data(template, "edit", form)
    pipes = processor.process(template, {"prompts": [], "mode": "edit", "form_data": form})
    return pipes


def test_the_mode_id_stays_edit_and_is_named_edit_image(template):
    assert "edit" in template.modes
    naming = read_mode_naming(Path("content/presets/marketplace/QwenImage-2.1/modes/edit"))
    assert mode_display_name("edit", naming["mode_label"]) == "Edit image"


def _by_id(pipes):
    return {p.get("id") or p["name"]: p for p in pipes}


def _inputs(pipe):
    return {i["name"]: (i["provider"], i["output_var"]) for i in pipe["input"]}



def test_without_a_region_the_pipeline_is_the_plain_edit_flow(template):
    pipes = _render(template)
    assert [p["name"] for p in pipes if p["enabled"]] == UNREGIONED_PIPES
    by_id = _by_id(pipes)
    for name in ("mask_loader", "inpaint_region_crop", "inpaint_region_restore"):
        assert by_id[name]["enabled"] is False
    assert _inputs(by_id["prompt_encoder"])["image"] == ("media_loader", "image")
    assert _inputs(by_id["generator/qwen_image21"])["image"] == ("media_loader", "image")
    assert _inputs(by_id["gallery"])["image"] == ("generator/qwen_image21", "image")
    assert by_id["media_loader"]["config"]["media"] == [{"type": "image", "path": "/uploads/source.png"}]
    assert by_id["generator/qwen_image21"]["config"]["mode"] == "edit"


def test_a_painted_region_crops_before_and_composites_after(template):
    pipes = _render(template, source_image_inpaint_mask=MASK)
    by_id = _by_id(pipes)
    for name in ("mask_loader", "inpaint_region_crop", "inpaint_region_restore"):
        assert by_id[name]["enabled"] is True
    assert by_id["mask_loader"]["config"]["media"][0]["path"] == MASK
    assert _inputs(by_id["inpaint_region_crop"]) == {
        "image": ("media_loader", "image"),
        "mask": ("mask_loader", "image"),
    }
    assert _inputs(by_id["prompt_encoder"])["image"] == ("inpaint_region_crop", "images")
    assert _inputs(by_id["generator/qwen_image21"])["image"] == ("inpaint_region_crop", "images")
    restore = _inputs(by_id["inpaint_region_restore"])
    assert restore["inpainted_image"] == ("generator/qwen_image21", "image")
    assert restore["crop_region"] == ("inpaint_region_crop", "crop_region")
    assert restore["original_image"] == ("inpaint_region_crop", "original_image")
    assert restore["original_mask"] == ("inpaint_region_crop", "original_mask")
    assert _inputs(by_id["gallery"])["image"] == ("inpaint_region_restore", "image")


def test_region_strategy_values(template):
    by_id = _by_id(_render(template, source_image_inpaint_mask=MASK))
    crop = by_id["inpaint_region_crop"]["config"]
    assert crop["strategy"] == "scaled"
    assert crop["multiple"] == 16
    assert float(crop["context"]) == 0.5
    assert by_id["inpaint_region_restore"]["config"]["strategy"] == "composite"


def test_region_context_field_reaches_the_crop(template):
    by_id = _by_id(_render(template, source_image_inpaint_mask=MASK, region_context="1.0"))
    assert float(by_id["inpaint_region_crop"]["config"]["context"]) == 1.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from src.features.forms.binding import FormBindingError, bind_form
from src.features.presets import PresetTemplateLoader
from src.features.presets.mode_labels import mode_display_name
from src.features.presets.processor import PresetProcessor
from src.platform.runtime.native.arch.image_to_splat import DETAILS
from src.platform.templating.processor import TemplateProcessor

PRESET_DIR = Path("content/presets/marketplace/ImageToSplat")
PIXAL3D_DIR = Path("content/presets/marketplace/Pixal3D")
PRESET_ID = "01KX47GSP4TQ0000000000SP3D"
MOGE = "/models/moge_2_vitl_normal_fp16.safetensors"


@pytest.fixture(scope="module")
def template():
    loader = PresetTemplateLoader(["content/presets"])
    loader.load_presets()
    found = next((p for p in loader.presets if "marketplace/ImageToSplat" in str(p.path).replace("\\", "/")), None)
    if found is None:
        pytest.fail("marketplace/ImageToSplat preset not present")
    return found


def _form(**over) -> dict:
    form = {"source_image": "/uploads/street.png", "geometry_estimator": MOGE}
    form.update(over)
    return {key: value for key, value in form.items() if value is not None}


def _bind(template, **over):
    return bind_form(template, "img2splat", form_name=None, raw_form_data=_form(**over), user_id=None, storage_dir=None)


def _process(template, **over):
    processor = PresetProcessor(
        template_processor=TemplateProcessor(settings=Mock()),
        settings=Mock(),
        preset_template_loader=Mock(),
    )
    bound = _bind(template, **over)
    return processor.process(template, {"prompts": [], "mode": "img2splat", "form_data": dict(bound.values)})


def _config(pipes, name):
    return next(p for p in pipes if p["name"] == name)["config"]


def _fields(directory: Path, mode: str) -> dict:
    found = {}

    def walk(items):
        for item in items or []:
            if isinstance(item, dict):
                if item.get("name"):
                    found[item["name"]] = item
                children = item.get("children")
                if isinstance(children, list):
                    walk(children)

    for tab in sorted((directory / "modes" / mode / "tabs").glob("*.yml")):
        walk(yaml.safe_load(tab.read_text())["fields"])
    return found


def test_the_preset_is_the_contracts_image_to_splat(template):
    assert template.id == PRESET_ID
    assert template.name == "Image to Splat"
    assert template.category == "3d"
    assert list(template.modes) == ["img2splat"]
    assert mode_display_name("img2splat") == "Image to 3D scene"


def test_detail_defaults_to_standard(template):
    assert _bind(template).values["detail"] == "standard"


def test_the_pipeline_loads_moge_and_lifts_the_source_image(template):
    pipes = _process(template, detail="high")
    assert [pipe["name"] for pipe in pipes] == [
        "media_loader", "model_loader/image_to_splat", "param_emitter", "generator/image_to_splat",
    ]
    assert _config(pipes, "media_loader")["media"] == [{"type": "image", "path": "/uploads/street.png"}]
    assert _config(pipes, "model_loader/image_to_splat")["geometry_estimator"]["file_path"] == MOGE
    assert _config(pipes, "generator/image_to_splat")["detail"] == "high"
    generator = next(pipe for pipe in pipes if pipe["name"] == "generator/image_to_splat")
    assert [(link["name"], link["provider"], link["output_var"]) for link in generator["input"]] == [
        ("model", "model_loader", "model"), ("image", "media_loader", "image"),
    ]


@pytest.mark.parametrize("missing", ["source_image", "geometry_estimator"])
def test_both_inputs_are_required(template, missing):
    with pytest.raises(FormBindingError) as refused:
        _bind(template, **{missing: None})
    assert missing in refused.value.field_errors


def test_detail_choices_are_the_generators():
    fields = _fields(PRESET_DIR, "img2splat")
    assert [option["value"] for option in fields["detail"]["configuration"]["options"]] == list(DETAILS)


def test_the_moge_recommendation_is_exactly_pixal3ds():
    ours = _fields(PRESET_DIR, "img2splat")["geometry_estimator"]["configuration"]
    theirs = _fields(PIXAL3D_DIR, "img2mesh")["camera_estimator"]["configuration"]
    assert ours["model_type"] == theirs["model_type"] == "geometry_estimation"
    assert ours["recommendations"] == theirs["recommendations"]


def test_the_source_image_is_the_only_media_field():
    fields = _fields(PRESET_DIR, "img2splat")
    assert [name for name, field in fields.items() if field.get("type") in ("image", "video", "audio", "media")] == [
        "source_image"
    ]


def test_fields_carry_no_description_lines():
    assert [name for name, field in _fields(PRESET_DIR, "img2splat").items() if "description" in field] == []

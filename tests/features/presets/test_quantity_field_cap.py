import copy

import pytest

from scripts.preset_render import (
    ROOT,
    build_fixture_form_data,
    build_generation_data,
    build_processor,
    load_all_presets,
)
from src.features.forms.binding import bind_form
from src.features.generation.engine import deep_update, validate_pipe_configuration
from src.features.presets.fixture_form import default_form_fields
from src.pipelines.catalog import PipeCatalog


def _quantity_max(fields):
    for field in fields:
        if field.name == "quantity":
            return (field.configuration or {}).get("max")
        if isinstance(field.children, list):
            found = _quantity_max(field.children)
            if found is not None:
                return found
    return None


def _cases():
    presets, _ = load_all_presets(presets_root=ROOT / "content" / "presets" / "marketplace", include_plugins=False)
    cases = []
    for preset in sorted(presets, key=lambda p: p.id):
        if preset.category != "video":
            continue
        for mode in sorted(preset.modes):
            cap = _quantity_max(default_form_fields(preset, mode))
            if cap is not None:
                cases.append(pytest.param(preset, mode, cap, id=f"{preset.name}/{mode}"))
    return cases


@pytest.fixture(scope="module")
def processor():
    return build_processor()


@pytest.fixture(scope="module")
def pipe_catalog():
    return PipeCatalog("src/pipelines/pipes", "pipes/custom")


@pytest.mark.parametrize("preset,mode,field_max", _cases())
def test_quantity_field_max_passes_every_pipe_spec(preset, mode, field_max, processor, pipe_catalog):
    form = {**build_fixture_form_data(preset, mode), "quantity": field_max}
    values = dict(bind_form(preset, mode, form_name=None, raw_form_data=form, user_id=None, storage_dir=None).values)
    assert values["quantity"] == field_max
    pipes = processor.process(preset, build_generation_data(mode, values, preset.modes[mode]))

    failures = []
    for pipe in pipes:
        if not pipe.get("enabled"):
            continue
        pipe_class = pipe_catalog.get_pipe(pipe["name"])
        if pipe_class is None:
            continue
        merged = deep_update(copy.deepcopy(pipe_class.get_default_config() or {}), copy.deepcopy(pipe.get("config") or {}))
        try:
            validate_pipe_configuration(pipe_class, merged)
        except ValueError as e:
            failures.append(f"{pipe['name']} (id={pipe.get('id')}): {e}")
    assert not failures, f"quantity {field_max} (the form's max) is rejected: {failures}"

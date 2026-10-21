from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.features.presets import PresetTemplateLoader
from src.features.presets.form_serializer import PresetFormSerializer
from src.platform.plugins.loader import PluginLoader
from src.platform.templating.processor import TemplateProcessor

PLUGIN_IDS = {"comfyui-backend"}

# (preset path suffix, mode, variant or None, has References tab)
CASES = [
    ("comfyui-backend/presets/FluxKlein9b", "txt2img", None, False),
    ("comfyui-backend/presets/FluxKlein9b", "img2img", None, True),
    ("comfyui-backend/presets/Krea-2", "txt2img", None, False),
    ("comfyui-backend/presets/QwenImage", "txt2img", None, False),
    ("comfyui-backend/presets/QwenImage", "img2img", None, True),
    ("comfyui-backend/presets/SDXL", "txt2img", None, False),
    ("comfyui-backend/presets/zImage", "txt2img", None, False),
]

DETAILER_FIELD_NAMES = {"enhancement_strength", "face_detector_model", "hand_detector_model"}

NATIVE_CASES = [
    ("presets/marketplace/Anima", "txt2img"),
    ("presets/marketplace/Flux1", "txt2img"),
    ("presets/marketplace/Flux2", "txt2img"),
    ("presets/marketplace/SDXL", "txt2img"),
    ("presets/marketplace/SDXL", "inpaint"),
    ("presets/marketplace/SeedVR2", "upscale"),
    ("presets/marketplace/SeedVR2", "video_upscale"),
    ("presets/marketplace/Wan", "video"),
    ("presets/marketplace/ZImage", "txt2img"),
    ("presets/marketplace/LTX-2", "video"),
]

NATIVE_SAMPLING_CASES = [c for c in NATIVE_CASES if c[0] != "presets/marketplace/SeedVR2"]

GATE_CASES = [
    ("presets/marketplace/Wan", "video", "Advanced", None, "riflex", {"riflex_trained_frames"}, False),
    ("presets/marketplace/LTX-2", "video", "Enhance", None, "enhance_faces_hands", DETAILER_FIELD_NAMES, False),
    ("presets/marketplace/LTX-2", "upscale", "Generation", "Enhancement", "enhance_faces_hands", DETAILER_FIELD_NAMES, False),
    *[
        ("presets/marketplace/SDXL", mode, "Advanced", None, gate, children, default)
        for mode in ("txt2img", "inpaint")
        for gate, children, default in (
            ("adm_guidance_enabled", {"adm_positive_scale", "adm_negative_scale", "adm_scaler_end"}, True),
            ("sag_enabled", {"sag_scale", "sag_sigma", "sag_threshold"}, False),
        )
    ],
]


@pytest.fixture(scope="module")
def loader():
    manifests = [m for m in PluginLoader().discover_plugins() if m.id in PLUGIN_IDS]
    registry = SimpleNamespace(get_enabled_plugins=lambda: list(manifests))
    loader = PresetTemplateLoader(
        ["content/presets/marketplace", "content/presets/local"], plugin_registry=registry
    )
    loader.load_presets()
    return loader


@pytest.fixture(scope="module")
def serializer(loader):
    return PresetFormSerializer(loader, TemplateProcessor(settings=Mock()))


def _preset(loader, suffix):
    matches = [p for p in loader.presets if str(p.path).replace("\\", "/").endswith(suffix)]
    assert len(matches) == 1, f"{suffix} matched {len(matches)} presets"
    return matches[0]


def _tabs(loader, serializer, suffix, mode, variant):
    preset = _preset(loader, suffix)
    forms = preset.modes[mode].forms
    if variant:
        form = next(f for f in forms if getattr(f, "name", None) == variant)
    else:
        form = forms[0]
    schema = serializer.process_form_fields(form, preset.id)
    container = schema["properties"].get("tabs")
    assert container is not None, f"{suffix}/{mode}/{variant} form is not a tabs container"
    return container["children"]


def _by_title(tabs, title):
    titles = [t.get("title") for t in tabs]
    assert title in titles, f"expected a {title!r} tab, tabs are {titles}"
    return tabs[titles.index(title)]


def _all_field_names(node):
    found = set()
    for child in node.get("children") or []:
        if child.get("name"):
            found.add(child["name"])
        found |= _all_field_names(child)
    return found


def _case_id(case):
    suffix, mode, variant, _ = case
    return f"{suffix.rsplit('/', 1)[-1]}-{mode}-{variant or 'default'}"


def _native_id(case):
    return f"{case[0].rsplit('/', 1)[-1]}-{case[1]}"


def _gate_id(case):
    return f"{case[0].rsplit('/', 1)[-1]}-{case[1]}-{case[4]}"


def _find(node, predicate):
    for child in node.get("children") or []:
        if predicate(child):
            return child
        found = _find(child, predicate)
        if found:
            return found
    return None


def _names_in_order(node):
    out = []
    for child in node.get("children") or []:
        if child.get("name"):
            out.append(child["name"])
        out.extend(_names_in_order(child))
    return out


@pytest.mark.parametrize("suffix,mode,variant,has_references", CASES, ids=[_case_id(c) for c in CASES])
def test_tab_order_and_icons(loader, serializer, suffix, mode, variant, has_references):
    tabs = _tabs(loader, serializer, suffix, mode, variant)
    titles = [t.get("title") for t in tabs]

    generation = _by_title(tabs, "Generation")
    assert generation["configuration"]["icon"] == "generation"

    advanced = _by_title(tabs, "Advanced")
    assert advanced["configuration"]["icon"] == "settings"
    assert advanced["audience"] == "advanced"
    assert titles[-1] == "Advanced", f"{suffix}/{mode}/{variant} tabs are {titles}"

    if has_references:
        ref_index = titles.index("References")
        assert titles[ref_index]
        assert titles.index("Generation") < ref_index < titles.index("LoRA")
        references = tabs[ref_index]
        assert references["configuration"]["icon"] == "image"
    else:
        assert "References" not in titles, f"{suffix}/{mode}/{variant} unexpectedly has a References tab"

    assert titles.index("Generation") < titles.index("LoRA") < titles.index("Advanced")


@pytest.mark.parametrize("suffix,mode,variant,has_references", CASES, ids=[_case_id(c) for c in CASES])
def test_generation_tab_has_image_then_models_sections(loader, serializer, suffix, mode, variant, has_references):
    generation = _by_title(_tabs(loader, serializer, suffix, mode, variant), "Generation")
    top_level = generation["children"]
    sections = [c for c in top_level if c.get("type") == "section"]
    titles = [s.get("title") for s in sections]
    assert titles == ["Image", "Models"], f"{suffix}/{mode}/{variant} Generation sections are {titles}"

    models_section = sections[titles.index("Models")]
    model_field_names = {c["name"] for c in models_section["children"] if c.get("type") == "model"}
    assert model_field_names, f"{suffix}/{mode}/{variant} Models section has no model pickers"
    # Every model field lives in the Models section, none stray elsewhere on the tab.
    assert _all_field_names(generation) & model_field_names == model_field_names


@pytest.mark.parametrize("suffix,mode,variant,has_references", CASES, ids=[_case_id(c) for c in CASES])
def test_advanced_tab_top_level_is_named_sections_sampling_first(
    loader, serializer, suffix, mode, variant, has_references
):
    advanced = _by_title(_tabs(loader, serializer, suffix, mode, variant), "Advanced")
    top_level = advanced["children"]
    # A gate is never wrapped in a section, so it is the one non-section
    # container allowed at the top level.
    assert all(c.get("type") in ("section", "gate") for c in top_level), (
        f"{suffix}/{mode}/{variant} Advanced tab has a non-section top-level child: "
        f"{[c.get('type') for c in top_level]}"
    )
    assert top_level[0].get("title") == "Sampling", (
        f"{suffix}/{mode}/{variant} Advanced tab's first section is "
        f"{top_level[0].get('title')!r}, not 'Sampling'"
    )
    sampling_names = [c.get("name") for c in top_level[0]["children"] if c.get("name")]
    assert sampling_names[0] == "steps", (
        f"{suffix}/{mode}/{variant} Sampling section doesn't start with steps: {sampling_names}"
    )


@pytest.mark.parametrize("suffix,mode,variant,has_references", CASES, ids=[_case_id(c) for c in CASES])
def test_no_group_or_header_containers_anywhere_on_the_form(
    loader, serializer, suffix, mode, variant, has_references
):
    def _walk(node, banned_types):
        for child in node.get("children") or []:
            assert child.get("type") not in banned_types, (
                f"{suffix}/{mode}/{variant} has a {child.get('type')!r} container "
                f"({child.get('name') or child.get('title')})"
            )
            _walk(child, banned_types)

    for tab in _tabs(loader, serializer, suffix, mode, variant):
        _walk(tab, {"group", "header"})


def test_qwenimage_references_tab_holds_only_media_fields(loader, serializer):
    references = _by_title(_tabs(loader, serializer, "comfyui-backend/presets/QwenImage", "img2img", None), "References")
    assert _all_field_names(references) == {"source_image", "ref_image_2", "ref_image_3"}


def test_fluxklein9b_references_tab_holds_only_media_fields(loader, serializer):
    references = _by_title(_tabs(loader, serializer, "comfyui-backend/presets/FluxKlein9b", "img2img", None), "References")
    assert _all_field_names(references) == {"input_image", "input_image_2"}


def test_quantity_is_a_stepper_with_the_original_range(loader, serializer):
    """Rule: quantity's type moves to stepper, but its min/max/step must not
    change (the ComfyUI pipe loops on it -- a max of 4 must stay 4, not grow
    to the native-preset convention of 10)."""
    for suffix, mode, variant, _ in CASES:
        preset = _preset(loader, suffix)
        if "quantity" not in _all_field_names({"children": _tabs(loader, serializer, suffix, mode, variant)}):
            continue
        generation = _by_title(_tabs(loader, serializer, suffix, mode, variant), "Generation")

        def _find(node, name):
            for child in node.get("children") or []:
                if child.get("name") == name:
                    return child
                found = _find(child, name)
                if found:
                    return found
            return None

        quantity = _find(generation, "quantity")
        if quantity is None:
            continue
        assert quantity["type"] == "stepper", f"{suffix}/{mode}/{variant} quantity is {quantity['type']!r}"
        assert quantity["minimum"] == 1
        assert quantity["maximum"] == 4
        assert quantity["step"] == 1


@pytest.mark.parametrize("suffix,mode", NATIVE_CASES, ids=[_native_id(c) for c in NATIVE_CASES])
def test_native_advanced_tab_has_no_group_or_header_containers(loader, serializer, suffix, mode):
    advanced = _by_title(_tabs(loader, serializer, suffix, mode, None), "Advanced")
    offender = _find(advanced, lambda c: c.get("type") in {"group", "header"})
    assert offender is None, f"{suffix}/{mode} Advanced tab has a {offender['type']!r} container"


@pytest.mark.parametrize("suffix,mode", NATIVE_SAMPLING_CASES, ids=[_native_id(c) for c in NATIVE_SAMPLING_CASES])
def test_native_advanced_tab_is_named_sections_sampling_first(loader, serializer, suffix, mode):
    top_level = _by_title(_tabs(loader, serializer, suffix, mode, None), "Advanced")["children"]
    assert all(c.get("type") in ("section", "gate") for c in top_level), [c.get("type") for c in top_level]
    assert top_level[0].get("title") == "Sampling"
    assert _names_in_order(top_level[0])[0] == "steps"


@pytest.mark.parametrize("suffix,mode,tab,section,gate_name,children,default", GATE_CASES, ids=[_gate_id(c) for c in GATE_CASES])
def test_gate_owns_its_boolean_and_children_carry_no_reactions(
    loader, serializer, suffix, mode, tab, section, gate_name, children, default
):
    container = _by_title(_tabs(loader, serializer, suffix, mode, None), tab)
    if section:
        container = _find(container, lambda c: c.get("title") == section)
    gate = _find(container, lambda c: c.get("name") == gate_name)
    assert gate["type"] == "gate"
    assert gate["default"] is default
    assert _all_field_names(gate) == children
    assert not [c["name"] for c in gate["children"] if c.get("reactions")]

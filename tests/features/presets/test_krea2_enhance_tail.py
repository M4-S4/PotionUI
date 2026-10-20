import pytest

from scripts.preset_render import (
    ROOT,
    build_fixture_form_data,
    build_form_serializer,
    build_generation_data,
    build_processor,
    load_all_presets,
)


@pytest.fixture(scope="module")
def krea2():
    presets, _ = load_all_presets(presets_root=ROOT / "content" / "presets" / "marketplace", include_plugins=False)
    preset = next(p for p in presets if p.path.replace("\\", "/").rstrip("/").endswith("/Krea2"))
    return preset, build_processor(), build_form_serializer()


def _refine_tail(krea2, mode, **overrides):
    preset, processor, serializer = krea2
    form = build_fixture_form_data(serializer, preset, mode)
    form.update(overrides)
    pipes = processor.process(preset, build_generation_data(mode, form, preset.modes.get(mode)))
    configs = [p["config"] for p in pipes if p["name"].startswith("generator") and p["config"].get("refine_tail")]
    return [c["refine_tail"] for c in configs]


@pytest.mark.parametrize("mode", ["enhance", "txt2img"])
@pytest.mark.parametrize(
    "upscale_by,detail,expected",
    [
        ("4x", "subtle", "balanced"),
        ("4x", "strong", "strong"),
        ("2x", "subtle", "subtle"),
    ],
)
def test_subtle_falls_back_to_balanced_only_at_4x(krea2, mode, upscale_by, detail, expected):
    overrides = {"upscale_by": upscale_by, "enhance_detail": detail}
    if mode == "txt2img":
        overrides["enhance_enabled"] = True
    tails = _refine_tail(krea2, mode, **overrides)
    assert tails
    assert set(tails) == {expected}

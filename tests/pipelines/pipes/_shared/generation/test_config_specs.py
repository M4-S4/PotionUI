from pathlib import Path

import pytest

from src.pipelines.catalog import PipeCatalog
from src.pipelines.pipes._shared.generation.config_specs import (
    device_config_spec,
    dtype_config_spec,
    loras_config_spec,
    nag_config_specs,
    preview_config_spec,
    seed_config_spec,
    vram_limit_config_spec,
)
from src.pipelines.pipes._shared.generation.loader_base import BaseModelLoaderPipe

REPO_ROOT = Path(__file__).resolve().parents[5]


def shape(spec):
    return (spec.name, spec.param_type, spec.default, spec.required, spec.choices, spec.min_value, spec.max_value)


def test_device_spec():
    assert shape(device_config_spec()) == ("device", str, "cuda", False, ["cuda", "cpu"], None, None)


def test_dtype_spec():
    assert shape(dtype_config_spec()) == (
        "dtype", str, "bfloat16", False, ["bfloat16", "float16", "float32"], None, None,
    )


def test_vram_limit_spec():
    assert shape(vram_limit_config_spec()) == ("vram_limit_gb", float, None, False, None, None, None)


def test_seed_spec_takes_a_description():
    spec = seed_config_spec("Seed")
    assert shape(spec) == ("seed", int, -1, False, None, -1, None)
    assert spec.description == "Seed"
    assert seed_config_spec().description == "Random seed"


def test_preview_spec():
    assert shape(preview_config_spec("x")) == ("preview", bool, True, False, None, None, None)


def test_loras_spec_default_is_not_shared():
    first, second = loras_config_spec("a"), loras_config_spec("b")
    assert shape(first) == ("loras", list, [], False, None, None, None)
    first.default.append(1)
    assert second.default == []


def test_nag_specs_order_and_bounds():
    specs = nag_config_specs("scale")
    assert [shape(s) for s in specs] == [
        ("nag_scale", float, 1.0, False, None, 1.0, 20.0),
        ("nag_tau", float, 3.5, False, None, 0.1, 20.0),
        ("nag_alpha", float, 0.5, False, None, 0.0, 1.0),
    ]
    assert specs[0].description == "scale"


def test_loader_runtime_specs():
    assert [shape(s) for s in BaseModelLoaderPipe.runtime_config_specs()] == [
        shape(device_config_spec()), shape(dtype_config_spec()), shape(vram_limit_config_spec()),
    ]


@pytest.fixture(scope="module")
def catalog():
    catalog = PipeCatalog(str(REPO_ROOT / "src" / "pipelines" / "pipes"), str(REPO_ROOT / "pipes" / "custom"))
    catalog.discover_pipes()
    return catalog


def _pipe_names():
    catalog = PipeCatalog(str(REPO_ROOT / "src" / "pipelines" / "pipes"), str(REPO_ROOT / "pipes" / "custom"))
    catalog.discover_pipes()
    return sorted(catalog.pipes)


@pytest.mark.parametrize("name", _pipe_names())
def test_default_config_keys_match_configuration_names(catalog, name):
    pipe = catalog.pipes[name]
    assert list(pipe.get_default_config()) == [spec.name for spec in pipe.configuration()]


def spec_of(catalog, pipe, name):
    return next(s for s in catalog.pipes[pipe].configuration() if s.name == name)


def test_reconciled_defaults(catalog):
    assert spec_of(catalog, "tiled_detailer/sdxl", "vram_limit_gb").default == 8
    assert spec_of(catalog, "checkpoint_loader/sdxl", "clip_skip").default == 2


def test_detailer_detections_default(catalog):
    detections = spec_of(catalog, "detailer/sdxl", "detections").default
    assert set(detections) == {"face", "hand", "eyes", "teeth", "person"}
    assert all("device" not in entry for entry in detections.values())
    assert detections["hand"]["confidence"] == 0.5
    assert detections["face"]["confidence"] == 0.35

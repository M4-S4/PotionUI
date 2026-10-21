from __future__ import annotations

import pytest

from src.features.generation.engine import validate_pipe_configuration
from src.pipelines.catalog import PipeCatalog
from src.pipelines.contracts import IOType


def catalog():
    return PipeCatalog("src/pipelines/pipes", "pipes/custom")


def test_the_catalog_discovers_the_pipe():
    pipe = catalog().get_pipe("model_tools")
    assert pipe is not None
    assert pipe.name == "model_tools"
    assert not pipe.outputs()


def test_the_pipe_asks_for_the_models_service_only():
    pipe = catalog().get_pipe("model_tools")
    specs = pipe.inputs()
    assert [(s.name, s.io_type, s.required) for s in specs] == [("MODELS", IOType.SERVICE, False)]


def test_default_configuration_validates_and_is_not_admin():
    pipe = catalog().get_pipe("model_tools")
    config = validate_pipe_configuration(pipe, pipe.get_default_config())
    assert config["is_admin"] is False
    assert config["overwrite"] is False
    assert config["mode"] == "quantize"
    assert config["quant_format"] == "fp8_e4m3"


def test_the_pipe_offers_exactly_the_formats_the_quantizer_writes():
    from src.pipelines.pipes.model_tools.main import QUANT_FORMATS
    from src.platform.runtime.native.ops.model_tools.formats import QUANT_FORMATS as WRITERS

    assert QUANT_FORMATS == list(WRITERS)


def test_unknown_format_fails_configuration_validation():
    pipe = catalog().get_pipe("model_tools")
    with pytest.raises(ValueError):
        validate_pipe_configuration(pipe, {**pipe.get_default_config(), "quant_format": "int3"})


def test_unknown_mode_fails_configuration_validation():
    pipe = catalog().get_pipe("model_tools")
    with pytest.raises(ValueError):
        validate_pipe_configuration(pipe, {**pipe.get_default_config(), "mode": "explode"})

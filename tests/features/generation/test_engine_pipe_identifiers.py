from unittest.mock import Mock

import pytest

from src.features.generation.engine import GenerationEngine
from src.pipelines.contracts import BasePipe, IOType, PipeInputSpec, PipeOutput, PipeOutputSpec

SEEN = {}


class _Base(BasePipe):
    description = "test pipe"

    def process(self, pipe_input, generation_outputs):
        return PipeOutput(output={"text": self.emits})

    @classmethod
    def get_default_config(cls):
        return {}

    @classmethod
    def configuration(cls):
        return []

    @classmethod
    def inputs(cls):
        return []

    @classmethod
    def outputs(cls):
        return [PipeOutputSpec("text", IOType.INT)]


class EmitA(_Base):
    name = "emit_a"
    emits = 1


class EmitB(_Base):
    name = "emit_b"
    emits = 2


class Sink(_Base):
    name = "sink"

    def process(self, pipe_input, generation_outputs):
        SEEN["text"] = pipe_input.input["text"]
        return PipeOutput(output={})

    @classmethod
    def inputs(cls):
        return [PipeInputSpec("text", IOType.INT, True)]

    @classmethod
    def outputs(cls):
        return []


CATALOG = {cls.name: cls for cls in (EmitA, EmitB, Sink)}


def _engine():
    catalog = Mock()
    catalog.get_pipe = Mock(side_effect=CATALOG.get)
    return GenerationEngine(
        gpu=Mock(),
        pipe_catalog=catalog,
        settings=Mock(),
        system_monitor=Mock(),
        memory_advisor=Mock(),
        llm_service=Mock(),
    )


def _pipe(name, pipe_id, inputs=()):
    return {"name": name, "id": pipe_id, "enabled": True, "config": {}, "input": list(inputs)}


@pytest.fixture(autouse=True)
def _clear_seen():
    SEEN.clear()


def _pipes_with_idless_producers():
    return [
        _pipe("emit_a", None),
        _pipe("emit_b", None),
        _pipe("sink", "sink_1", [{"name": "text", "provider": "emit_b", "output_var": "text"}]),
    ]


def test_validate_accepts_idless_producers_referenced_by_name():
    _engine().validate_pipeline(_pipes_with_idless_producers())


def test_generate_routes_each_idless_pipes_output_to_its_own_consumer():
    _engine().generate(_pipes_with_idless_producers(), Mock(), generation_id="g1")

    assert SEEN["text"] == 2


def test_failure_point_names_an_idless_pipe_by_its_name():
    class Boom(_Base):
        name = "boom"

        def process(self, pipe_input, generation_outputs):
            raise RuntimeError("kaput")

    CATALOG["boom"] = Boom
    outputs = []
    try:
        with pytest.raises(RuntimeError):
            _engine().generate([_pipe("boom", None)], outputs.append, generation_id="g3")
    finally:
        del CATALOG["boom"]

    errors = [o for o in outputs if type(o).__name__ == "ErrorGenerationOutput"]
    assert [e.pipe_key for e in errors] == ["boom"]

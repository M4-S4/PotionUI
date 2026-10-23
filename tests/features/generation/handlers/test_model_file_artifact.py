import json
from unittest.mock import Mock

from src.features.generation.handlers import artifact_handlers
from src.features.generation.output_processor import OutputProcessor
from src.features.generation.output_serializer import GenerationOutputSerializer
from src.features.generation.output_types import output_type_registry
from src.features.generation.run_report_recorder import RunReportRecorder
from src.features.generation.run_report_repository import GenerationRunReportRepository
from src.pipelines.outputs import ModelFileGenerationOutput

SUMMARY = {
    "layers_quantized": 412,
    "layers_kept": 38,
    "worst_layers": [{"key": "blocks.3.attn.qkv", "rel_l2": 0.031, "cosine": 0.9994}],
    "energy_kept": 0.987,
    "sources": [{"name": "style.safetensors", "sha256": "ab" * 32, "strength": 0.8}],
    "licence_note": "Derived from a non-commercial model",
}


def _output(**overrides):
    fields = dict(
        pipe_id=2,
        pipe_name="quantize",
        path="/models/diffusion_models/flux-fp8.safetensors",
        kind="model",
        model_type="diffusion_model",
        file_name="flux-fp8.safetensors",
        size_bytes=11_900_000_000,
        source_size_bytes=23_800_000_000,
        format="fp8_scaled",
        summary=SUMMARY,
    )
    fields.update(overrides)
    return ModelFileGenerationOutput(**fields)


def _handler(indexer):
    handler = artifact_handlers.ModelFileGenerationOutputHandler("gen-1", "user-1", None, Mock())
    handler.model_file_indexer = indexer
    return handler


def test_registered_as_a_pipe_artifact_with_its_handler():
    spec = output_type_registry.spec_for(_output())
    assert spec.key == "model_file"
    assert spec.resolve_message_type(_output()) == "pipe_artifact"
    assert spec.handler_cls is artifact_handlers.ModelFileGenerationOutputHandler
    assert not spec.server_only


def test_handler_indexes_the_written_path_and_records_the_model_id():
    indexer = Mock(return_value={"model_id": "m-9", "indexed": True, "warning": None})
    output = _output()

    metadata = _handler(indexer).handle(output)

    indexer.assert_called_once_with("/models/diffusion_models/flux-fp8.safetensors")
    assert metadata["processed"] is True
    assert metadata["indexed"] is True
    assert output.model_id == "m-9"
    assert output.indexed is True


def test_handler_survives_an_indexing_failure_and_reports_it():
    indexer = Mock(side_effect=RuntimeError("not under any known model root"))
    output = _output()

    metadata = _handler(indexer).handle(output)

    assert metadata["processed"] is True
    assert metadata["indexed"] is False
    assert output.model_id is None
    assert output.indexed is False
    assert "known model root" in output.index_warning


def test_handler_without_an_indexer_still_processes():
    output = _output()
    metadata = _handler(None).handle(output)
    assert metadata["processed"] is True
    assert output.indexed is False


def test_handler_does_not_copy_the_file_into_generation_storage():
    storage = Mock()
    handler = artifact_handlers.ModelFileGenerationOutputHandler("gen-1", "user-1", None, storage)
    handler.model_file_indexer = Mock(return_value={"model_id": "m", "indexed": True})

    metadata = handler.handle(_output())

    assert storage.mock_calls == []
    assert "saved_path" not in metadata
    assert "file_size" not in metadata


async def test_processor_hands_its_indexer_to_the_handler():
    indexer = Mock(return_value={"model_id": "m-1", "indexed": True, "warning": None})
    processor = OutputProcessor(Mock())
    processor.model_file_indexer = indexer
    output = _output()

    metadata = await processor._process_via_spec(output, "gen-1", "user-1")

    assert metadata["handler"] == "ModelFileGenerationOutputHandler"
    indexer.assert_called_once()
    assert output.model_id == "m-1"


def test_serialized_message_shape_after_indexing():
    output = _output(model_id="m-9", indexed=True)
    message = GenerationOutputSerializer(generation_id="gen-1").serialize_output(output)
    decoded = json.loads(json.dumps(message))

    assert decoded["type"] == "pipe_artifact"
    assert decoded["artifact_type"] == "model_file"
    assert decoded["generation_id"] == "gen-1"
    assert decoded["pipe_id"] == 2
    assert "path" not in decoded["artifact_data"]
    assert decoded["artifact_data"] == {
        "index": 0,
        "kind": "model",
        "model_type": "diffusion_model",
        "file_name": "flux-fp8.safetensors",
        "size_bytes": 11_900_000_000,
        "source_size_bytes": 23_800_000_000,
        "format": "fp8_scaled",
        "summary": SUMMARY,
        "model_id": "m-9",
        "indexed": True,
        "index_warning": None,
    }


def test_file_name_falls_back_to_the_path_basename():
    output = _output(file_name="", source_size_bytes=None, summary={})
    data = GenerationOutputSerializer(generation_id="gen-1").serialize_output(output)["artifact_data"]
    assert data["file_name"] == "flux-fp8.safetensors"
    assert data["source_size_bytes"] is None
    assert data["summary"] == {}


def test_lora_variant_serializes_its_kind_and_type():
    output = _output(kind="lora", model_type="lora", format="lora", summary={"rank": 32, "energy_kept": 0.93})
    data = GenerationOutputSerializer(generation_id="gen-1").serialize_output(output)["artifact_data"]
    assert data["kind"] == "lora"
    assert data["model_type"] == "lora"
    assert data["summary"] == {"rank": 32, "energy_kept": 0.93}


def test_run_report_keeps_the_card_data_for_history():
    recorder = RunReportRecorder(Mock(spec=GenerationRunReportRepository))
    message = GenerationOutputSerializer(generation_id="gen-1").serialize_output(
        _output(model_id="m-9", indexed=True)
    )

    recorder.record_output("gen-1", message)
    report = recorder.flush("gen-1", terminal_status="completed")

    [entry] = report["artifacts"]
    assert entry["artifact_type"] == "model_file"
    assert "omitted" not in entry
    assert entry["artifact_data"]["summary"] == SUMMARY
    assert entry["artifact_data"]["model_id"] == "m-9"
    assert entry["artifact_data"]["size_bytes"] == 11_900_000_000

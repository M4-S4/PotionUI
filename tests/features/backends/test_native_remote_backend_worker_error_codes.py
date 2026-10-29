from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.features.backends.backend_config import NativeRemoteBackendConfig
from src.features.backends.native_remote_backend import RemoteNativeBackend, worker_failure_output
from src.features.generation.failure import failure_from_output
from src.pipelines.outputs import ErrorGenerationOutput
from src.platform.worker_protocol import JobErrorV1, JobEventV1


def _failed_event(code, message, detail=None):
    return JobEventV1(
        execution_id="gen-1", worker_id="worker-1", cursor=3, emitted_at=datetime(2026, 10, 29, tzinfo=timezone.utc),
        kind="failed", error=JobErrorV1(code=code, message=message, detail=detail),
    )


async def _outputs_of(event):
    backend = RemoteNativeBackend(NativeRemoteBackendConfig(
        id="remote-1", name="RunPod A100", base_url="http://fake-worker", worker_token="tok", enabled=True,
    ))
    outputs = []
    await backend._handle_event(event, None, Path("imports"), {}, {}, outputs.append)
    return outputs


@pytest.mark.parametrize("code", ["expired", "model_digest_mismatch", "pipe_failed", "worker_error"])
async def test_a_worker_only_code_lands_in_the_detail_and_leaves_the_category_to_the_message(code):
    outputs = await _outputs_of(_failed_event(code, "package expired before execution"))

    assert len(outputs) == 1 and isinstance(outputs[0], ErrorGenerationOutput)
    assert outputs[0].error_code is None
    assert outputs[0].error == "package expired before execution"
    assert outputs[0].detail == f"Worker error code: {code}"


async def test_a_worker_detail_keeps_its_text_after_the_code():
    outputs = await _outputs_of(_failed_event("pipe_failed", "boom", detail="Traceback (most recent call last):\n  boom"))

    assert outputs[0].detail == "Worker error code: pipe_failed\n\nTraceback (most recent call last):\n  boom"


async def test_a_code_the_host_classifies_travels_as_the_error_code_with_the_detail_untouched():
    outputs = await _outputs_of(_failed_event("cuda_oom", "CUDA out of memory. Tried to allocate 2.00 GiB", detail="trace"))

    assert (outputs[0].error_code, outputs[0].detail) == ("cuda_oom", "trace")


def test_the_stored_failure_of_an_expired_package_carries_the_worker_code_for_the_administrator():
    output = worker_failure_output(JobErrorV1(code="expired", message="package expired before execution"))

    failure = failure_from_output(output)

    assert failure.error_code == "unclassified"
    assert failure.raw_error == "package expired before execution"
    assert failure.stored_detail == "package expired before execution\n\nWorker error code: expired"
    assert failure.admin_payload()["detail"].endswith("Worker error code: expired")


def test_the_stored_failure_of_a_worker_oom_keeps_the_text_classification_for_a_driver_error():
    output = worker_failure_output(JobErrorV1(code="pipe_failed", message="RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"))

    assert failure_from_output(output).error_code == "cuda_oom"

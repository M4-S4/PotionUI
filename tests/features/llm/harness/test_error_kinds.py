import pytest

from src.features.llm.harness.errors import (
    ERROR_KINDS,
    HarnessError,
    OutputTruncated,
    StructuredOutputFailed,
)


@pytest.mark.parametrize("error_type", [OutputTruncated, StructuredOutputFailed])
def test_the_new_failures_are_registered_under_their_kind(error_type):
    assert ERROR_KINDS[error_type.kind] is error_type
    assert error_type.retryable is False


@pytest.mark.parametrize("error_type", [OutputTruncated, StructuredOutputFailed])
def test_the_new_failures_survive_a_json_round_trip_as_their_own_type(error_type):
    original = error_type("cut off", status=200, code="length", retry_after=None, detail={"max_tokens": 1000, "retries": 1})
    restored = HarnessError.from_json(original.to_json())
    assert type(restored) is error_type
    assert restored == original
    assert restored.detail == {"max_tokens": 1000, "retries": 1}


def test_the_kinds_are_distinct():
    assert OutputTruncated.kind != StructuredOutputFailed.kind
    assert len({cls.kind for cls in ERROR_KINDS.values()}) == len(ERROR_KINDS)

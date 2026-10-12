"""Tests for the catalog actions of the `src.plugin_api.models` surface."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from src.features.models.exceptions import ModelAlreadyAssignedException, ModelAssignmentException, ModelNotFoundException
from src.features.models.records import Model
from src.plugin_api.models import (
    ProviderFetchException,
    assign_model_to_user,
    model_for_sha256,
    run_provider_fetch,
    update_model_metadata,
    visible_model_ids,
)

OPERATIONAL_FIELDS = {"location", "copies", "file_size", "sha256", "prompting_guidance", "indexed_at"}


def _model(**fields):
    defaults = dict(
        id="m1", filename="m.safetensors", model_type="lora", sha256="a" * 64, file_size=123,
        location={"root": "/depot", "path": "loras/m.safetensors"}, prompting_guidance="admin notes",
        model_metadata={"strength": 0.8},
    )
    defaults.update(fields)
    return Model(**defaults)


def test_model_for_sha256_normalises_the_hash_and_answers_the_non_admin_dict():
    repository = Mock(get_by_sha256=Mock(return_value=_model()))
    container = SimpleNamespace(model_repository=repository)

    with patch("src.plugin_api.models.get_container", return_value=container):
        result = model_for_sha256(" " + "A" * 64)
        assert model_for_sha256("") is None

    assert result["id"] == "m1"
    assert result["model_metadata"] == {"strength": 0.8}
    assert not OPERATIONAL_FIELDS & result.keys()
    repository.get_by_sha256.assert_called_once_with("a" * 64, include_providers=False)


def test_model_for_sha256_hands_the_operational_fields_to_an_admin_caller_only():
    repository = Mock(get_by_sha256=Mock(return_value=_model()))
    container = SimpleNamespace(model_repository=repository)

    with patch("src.plugin_api.models.get_container", return_value=container):
        result = model_for_sha256("a" * 64, admin=True)

    assert result["sha256"] == "a" * 64
    assert result["file_size"] == 123
    assert result["location"] == {"root": "/depot", "path": "loras/m.safetensors"}
    assert result["prompting_guidance"] == "admin notes"


def test_model_for_sha256_answers_none_for_an_unknown_hash():
    repository = Mock(get_by_sha256=Mock(return_value=None))
    container = SimpleNamespace(model_repository=repository)

    with patch("src.plugin_api.models.get_container", return_value=container):
        assert model_for_sha256("b" * 64) is None


def test_visible_model_ids_reads_the_users_assignments():
    repository = Mock(get_available_model_ids_for_user=Mock(return_value=["m1", "m2"]))
    container = SimpleNamespace(model_repository=repository)

    with patch("src.plugin_api.models.get_container", return_value=container):
        assert visible_model_ids("user-1") == ["m1", "m2"]
        assert visible_model_ids("") == []

    repository.get_available_model_ids_for_user.assert_called_once_with("user-1")


def test_assign_model_to_user_reports_a_new_assignment_and_an_existing_one():
    assignments = Mock(assign_model_to_user=Mock(side_effect=[{"id": "a1"}, ModelAlreadyAssignedException("already")]))
    container = SimpleNamespace(model_index_manager=SimpleNamespace(assignments=assignments))

    with patch("src.plugin_api.models.get_container", return_value=container):
        assert assign_model_to_user("m1", "user-1") is True
        assert assign_model_to_user("m1", "user-1") is False


def test_assign_model_to_user_lets_a_refusal_through():
    assignments = Mock(assign_model_to_user=Mock(side_effect=ModelAssignmentException("blocked")))
    container = SimpleNamespace(model_index_manager=SimpleNamespace(assignments=assignments))

    with patch("src.plugin_api.models.get_container", return_value=container):
        with pytest.raises(ModelAssignmentException):
            assign_model_to_user("m1", "user-1")


def test_update_model_metadata_merges_into_the_current_values():
    repository = Mock(get_by_id=Mock(return_value=_model()))
    editor = Mock()
    container = SimpleNamespace(model_repository=repository, model_index_manager=SimpleNamespace(metadata=editor))

    with patch("src.plugin_api.models.get_container", return_value=container):
        result = update_model_metadata("m1", {"mb_license": "CC0"})

    assert result == {"strength": 0.8, "mb_license": "CC0"}
    editor.update_model_metadata.assert_called_once_with("m1", {"strength": 0.8, "mb_license": "CC0"})


def test_update_model_metadata_writes_nothing_when_nothing_changes():
    repository = Mock(get_by_id=Mock(return_value=_model()))
    editor = Mock()
    container = SimpleNamespace(model_repository=repository, model_index_manager=SimpleNamespace(metadata=editor))

    with patch("src.plugin_api.models.get_container", return_value=container):
        assert update_model_metadata("m1", {"strength": 0.8}) == {"strength": 0.8}

    editor.update_model_metadata.assert_not_called()


def test_update_model_metadata_refuses_an_unknown_model():
    repository = Mock(get_by_id=Mock(return_value=None))
    container = SimpleNamespace(model_repository=repository, model_index_manager=SimpleNamespace(metadata=Mock()))

    with patch("src.plugin_api.models.get_container", return_value=container):
        with pytest.raises(ModelNotFoundException):
            update_model_metadata("missing", {"x": 1})


@pytest.mark.asyncio
async def test_run_provider_fetch_checks_the_provider_and_the_veto_hook_before_fetching_the_named_models():
    fetcher = Mock(
        fetch_provider_info=Mock(return_value={"status": "running"}),
        run_provider_fetch=AsyncMock(return_value={"successful": 1, "failed": 0, "errors": []}),
    )
    container = SimpleNamespace(model_index_manager=SimpleNamespace(provider_info=fetcher))

    with patch("src.plugin_api.models.get_container", return_value=container):
        summary = await run_provider_fetch("civitai", ["m1", ""])
        empty = await run_provider_fetch("civitai", [])

    assert summary == {"successful": 1, "failed": 0, "errors": []}
    assert empty == {"successful": 0, "failed": 0, "errors": []}
    assert fetcher.mock_calls == [
        call.fetch_provider_info("civitai", model_ids=["m1"]),
        call.run_provider_fetch("civitai", model_ids=["m1"]),
    ]


@pytest.mark.asyncio
async def test_run_provider_fetch_lets_the_providers_refusal_through_without_fetching():
    fetcher = Mock(
        fetch_provider_info=Mock(side_effect=ProviderFetchException("Provider 'nope' not found")),
        run_provider_fetch=AsyncMock(),
    )
    container = SimpleNamespace(model_index_manager=SimpleNamespace(provider_info=fetcher))

    with patch("src.plugin_api.models.get_container", return_value=container):
        with pytest.raises(ProviderFetchException):
            await run_provider_fetch("nope", ["m1"])

    fetcher.run_provider_fetch.assert_not_awaited()

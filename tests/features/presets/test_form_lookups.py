from unittest.mock import Mock, patch

import pytest

from src.features.fields.field_factory import FieldFactory
from src.features.models.records import Model
from src.features.models.repository import model_repo
from src.features.presets.configuration import resolve_field_filter_tags
from src.features.presets.form_lookups import current_form_lookups, form_lookups_scope
from src.features.presets.form_serializer import PresetFormSerializer
from src.features.presets.repository import preset_repo


def _model_field(name, recommendations, filter_tags=None):
    configuration = {"recommendations": recommendations}
    if filter_tags:
        configuration["filter_tags"] = filter_tags
    return {"type": "model", "name": name, "label": name, "configuration": configuration}


class _Form:
    def __init__(self, fields):
        self.fields = fields


@pytest.fixture
def serializer(mock_db):
    return PresetFormSerializer(Mock(), Mock())


@pytest.fixture
def installed_model(mock_db):
    return model_repo.create(
        Model(filename="alpha.safetensors", file_size=1, sha256="a" * 64, model_type="checkpoint")
    )


def _recommendations(*names):
    return [{"name": name, "link": f"https://example.invalid/{name}"} for name in names]


def test_form_resolves_configuration_and_filenames_once_per_scope(serializer, installed_model):
    form = _Form([
        _model_field("first", _recommendations("alpha", "beta"), "@config:base"),
        _model_field("second", _recommendations("alpha", "gamma"), "@config:base"),
    ])
    with patch.object(preset_repo, "get_preset_configuration", return_value={}) as configuration, \
            patch.object(model_repo, "existing_filenames", wraps=model_repo.existing_filenames) as filenames:
        schema = serializer.process_form_fields(form, "preset-1")

    assert configuration.call_count == 1
    assert filenames.call_count == 2
    first = schema["properties"]["first"]["configuration"]["recommendations"]
    second = schema["properties"]["second"]["configuration"]["recommendations"]
    assert [item["installed"] for item in first] == [True, False]
    assert [item["installed"] for item in second] == [True, False]


def test_calls_outside_the_scope_take_the_direct_path(mock_db):
    with patch.object(preset_repo, "get_preset_configuration", return_value={}) as configuration:
        resolve_field_filter_tags("@config:base", "preset-1")
        resolve_field_filter_tags("@config:base", "preset-1")

    assert configuration.call_count == 2


def test_scope_does_not_leak_past_its_block():
    with form_lookups_scope() as scoped:
        assert current_form_lookups() is scoped
    first = current_form_lookups()
    second = current_form_lookups()
    assert first is not scoped
    assert first is not second


def test_outside_scope_configuration_is_not_cached():
    with patch.object(preset_repo, "get_preset_configuration", side_effect=[{"a": 1}, {"a": 2}]):
        assert current_form_lookups().configuration("preset-1") == {"a": 1}
        assert current_form_lookups().configuration("preset-1") == {"a": 2}


def test_failed_configuration_lookup_returns_empty_and_is_retried():
    with form_lookups_scope() as scoped:
        with patch.object(preset_repo, "get_preset_configuration", side_effect=RuntimeError):
            assert scoped.configuration("preset-1") == {}
        with patch.object(preset_repo, "get_preset_configuration", return_value={"a": 1}):
            assert scoped.configuration("preset-1") == {"a": 1}


def test_scope_is_reset_when_the_block_raises():
    with pytest.raises(RuntimeError):
        with form_lookups_scope() as scoped:
            raise RuntimeError
    assert current_form_lookups() is not scoped


def test_failed_filename_lookup_is_not_cached():
    with form_lookups_scope() as scoped:
        with patch.object(model_repo, "existing_filenames", side_effect=RuntimeError):
            assert scoped.existing_filenames(["alpha"]) == set()
        with patch.object(model_repo, "existing_filenames", return_value={"alpha"}):
            assert scoped.existing_filenames(["alpha"]) == {"alpha"}


def test_factory_output_matches_without_a_scope(mock_db, installed_model):
    factory = FieldFactory(Mock(), Mock())
    field = _model_field("only", _recommendations("alpha", "beta"))

    outside = factory.map_field(field, "preset-1")
    serializer = PresetFormSerializer(Mock(), Mock())
    inside = serializer.process_form_fields(_Form([field]), "preset-1")["properties"]["only"]

    assert outside == inside

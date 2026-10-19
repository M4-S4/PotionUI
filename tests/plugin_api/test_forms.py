from src.features.fields.builtin import register_builtin_fields
from src.platform.plugins.field_types import FieldTypeRegistry
from src.plugin_api import downloads as api_downloads
from src.plugin_api import models as api_models
from src.plugin_api import presets as api_presets
from src.plugin_api.forms import core_field_type_names


def _registered():
    registry = FieldTypeRegistry()
    register_builtin_fields(registry)
    return registry.all()


def test_core_field_type_names_are_the_registered_non_container_types():
    expected = {d.type_name for d in _registered() if not d.container}
    assert core_field_type_names() == expected


def test_container_types_are_excluded():
    containers = {d.type_name for d in _registered() if d.container}
    assert containers
    assert not containers & core_field_type_names()


def test_types_the_hand_copied_lists_missed_are_included():
    assert {"cloud_options", "sampler", "schedule", "tags"} <= core_field_type_names()


def test_typed_refusals_and_backend_info_are_exported():
    from src.features.downloads.exceptions import DownloadDestinationInvalid, DownloadQueueException
    from src.features.models.locator import ModelFileUnavailable
    from src.features.presets.requirements.contracts import RequirementBackendInfo

    assert api_downloads.DownloadDestinationInvalid is DownloadDestinationInvalid
    assert issubclass(DownloadDestinationInvalid, DownloadQueueException)
    assert api_models.ModelFileUnavailable is ModelFileUnavailable
    assert api_presets.RequirementBackendInfo is RequirementBackendInfo

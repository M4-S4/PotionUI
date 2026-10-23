from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from src.features.forms.exceptions import FormNotFoundException
from src.features.presets.exceptions import (
    InvalidConfigurationException,
    InvalidFormOverridesException,
    InvalidModeDataException,
    InvalidUsersException,
    ModeNotFoundException,
    NoModesAvailableException,
    PermissionDeniedException,
    PresetAlreadyInstalledException,
    PresetNotAssignedException,
    PresetNotFoundException,
    PresetNotInstalledException,
    UserNotFoundException,
)
from src.features.presets.routes import PresetController

REFUSALS = [
    (PresetNotFoundException, "preset_not_found"),
    (ModeNotFoundException, "mode_not_found"),
    (FormNotFoundException, "form_not_found"),
    (NoModesAvailableException, "no_modes_available"),
    (PresetNotInstalledException, "preset_not_installed"),
    (PresetAlreadyInstalledException, "preset_already_installed"),
    (PresetNotAssignedException, "preset_not_assigned"),
    (UserNotFoundException, "user_not_found"),
    (InvalidUsersException, "invalid_users"),
    (PermissionDeniedException, "permission_denied"),
    (InvalidModeDataException, "invalid_mode_data"),
    (InvalidConfigurationException, "invalid_configuration"),
    (InvalidFormOverridesException, "invalid_form_overrides"),
]


def _raised(cls):
    exc = cls.__new__(cls)
    Exception.__init__(exc, f"{cls.__name__} text")
    return exc


@pytest.fixture
def controller():
    return PresetController(Mock(), Mock())


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,code", REFUSALS)
async def test_domain_refusals_answer_200_with_their_code(controller, cls, code):
    with patch("src.features.presets.routes.operations.get_available_modes", side_effect=_raised(cls)):
        response = await controller.get_available_modes("p")

    assert response.success is False
    assert response.error == code
    assert response.message == f"{cls.__name__} text"


@pytest.mark.asyncio
async def test_unexpected_failure_is_a_500_with_the_endpoint_message_only(controller):
    with patch("src.features.presets.routes.operations.get_available_modes", side_effect=RuntimeError("/srv/x leak")):
        with pytest.raises(HTTPException) as caught:
            await controller.get_available_modes("p")

    assert caught.value.status_code == 500
    assert caught.value.detail["error"] == "modes_get_failed"
    assert caught.value.detail["message"] == "Failed to get modes"


@pytest.mark.asyncio
async def test_http_exceptions_pass_through_unchanged(controller):
    original = HTTPException(status_code=403, detail="nope")
    with patch("src.features.presets.routes.operations.get_available_modes", side_effect=original):
        with pytest.raises(HTTPException) as caught:
            await controller.get_available_modes("p")

    assert caught.value is original
